"""End-to-end, in-memory pose pipeline: scene points -> ranked object poses.

Reuses the same building blocks as the offline export+eval path (instance
segmentation, voted-center clustering, pose-crop feature construction, and the
voting pose decode) so a library/service call produces the same poses as the
research pipeline, without writing any intermediate files.

Frames: scene points enter in the positive-forward depth convention (as produced
by depth unprojection / the processed dataset). Crops are flipped into the
Blender metadata camera convention before the pose stage, exactly as
`PoseCropDataset` does, and `object_to_camera` is returned in that convention.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.data.depth_unprojection import unproject_depth_to_camera
from src.data.pose_crop_dataset import build_pose_model_features, positive_forward_depth_to_metadata_camera
from src.inference.confidence import ConfidenceConfig, cluster_isolation, compute_instance_confidence
from src.inference.instance_clustering import (
    TwoStageClusteringConfig,
    VotedCenterClusteringConfig,
    cluster_voted_centers,
    refine_instance_labels_two_stage,
)
from src.inference.pose_bridge import build_pose_instance_crops
from src.training.pose_metrics import pose_predictions_to_object_to_camera


@dataclass
class ScenePoseInstance:
    """One object instance pose decoded from the scene."""

    instance_id: int
    object_to_camera: np.ndarray  # (4, 4), metadata camera frame
    point_count: int
    crop_centroid_camera: np.ndarray
    matched_gt_iou: float | None = None
    confidence: float | None = None
    point_indices: np.ndarray | None = None  # indices into the scene points fed to infer_from_points
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenePoseResult:
    instances: list[ScenePoseInstance]
    timings_ms: dict[str, float]
    warnings: list[str] = field(default_factory=list)


class PosePipeline:
    """Load instance + pose models once and run scene -> poses in memory."""

    def __init__(
        self,
        *,
        instance_model: Any,
        pose_model: Any,
        model_points_object: np.ndarray,
        keypoints_object: np.ndarray,
        diameter_m: float,
        clustering: VotedCenterClusteringConfig,
        two_stage: TwoStageClusteringConfig | None = None,
        confidence: ConfidenceConfig | None = None,
        instance_num_classes: int = 2,
        pose_num_points: int = 1024,
        scene_num_points: int = 16384,
        device: str = "cuda",
        seed: int = 7,
    ) -> None:
        self.instance_model = instance_model
        self.pose_model = pose_model
        self.model_points_object = np.asarray(model_points_object, dtype=np.float32)
        self.keypoints_object = np.asarray(keypoints_object, dtype=np.float32)
        self.diameter_m = float(diameter_m)
        self.clustering = clustering
        self.two_stage = two_stage or TwoStageClusteringConfig()
        self.confidence_config = confidence or ConfidenceConfig()
        self.instance_num_classes = int(instance_num_classes)
        self.pose_num_points = int(pose_num_points)
        self.scene_num_points = int(scene_num_points)
        self.device = device
        self.seed = int(seed)
        self.model_fit_max_points = 2048  # cap crop points used for the model-fit chamfer

    @classmethod
    def from_loaded_bundle(cls, loaded: Any, *, seed: int = 7) -> "PosePipeline":
        manifest = loaded.bundle.manifest
        inference = manifest.inference
        clustering = VotedCenterClusteringConfig.from_mapping(
            {
                "object_probability_threshold": inference.get("object_probability_threshold", 0.5),
                "dbscan_eps_m": inference.get("dbscan_eps_m", 0.005),
                "dbscan_min_samples": inference.get("dbscan_min_samples", 8),
                "min_cluster_points": inference.get("min_cluster_points", 64),
            }
        )
        pose_num_points = int(loaded.pose_config.get("model", {}).get("num_points", 1024))
        instance_num_classes = int(loaded.instance_config.get("model", {}).get("num_classes", 2))
        return cls(
            instance_model=loaded.instance_model,
            pose_model=loaded.pose_model,
            model_points_object=loaded.model_points_object,
            keypoints_object=loaded.keypoints_object,
            diameter_m=loaded.diameter_m,
            clustering=clustering,
            two_stage=TwoStageClusteringConfig.from_mapping(inference.get("two_stage_clustering")),
            confidence=ConfidenceConfig.from_mapping(inference.get("confidence")),
            instance_num_classes=instance_num_classes,
            pose_num_points=pose_num_points,
            scene_num_points=int(inference.get("num_points", 16384)),
            device=loaded.device,
            seed=seed,
        )

    def infer_scene(
        self,
        depth_m: np.ndarray,
        intrinsics: dict[str, Any],
        normal_camera: np.ndarray | None = None,
        *,
        options: Any = None,
    ) -> ScenePoseResult:
        """Unproject a depth frame and run the pipeline.

        `normal_camera` is an optional (H, W, 3) per-pixel normal map; when given,
        normals are looked up at the valid pixels to form the model features.
        """

        points, pixels = unproject_depth_to_camera(depth_m, intrinsics)
        features = None
        if normal_camera is not None:
            normals = np.asarray(normal_camera, dtype=np.float32)
            features = normals[pixels[:, 1].astype(np.int64), pixels[:, 0].astype(np.int64)]
        if points.shape[0] > self.scene_num_points:
            rng = np.random.default_rng(self.seed)
            keep = rng.choice(points.shape[0], size=self.scene_num_points, replace=False)
            points = points[keep]
            features = None if features is None else features[keep]
        return self.infer_from_points(points, features, options=options)

    def warmup(self) -> None:
        points = np.zeros((self.pose_num_points, 3), dtype=np.float32)
        features = np.zeros((self.pose_num_points, 3), dtype=np.float32)
        labels = np.ones(self.pose_num_points, dtype=np.int64)
        self.infer_from_points(points + 0.05, features, gt_instance_labels=labels)

    def predict_instance_labels(self, points_camera: np.ndarray, features: np.ndarray | None) -> np.ndarray:
        import torch

        points = np.asarray(points_camera, dtype=np.float32)
        center = points.mean(axis=0, keepdims=True).astype(np.float32)
        points_normalized = points - center
        if features is not None:
            model_input = np.concatenate([points_normalized, np.asarray(features, dtype=np.float32)], axis=1)
        else:
            model_input = points_normalized
        tensor = torch.from_numpy(model_input.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.instance_model(tensor)
            probabilities = torch.softmax(outputs["semantic_logits"], dim=-1).squeeze(0).cpu().numpy()
            offsets = outputs["offsets"].squeeze(0).cpu().numpy().astype(np.float32)
        object_probabilities = (
            probabilities[:, 1] if self.instance_num_classes > 1 else np.zeros(points.shape[0], dtype=np.float32)
        )
        clustered = cluster_voted_centers(points_normalized, object_probabilities, offsets, self.clustering)
        labels = np.asarray(clustered["instance_labels"], dtype=np.int64)
        if self.two_stage.enabled:
            labels = refine_instance_labels_two_stage(
                points_normalized, np.asarray(clustered["voted_centers"], dtype=np.float32), labels, self.diameter_m, self.two_stage
            )
        return labels

    def _sample_indices(self, count: int, rng: np.random.Generator) -> np.ndarray:
        if count <= 0:
            return np.zeros(0, dtype=np.int64)
        replace = count < self.pose_num_points
        return rng.choice(count, size=self.pose_num_points, replace=replace).astype(np.int64)

    def _decode_crop_pose(self, crop: Any, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        import torch

        # Flip crop points/normals from positive-forward depth into metadata camera frame.
        points_camera = positive_forward_depth_to_metadata_camera(np.asarray(crop.points_camera, dtype=np.float32))
        features = None
        if crop.features is not None and np.asarray(crop.features).shape[1] >= 3:
            features = positive_forward_depth_to_metadata_camera(np.asarray(crop.features, dtype=np.float32))
        centroid = points_camera.mean(axis=0).astype(np.float32)

        indices = self._sample_indices(points_camera.shape[0], rng)
        sampled_points = points_camera[indices]
        sampled_features = None if features is None else features[indices]
        points_pose_input, crop_aux_features, _bmin, _bmax = build_pose_model_features(
            all_crop_points_camera=points_camera,
            sampled_points_camera=sampled_points,
            sampled_features=sampled_features,
            crop_centroid_camera=centroid,
            model_diameter_m=self.diameter_m,
        )

        pose_input = torch.from_numpy(points_pose_input.astype(np.float32)).unsqueeze(0).to(self.device)
        aux = torch.from_numpy(crop_aux_features.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.pose_model(pose_input, aux)
        batch = {
            "model_diameter_m": torch.tensor([self.diameter_m], dtype=torch.float32, device=self.device),
            "sampled_points_camera": torch.from_numpy(sampled_points.astype(np.float32)).unsqueeze(0).to(self.device),
            "crop_centroid_camera": torch.from_numpy(centroid.astype(np.float32)).unsqueeze(0).to(self.device),
            "keypoints_object": torch.from_numpy(self.keypoints_object.astype(np.float32)).unsqueeze(0).to(self.device),
        }
        decoded = pose_predictions_to_object_to_camera(outputs, batch)
        rotation = decoded["rotation_object_to_camera"].squeeze(0).cpu().numpy().astype(np.float64)
        origin = decoded["object_origin_camera"].squeeze(0).cpu().numpy().astype(np.float64)

        # Vote-dispersion diagnostics: how much the per-point votes disagree,
        # normalized by diameter. Low dispersion -> confident pose.
        diameter = max(self.diameter_m, 1e-8)
        diagnostics: dict[str, float] = {}
        if "predicted_origin_offsets_camera_normalized" in outputs:
            origin_offsets = outputs["predicted_origin_offsets_camera_normalized"].squeeze(0).cpu().numpy()
            origin_votes = sampled_points + origin_offsets * diameter
            diagnostics["center_vote_dispersion"] = float(np.linalg.norm(origin_votes.std(axis=0))) / diameter
        if "predicted_keypoint_offsets_camera_normalized" in outputs:
            keypoint_offsets = outputs["predicted_keypoint_offsets_camera_normalized"].squeeze(0).cpu().numpy()
            keypoint_votes = sampled_points[:, None, :] + keypoint_offsets * diameter
            diagnostics["keypoint_dispersion"] = float(np.mean(np.linalg.norm(keypoint_votes.std(axis=0), axis=1))) / diameter

        # Model-fit: mean distance from each observed crop point to the posed model
        # surface, normalized by diameter. A wrong pose does not land on its own crop,
        # so this is the strongest correct-vs-wrong signal on predicted-cluster crops.
        diagnostics["model_fit"] = self._model_fit(points_camera, rotation, origin)
        return rotation, origin, diagnostics

    def _model_fit(self, crop_points_camera: np.ndarray, rotation: np.ndarray, origin: np.ndarray) -> float:
        import torch

        crop = np.asarray(crop_points_camera, dtype=np.float32)
        if crop.shape[0] == 0:
            return float("inf")
        if crop.shape[0] > self.model_fit_max_points:
            idx = np.random.default_rng(self.seed).choice(crop.shape[0], self.model_fit_max_points, replace=False)
            crop = crop[idx]
        with torch.no_grad():
            model = torch.from_numpy(self.model_points_object.astype(np.float32)).to(self.device)
            rot = torch.from_numpy(rotation.astype(np.float32)).to(self.device)
            trans = torch.from_numpy(origin.astype(np.float32)).to(self.device)
            model_camera = model @ rot.T + trans
            crop_tensor = torch.from_numpy(crop).to(self.device)
            nearest = torch.cdist(crop_tensor, model_camera).min(dim=1).values
            fit = float(nearest.mean().item())
        return fit / max(self.diameter_m, 1e-8)

    def infer_from_points(
        self,
        points_camera: np.ndarray,
        features: np.ndarray | None = None,
        *,
        options: Any = None,
        gt_instance_labels: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ScenePoseResult:
        """Run instance + pose stages on scene points (positive-forward frame).

        Pass `gt_instance_labels` to bypass clustering (used for verification);
        otherwise instance segmentation + clustering produce the labels.
        """

        timings: dict[str, float] = {}
        warnings: list[str] = []
        points = np.asarray(points_camera, dtype=np.float32)
        feats = None if features is None else np.asarray(features, dtype=np.float32)

        t0 = time.perf_counter()
        if gt_instance_labels is not None:
            labels = np.asarray(gt_instance_labels, dtype=np.int64)
            source = "gt"
        else:
            labels = self.predict_instance_labels(points, feats)
            source = "predicted"
        timings["instance_ms"] = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        crops = build_pose_instance_crops(
            sample_name="scene",
            source=source,
            points_camera=points,
            instance_labels=labels,
            point_features=feats,
            target_instance_labels=labels if source == "gt" else None,
            metadata=metadata,
            min_points=int(self.clustering.min_cluster_points) if source == "predicted" else 1,
        )
        timings["crops_ms"] = (time.perf_counter() - t1) * 1000.0

        rng = np.random.default_rng(self.seed)
        instances: list[ScenePoseInstance] = []
        t2 = time.perf_counter()
        for crop in crops:
            rotation, origin, diagnostics = self._decode_crop_pose(crop, rng)
            object_to_camera = np.eye(4, dtype=np.float64)
            object_to_camera[:3, :3] = rotation
            object_to_camera[:3, 3] = origin
            instances.append(
                ScenePoseInstance(
                    instance_id=int(crop.instance_id),
                    object_to_camera=object_to_camera,
                    point_count=int(crop.point_count),
                    crop_centroid_camera=positive_forward_depth_to_metadata_camera(
                        np.asarray(crop.points_camera, dtype=np.float32)
                    ).mean(axis=0),
                    matched_gt_iou=crop.matched_gt_iou,
                    point_indices=np.asarray(crop.point_indices, dtype=np.int64),
                    diagnostics=dict(diagnostics),
                )
            )
        timings["pose_ms"] = (time.perf_counter() - t2) * 1000.0

        # Confidence scoring: add cluster isolation, then rank highest-first.
        if instances:
            centroids = np.stack([inst.crop_centroid_camera for inst in instances], axis=0)
            isolation = cluster_isolation(centroids, self.diameter_m)
            for inst, iso in zip(instances, isolation):
                features = dict(inst.diagnostics)
                features["point_count"] = inst.point_count
                features["cluster_isolation"] = None if not np.isfinite(iso) else float(iso)
                confidence, sub_scores = compute_instance_confidence(features, self.confidence_config)
                inst.confidence = confidence
                inst.diagnostics["cluster_isolation"] = float(iso) if np.isfinite(iso) else None
                inst.diagnostics["confidence_terms"] = sub_scores
            instances.sort(key=lambda inst: (inst.confidence if inst.confidence is not None else -1.0), reverse=True)

        max_model_fit = getattr(options, "max_model_fit", None) if options is not None else None
        if max_model_fit is not None:
            instances = [
                inst for inst in instances
                if inst.diagnostics.get("model_fit") is None or inst.diagnostics["model_fit"] <= float(max_model_fit)
            ]
        min_confidence = getattr(options, "min_confidence", None) if options is not None else None
        if min_confidence is not None:
            instances = [inst for inst in instances if (inst.confidence or 0.0) >= float(min_confidence)]
        max_instances = getattr(options, "max_instances", None) if options is not None else None
        if max_instances is not None and len(instances) > int(max_instances):
            instances = instances[: int(max_instances)]
        return ScenePoseResult(instances=instances, timings_ms=timings, warnings=warnings)
