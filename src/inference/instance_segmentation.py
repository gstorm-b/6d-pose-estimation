"""Standalone instance segmentation: scene points/depth -> per-point instance labels.

This is the instance stage of the pipeline in isolation - semantic foreground
prediction + per-point center voting + voted-center clustering - without the pose
stage. Use it to segment a bin into object instances (labels, clusters, centroids)
for visualization, debugging, or feeding a different downstream stage.

Loads the instance model either from a packaged bundle (`from_loaded_bundle`) or
directly from a training checkpoint (`from_checkpoint`), so it works with or
without the registry. Frames follow the rest of the codebase: scene points enter
in the positive-forward camera convention; results are reported in that frame.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.inference.instance_clustering import VotedCenterClusteringConfig, cluster_voted_centers


@dataclass
class SegmentedInstance:
    """One clustered object instance."""

    instance_id: int
    point_indices: np.ndarray  # indices into the result's points_camera
    point_count: int
    centroid_camera: np.ndarray  # (3,)
    bbox_min_camera: np.ndarray  # (3,)
    bbox_max_camera: np.ndarray  # (3,)
    mean_object_probability: float


@dataclass
class InstanceSegmentationResult:
    """Per-point labels plus per-instance summaries for one scene."""

    points_camera: np.ndarray            # (N, 3) the (possibly subsampled) scene points
    instance_labels: np.ndarray          # (N,) int, 0 = background, >0 = instance id
    object_probabilities: np.ndarray     # (N,) foreground probability
    voted_centers_camera: np.ndarray     # (N, 3) per-point voted object center
    instances: list[SegmentedInstance]
    timings_ms: dict[str, float]
    pixels: np.ndarray | None = None     # (N, 2) source (u, v) when segmenting a depth frame
    source_point_count: int = 0          # points before any subsampling

    @property
    def instance_count(self) -> int:
        return len(self.instances)


@dataclass
class SceneForward:
    """Cached model-forward output for a scene, so clustering can be re-run cheaply."""

    points_camera: np.ndarray         # (N, 3) subsampled scene points
    points_normalized: np.ndarray     # (N, 3) scene-centered points fed to clustering
    center: np.ndarray                # (1, 3) subtracted center
    object_probabilities: np.ndarray  # (N,)
    offsets: np.ndarray               # (N, 3) per-point center-vote offsets
    pixels: np.ndarray | None
    source_point_count: int
    inference_ms: float


class InstanceSegmenter:
    """Load the instance model once and segment scenes into object instances."""

    def __init__(
        self,
        *,
        instance_model: Any,
        clustering: VotedCenterClusteringConfig,
        num_classes: int = 2,
        scene_num_points: int = 16384,
        scene_subsample: str = "random",
        scene_voxel_size_m: float = 0.002,
        device: str = "cuda",
        seed: int = 7,
    ) -> None:
        self.instance_model = instance_model
        self.clustering = clustering
        self.num_classes = int(num_classes)
        self.scene_num_points = int(scene_num_points)
        if scene_subsample not in ("random", "voxel"):
            raise ValueError(f"unknown scene_subsample: {scene_subsample}")
        self.scene_subsample = scene_subsample
        self.scene_voxel_size_m = float(scene_voxel_size_m)
        self.device = device
        self.seed = int(seed)

    @classmethod
    def from_loaded_bundle(cls, loaded: Any, *, seed: int = 7) -> "InstanceSegmenter":
        inference = loaded.bundle.manifest.inference
        clustering = VotedCenterClusteringConfig.from_mapping(
            {
                "object_probability_threshold": inference.get("object_probability_threshold", 0.5),
                "dbscan_eps_m": inference.get("dbscan_eps_m", 0.005),
                "dbscan_min_samples": inference.get("dbscan_min_samples", 8),
                "min_cluster_points": inference.get("min_cluster_points", 64),
            }
        )
        num_classes = int(loaded.instance_config.get("model", {}).get("num_classes", 2))
        return cls(
            instance_model=loaded.instance_model,
            clustering=clustering,
            num_classes=num_classes,
            scene_num_points=int(inference.get("num_points", 16384)),
            scene_subsample=str(inference.get("scene_subsample", "random")),
            scene_voxel_size_m=float(inference.get("scene_voxel_size_m", 0.002)),
            device=loaded.device,
            seed=seed,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str = "cuda",
        ops_backend: str = "pytorch3d",
        clustering: dict[str, Any] | None = None,
        scene_num_points: int = 16384,
        seed: int = 7,
    ) -> "InstanceSegmenter":
        """Build a segmenter directly from an instance-seg training checkpoint."""

        from src.inference.device import resolve_device
        from src.models.pointnet2_instance_seg import build_pointnet2_instance_seg_from_config
        from src.training.checkpoint import load_checkpoint

        resolved_device = resolve_device(device)
        checkpoint = load_checkpoint(Path(checkpoint_path), map_location="cpu")
        config = dict(checkpoint.get("config", {}))
        if ops_backend:
            config.setdefault("model", {})["ops_backend"] = ops_backend
        model = build_pointnet2_instance_seg_from_config(config)
        model.load_state_dict(checkpoint["model_state"])
        model.to(resolved_device).eval()
        inference_config = config.get("inference") or {}
        clustering_config = VotedCenterClusteringConfig.from_mapping(clustering or inference_config)
        return cls(
            instance_model=model,
            clustering=clustering_config,
            num_classes=int(config.get("model", {}).get("num_classes", 2)),
            scene_num_points=scene_num_points,
            scene_subsample=str(inference_config.get("scene_subsample", "random")),
            scene_voxel_size_m=float(inference_config.get("scene_voxel_size_m", 0.002)),
            device=resolved_device,
            seed=seed,
        )

    def _subsample(self, points: np.ndarray, features: np.ndarray | None, pixels: np.ndarray | None):
        if points.shape[0] <= self.scene_num_points:
            return points, features, pixels
        rng = np.random.default_rng(self.seed)
        if self.scene_subsample == "voxel":
            from src.data.pointnet2_processing import voxel_sample_indices

            keep = voxel_sample_indices(
                points, num_points=self.scene_num_points, voxel_size_m=self.scene_voxel_size_m, rng=rng
            )
        else:
            keep = rng.choice(points.shape[0], size=self.scene_num_points, replace=False)
        return (
            points[keep],
            None if features is None else features[keep],
            None if pixels is None else pixels[keep],
        )

    def forward(
        self,
        points_camera: np.ndarray,
        features: np.ndarray | None = None,
        *,
        pixels: np.ndarray | None = None,
    ) -> SceneForward:
        """Run the instance model on a scene (the slow GPU step). Cache + re-cluster cheaply."""

        import torch

        points = np.asarray(points_camera, dtype=np.float32)
        feats = None if features is None else np.asarray(features, dtype=np.float32)
        source_count = points.shape[0]
        points, feats, pixels = self._subsample(points, feats, pixels)

        t0 = time.perf_counter()
        center = points.mean(axis=0, keepdims=True).astype(np.float32)
        points_normalized = points - center
        model_input = points_normalized if feats is None else np.concatenate([points_normalized, feats], axis=1)
        tensor = torch.from_numpy(model_input.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.instance_model(tensor)
            probabilities = torch.softmax(outputs["semantic_logits"], dim=-1).squeeze(0).cpu().numpy()
            offsets = outputs["offsets"].squeeze(0).cpu().numpy().astype(np.float32)
        object_probabilities = (
            probabilities[:, 1] if self.num_classes > 1 else np.zeros(points.shape[0], dtype=np.float32)
        ).astype(np.float32)
        return SceneForward(
            points_camera=points,
            points_normalized=points_normalized.astype(np.float32),
            center=center,
            object_probabilities=object_probabilities,
            offsets=offsets,
            pixels=pixels,
            source_point_count=source_count,
            inference_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def cluster(
        self,
        forward: SceneForward,
        clustering: VotedCenterClusteringConfig | None = None,
    ) -> InstanceSegmentationResult:
        """Cluster a cached forward pass into instances (the fast CPU step)."""

        config = clustering or self.clustering
        t1 = time.perf_counter()
        clustered = cluster_voted_centers(
            forward.points_normalized, forward.object_probabilities, forward.offsets, config
        )
        clustering_ms = (time.perf_counter() - t1) * 1000.0

        points = forward.points_camera
        labels = np.asarray(clustered["instance_labels"], dtype=np.int64)
        voted_centers_camera = np.asarray(clustered["voted_centers"], dtype=np.float32) + forward.center

        instances: list[SegmentedInstance] = []
        for instance_id in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
            idx = np.flatnonzero(labels == instance_id)
            cluster_points = points[idx]
            instances.append(
                SegmentedInstance(
                    instance_id=instance_id,
                    point_indices=idx,
                    point_count=int(idx.shape[0]),
                    centroid_camera=cluster_points.mean(axis=0),
                    bbox_min_camera=cluster_points.min(axis=0),
                    bbox_max_camera=cluster_points.max(axis=0),
                    mean_object_probability=float(forward.object_probabilities[idx].mean()),
                )
            )
        instances.sort(key=lambda inst: inst.point_count, reverse=True)
        return InstanceSegmentationResult(
            points_camera=points,
            instance_labels=labels,
            object_probabilities=forward.object_probabilities,
            voted_centers_camera=voted_centers_camera,
            instances=instances,
            timings_ms={"inference_ms": forward.inference_ms, "clustering_ms": clustering_ms},
            pixels=forward.pixels,
            source_point_count=forward.source_point_count,
        )

    def segment_points(
        self,
        points_camera: np.ndarray,
        features: np.ndarray | None = None,
        *,
        pixels: np.ndarray | None = None,
    ) -> InstanceSegmentationResult:
        """Segment a scene point cloud (positive-forward camera frame) into instances."""

        return self.cluster(self.forward(points_camera, features, pixels=pixels))

    def segment_depth(
        self,
        depth_m: np.ndarray,
        intrinsics: dict[str, Any],
        normal_camera: np.ndarray | None = None,
    ) -> InstanceSegmentationResult:
        """Unproject a depth frame and segment it; keeps source pixels for masks."""

        from src.data.depth_unprojection import unproject_depth_to_camera

        points, pixels = unproject_depth_to_camera(depth_m, intrinsics)
        features = None
        if normal_camera is not None:
            normals = np.asarray(normal_camera, dtype=np.float32)
            features = normals[pixels[:, 1].astype(np.int64), pixels[:, 0].astype(np.int64)]
        return self.segment_points(points, features, pixels=pixels)
