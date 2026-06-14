"""Dataset for pose estimation from exported instance crops."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.training.pose_geometry import (
    load_stl_vertices_m,
    model_diameter_m,
    resolve_symmetry_matrices,
    sample_model_points,
)

try:
    import torch
    from torch.utils.data import Dataset

    TORCH_AVAILABLE = True
except Exception:  # noqa: BLE001 - keep imports usable before training deps are installed.
    torch = None
    Dataset = object  # type: ignore[assignment,misc]
    TORCH_AVAILABLE = False


@dataclass(frozen=True)
class PoseModelGeometry:
    model_path: Path
    model_scale: float
    model_diameter_m: float
    model_points_object: np.ndarray
    keypoints_object: np.ndarray
    symmetry_matrices_object: np.ndarray
    symmetry_name: str


class PoseCropDataset(Dataset):  # type: ignore[misc]
    """Load per-instance crops exported by `export_pose_instance_crops.py`."""

    def __init__(
        self,
        root: str | Path,
        *,
        num_points: int = 1024,
        model_path: str | Path,
        model_scale: float = 1.0,
        model_point_count: int = 1024,
        keypoint_count: int = 8,
        symmetry_name: str = "none",
        min_matched_gt_iou: float | None = None,
        require_pose: bool = True,
        seed: int = 7,
        pose_augmentation: dict[str, Any] | None = None,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use PoseCropDataset.")
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.num_points = int(num_points)
        self.require_pose = bool(require_pose)
        self.min_matched_gt_iou = min_matched_gt_iou
        self.seed = int(seed)
        self.pose_augmentation = pose_augmentation or {}
        self.augmentation_repeats = max(1, int(self.pose_augmentation.get("repeats", 1)))
        vertices = load_stl_vertices_m(model_path, scale=float(model_scale))
        self.geometry = PoseModelGeometry(
            model_path=Path(model_path),
            model_scale=float(model_scale),
            model_diameter_m=model_diameter_m(vertices),
            model_points_object=sample_model_points(vertices, int(model_point_count), seed=seed),
            keypoints_object=farthest_point_keypoints(vertices, int(keypoint_count), seed=seed),
            symmetry_matrices_object=resolve_symmetry_matrices(symmetry_name),
            symmetry_name=symmetry_name,
        )
        self.records = self._build_records()
        if self.augmentation_repeats > 1:
            base_records = self.records
            self.records = []
            for augment_index in range(self.augmentation_repeats):
                for record in base_records:
                    repeated = dict(record)
                    repeated["augment_index"] = augment_index
                    self.records.append(repeated)
        if not self.records:
            raise ValueError(f"No pose crops available after filtering: {self.root}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        augment_index = int(record.get("augment_index", 0))
        with np.load(record["crop_file"]) as data:
            crop_offsets = np.asarray(data["crop_offsets"], dtype=np.int64)
            start, end = crop_offsets[record["crop_index"]]
            crop_point_count = int(end - start)
            points_depth_camera = np.asarray(data["crop_points_camera"][start:end], dtype=np.float32)
            points_camera = positive_forward_depth_to_metadata_camera(points_depth_camera)
            crop_features = None
            if "crop_features" in data.files and data["crop_features"].shape[1] > 0:
                crop_features_depth_camera = np.asarray(data["crop_features"][start:end], dtype=np.float32)
                crop_features = positive_forward_depth_to_metadata_camera(crop_features_depth_camera)
            point_indices = np.asarray(data["crop_point_indices"][start:end], dtype=np.int64)
            crop_centroid_depth_camera = np.asarray(data["crop_centroids_camera"][record["crop_index"]], dtype=np.float32)
            crop_centroid_camera = positive_forward_depth_to_metadata_camera(crop_centroid_depth_camera.reshape(1, 3))[0]
            raw_object_to_camera = np.asarray(data["crop_object_to_camera"][record["crop_index"]], dtype=np.float32)
            object_to_camera = rigid_object_to_camera_from_scaled(raw_object_to_camera)
            matched_gt_iou = float(np.asarray(data["crop_matched_gt_ious"][record["crop_index"]], dtype=np.float32))
            matched_gt_instance_id = int(np.asarray(data["crop_matched_gt_instance_ids"][record["crop_index"]], dtype=np.int64))
            instance_id = int(np.asarray(data["crop_instance_ids"][record["crop_index"]], dtype=np.int64))

        sampled_indices = self._sample_indices(points_camera.shape[0], index)
        sampled_points_camera = points_camera[sampled_indices]
        sampled_features = None if crop_features is None else crop_features[sampled_indices]
        sampled_point_indices = point_indices[sampled_indices]
        if self.pose_augmentation and augment_index > 0:
            sampled_points_camera, points_camera, crop_centroid_camera, object_to_camera, sampled_features, crop_features = (
                self._apply_pose_augmentation(
                    sampled_points_camera=sampled_points_camera,
                    all_points_camera=points_camera,
                    crop_centroid_camera=crop_centroid_camera,
                    object_to_camera=object_to_camera,
                    sampled_features=sampled_features,
                    all_features=crop_features,
                    sample_index=index,
                    augment_index=augment_index,
                )
            )
        points_crop_centered_camera = sampled_points_camera - crop_centroid_camera.reshape(1, 3)
        points_pose_xyz = points_crop_centered_camera / max(self.geometry.model_diameter_m, 1e-8)
        points_pose_input = points_pose_xyz
        if sampled_features is not None:
            points_pose_input = np.concatenate([points_pose_input, sampled_features.astype(np.float32)], axis=1)
        bbox_min_camera = points_camera.min(axis=0).astype(np.float32)
        bbox_max_camera = points_camera.max(axis=0).astype(np.float32)
        normalized_crop_points = points_pose_xyz.astype(np.float32)
        second_moment = np.mean(normalized_crop_points * normalized_crop_points, axis=0)
        covariance_cross = np.asarray(
            [
                np.mean(normalized_crop_points[:, 0] * normalized_crop_points[:, 1]),
                np.mean(normalized_crop_points[:, 0] * normalized_crop_points[:, 2]),
                np.mean(normalized_crop_points[:, 1] * normalized_crop_points[:, 2]),
            ],
            dtype=np.float32,
        )
        third_moment = np.mean(normalized_crop_points * normalized_crop_points * normalized_crop_points, axis=0)
        crop_aux_features = np.concatenate(
            [
                crop_centroid_camera / max(self.geometry.model_diameter_m, 1e-8),
                (bbox_min_camera - crop_centroid_camera) / max(self.geometry.model_diameter_m, 1e-8),
                (bbox_max_camera - crop_centroid_camera) / max(self.geometry.model_diameter_m, 1e-8),
                second_moment,
                covariance_cross,
                third_moment,
            ],
            axis=0,
        ).astype(np.float32)
        rotation_object_to_camera = object_to_camera[:3, :3]
        object_origin_camera = object_to_camera[:3, 3]
        offset_camera = object_origin_camera - crop_centroid_camera
        offset_normalized = offset_camera / max(self.geometry.model_diameter_m, 1e-8)
        sampled_points_object = (sampled_points_camera - object_origin_camera.reshape(1, 3)) @ rotation_object_to_camera
        sampled_points_object_normalized = sampled_points_object / max(self.geometry.model_diameter_m, 1e-8)
        sampled_origin_offsets_camera = object_origin_camera.reshape(1, 3) - sampled_points_camera
        sampled_origin_offsets_camera_normalized = sampled_origin_offsets_camera / max(self.geometry.model_diameter_m, 1e-8)
        keypoints_object = self.geometry.keypoints_object.astype(np.float32)
        keypoints_camera = keypoints_object @ rotation_object_to_camera.T + object_origin_camera.reshape(1, 3)
        sampled_keypoint_offsets_camera = keypoints_camera.reshape(1, -1, 3) - sampled_points_camera.reshape(-1, 1, 3)
        sampled_keypoint_offsets_camera_normalized = sampled_keypoint_offsets_camera / max(self.geometry.model_diameter_m, 1e-8)
        return {
            "points_crop_centered_camera": torch.from_numpy(points_crop_centered_camera.astype(np.float32)),
            "sampled_points_camera": torch.from_numpy(sampled_points_camera.astype(np.float32)),
            "points_pose_input": torch.from_numpy(points_pose_input.astype(np.float32)),
            "crop_aux_features": torch.from_numpy(crop_aux_features),
            "sampled_point_indices": torch.from_numpy(sampled_point_indices.astype(np.int64)),
            "crop_centroid_camera": torch.from_numpy(crop_centroid_camera.astype(np.float32)),
            "crop_centroid_positive_forward_depth_camera": torch.from_numpy(crop_centroid_depth_camera.astype(np.float32)),
            "object_to_camera": torch.from_numpy(object_to_camera.astype(np.float32)),
            "raw_object_to_camera": torch.from_numpy(raw_object_to_camera.astype(np.float32)),
            "rotation_object_to_camera": torch.from_numpy(rotation_object_to_camera.astype(np.float32)),
            "object_origin_camera": torch.from_numpy(object_origin_camera.astype(np.float32)),
            "object_origin_offset_from_crop_centroid_camera": torch.from_numpy(offset_camera.astype(np.float32)),
            "object_origin_offset_from_crop_centroid_camera_normalized": torch.from_numpy(offset_normalized.astype(np.float32)),
            "sampled_points_object": torch.from_numpy(sampled_points_object.astype(np.float32)),
            "sampled_points_object_normalized": torch.from_numpy(sampled_points_object_normalized.astype(np.float32)),
            "sampled_origin_offsets_camera": torch.from_numpy(sampled_origin_offsets_camera.astype(np.float32)),
            "sampled_origin_offsets_camera_normalized": torch.from_numpy(sampled_origin_offsets_camera_normalized.astype(np.float32)),
            "keypoints_object": torch.from_numpy(keypoints_object.astype(np.float32)),
            "keypoints_camera": torch.from_numpy(keypoints_camera.astype(np.float32)),
            "sampled_keypoint_offsets_camera_normalized": torch.from_numpy(sampled_keypoint_offsets_camera_normalized.astype(np.float32)),
            "crop_bbox_min_camera": torch.from_numpy(bbox_min_camera.astype(np.float32)),
            "crop_bbox_max_camera": torch.from_numpy(bbox_max_camera.astype(np.float32)),
            "model_diameter_m": torch.tensor(self.geometry.model_diameter_m, dtype=torch.float32),
            "model_points_object": torch.from_numpy(self.geometry.model_points_object.astype(np.float32)),
            "symmetry_matrices_object": torch.from_numpy(self.geometry.symmetry_matrices_object.astype(np.float32)),
            "matched_gt_iou": torch.tensor(matched_gt_iou, dtype=torch.float32),
            "crop_point_count": torch.tensor(crop_point_count, dtype=torch.int64),
            "matched_gt_instance_id": torch.tensor(matched_gt_instance_id, dtype=torch.int64),
            "instance_id": torch.tensor(instance_id, dtype=torch.int64),
            "crop_index": torch.tensor(record["crop_index"], dtype=torch.int64),
            "sample_name": record["sample_name"],
            "source": self.manifest.get("source", "unknown"),
            "symmetry_name": self.geometry.symmetry_name,
            "crop_file": str(record["crop_file"]),
        }

    def _build_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for sample in self.manifest.get("samples", []):
            crop_file = self.root / sample["crop_file"]
            with np.load(crop_file) as data:
                object_to_camera = np.asarray(data["crop_object_to_camera"], dtype=np.float32)
                matched_gt_ious = np.asarray(data["crop_matched_gt_ious"], dtype=np.float32)
                crop_count = int(data["crop_instance_ids"].shape[0])
                for crop_index in range(crop_count):
                    has_pose = bool(np.isfinite(object_to_camera[crop_index]).all())
                    if self.require_pose and not has_pose:
                        continue
                    matched_gt_iou = float(matched_gt_ious[crop_index])
                    if self.min_matched_gt_iou is not None and (not np.isfinite(matched_gt_iou) or matched_gt_iou < self.min_matched_gt_iou):
                        continue
                    records.append(
                        {
                            "sample_name": sample["sample_name"],
                            "crop_file": crop_file,
                            "crop_index": crop_index,
                        }
                    )
        return records

    def _sample_indices(self, point_count: int, index: int) -> np.ndarray:
        if point_count <= 0:
            raise ValueError("pose crop has no points")
        rng = np.random.default_rng(self.seed + int(index))
        replace = point_count < self.num_points
        return rng.choice(point_count, size=self.num_points, replace=replace).astype(np.int64)

    def _apply_pose_augmentation(
        self,
        *,
        sampled_points_camera: np.ndarray,
        all_points_camera: np.ndarray,
        crop_centroid_camera: np.ndarray,
        object_to_camera: np.ndarray,
        sampled_features: np.ndarray | None,
        all_features: np.ndarray | None,
        sample_index: int,
        augment_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        rng = np.random.default_rng(self.seed * 1000003 + int(sample_index) * 9176 + int(augment_index))
        max_rotation_deg = float(self.pose_augmentation.get("max_camera_rotation_deg", 0.0))
        translation_std_m = float(self.pose_augmentation.get("translation_std_m", 0.0))
        point_jitter_std_m = float(self.pose_augmentation.get("point_jitter_std_m", 0.0))
        rotation_camera_aug = random_rotation_matrix(max_rotation_deg, rng)
        translation_camera_aug = rng.normal(0.0, translation_std_m, size=3).astype(np.float32) if translation_std_m > 0 else np.zeros(3, dtype=np.float32)

        sampled_points_aug = sampled_points_camera @ rotation_camera_aug.T + translation_camera_aug.reshape(1, 3)
        all_points_aug = all_points_camera @ rotation_camera_aug.T + translation_camera_aug.reshape(1, 3)
        if point_jitter_std_m > 0:
            sampled_points_aug = sampled_points_aug + rng.normal(0.0, point_jitter_std_m, size=sampled_points_aug.shape).astype(np.float32)
            all_points_aug = all_points_aug + rng.normal(0.0, point_jitter_std_m, size=all_points_aug.shape).astype(np.float32)
        centroid_aug = crop_centroid_camera @ rotation_camera_aug.T + translation_camera_aug
        object_to_camera_aug = object_to_camera.copy()
        object_to_camera_aug[:3, :3] = rotation_camera_aug @ object_to_camera[:3, :3]
        object_to_camera_aug[:3, 3] = object_to_camera[:3, 3] @ rotation_camera_aug.T + translation_camera_aug
        sampled_features_aug = None if sampled_features is None else sampled_features @ rotation_camera_aug.T
        all_features_aug = None if all_features is None else all_features @ rotation_camera_aug.T
        return (
            sampled_points_aug.astype(np.float32),
            all_points_aug.astype(np.float32),
            centroid_aug.astype(np.float32),
            object_to_camera_aug.astype(np.float32),
            None if sampled_features_aug is None else sampled_features_aug.astype(np.float32),
            None if all_features_aug is None else all_features_aug.astype(np.float32),
        )


def build_pose_crop_dataset_from_config(
    config: dict[str, Any],
    root: str | Path,
    *,
    min_matched_gt_iou: float | None = None,
    enable_augmentation: bool = False,
) -> PoseCropDataset:
    dataset_config = config.get("dataset", {})
    model_config = config.get("model", {})
    object_config = config.get("object", {})
    return PoseCropDataset(
        root,
        num_points=int(model_config.get("num_points", dataset_config.get("num_points", 1024))),
        model_path=object_config["model_path"],
        model_scale=float(object_config.get("model_scale", 1.0)),
        model_point_count=int(object_config.get("model_point_count", 1024)),
        keypoint_count=int(model_config.get("keypoint_count", dataset_config.get("keypoint_count", 8))),
        symmetry_name=str(object_config.get("symmetry_name", "none")),
        min_matched_gt_iou=min_matched_gt_iou,
        require_pose=bool(dataset_config.get("require_pose", True)),
        seed=int(config.get("seed", config.get("train", {}).get("seed", 7))),
        pose_augmentation=dataset_config.get("pose_augmentation", {}) if enable_augmentation else None,
    )


def rigid_object_to_camera_from_scaled(object_to_camera: np.ndarray) -> np.ndarray:
    """Return a rigid meter-space pose from metadata that may include model scale.

    Blender metadata stores `object_to_camera` for the original STL coordinate
    frame. For millimeter-authored meshes such as bending_pipe, the upper-left
    block includes `model_scale`. Pose training uses model points already
    converted to meters, so the target rotation must be orthonormal.
    """

    matrix = np.asarray(object_to_camera, dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"expected object_to_camera shape (4, 4), got {matrix.shape}")
    result = matrix.copy()
    linear = matrix[:3, :3].astype(np.float64)
    column_norms = np.linalg.norm(linear, axis=0)
    scale = float(np.mean(column_norms))
    if scale <= 1e-12:
        raise ValueError("object_to_camera has degenerate linear block")
    normalized = linear / scale
    u, _s, vh = np.linalg.svd(normalized)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    result[:3, :3] = rotation.astype(np.float32)
    return result.astype(np.float32)


def positive_forward_depth_to_metadata_camera(points_camera: np.ndarray) -> np.ndarray:
    """Convert project point-cloud camera coords to metadata/Blender camera coords.

    Depth point clouds use `(x, y, z_forward)` with positive `z` in front of the
    camera. Blender metadata uses the camera's native frame, where visible
    points have negative local `z`. Pose rotation regression is done in the
    metadata camera frame so `rotation_object_to_camera` remains a proper SO(3)
    matrix.
    """

    points = np.asarray(points_camera, dtype=np.float32).copy()
    points[..., 2] *= -1.0
    return points.astype(np.float32)


def random_rotation_matrix(max_rotation_deg: float, rng: np.random.Generator) -> np.ndarray:
    """Sample a camera-frame rotation matrix for pose augmentation."""

    if max_rotation_deg <= 0:
        return np.eye(3, dtype=np.float32)
    axis = rng.normal(0.0, 1.0, size=3)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = axis / axis_norm
    angle = float(rng.uniform(-np.deg2rad(max_rotation_deg), np.deg2rad(max_rotation_deg)))
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    one_c = 1.0 - c
    rotation = np.asarray(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float32,
    )
    return rotation


def farthest_point_keypoints(vertices_object_m: np.ndarray, keypoint_count: int, *, seed: int = 7) -> np.ndarray:
    """Pick deterministic mesh keypoints with farthest-point sampling."""

    vertices = np.asarray(vertices_object_m, dtype=np.float32)
    count = max(1, int(keypoint_count))
    if vertices.shape[0] == 0:
        raise ValueError("cannot sample keypoints from an empty mesh")
    rng = np.random.default_rng(seed)
    centroid = vertices.mean(axis=0, keepdims=True)
    first = int(np.argmax(np.linalg.norm(vertices - centroid, axis=1)))
    selected = [first]
    min_dist2 = np.sum((vertices - vertices[first]) ** 2, axis=1)
    while len(selected) < count:
        next_index = int(np.argmax(min_dist2))
        if next_index in selected:
            next_index = int(rng.integers(0, vertices.shape[0]))
        selected.append(next_index)
        dist2 = np.sum((vertices - vertices[next_index]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
    return vertices[np.asarray(selected, dtype=np.int64)].astype(np.float32)
