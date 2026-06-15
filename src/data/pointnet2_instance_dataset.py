"""PyTorch dataset wrapper for PointNet++ instance segmentation targets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .augmentation import PointCloudAugmentationConfig, augment_point_cloud

try:
    import torch
    from torch.utils.data import Dataset

    TORCH_AVAILABLE = True
except Exception:  # noqa: BLE001 - keep imports usable before training deps are installed.
    torch = None
    Dataset = object  # type: ignore[assignment,misc]
    TORCH_AVAILABLE = False


class PointNet2InstanceSegDataset(Dataset):  # type: ignore[misc]
    """Load processed PointNet++ samples and derive per-point instance targets.

    The processed `points` array is treated as metric camera-frame xyz before
    normalization. Model inputs use `points_camera_normalized`; centroid offset
    targets are computed in the same normalized frame so they match the tensor
    passed to the network.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        *,
        use_normals: bool = True,
        normalize: str = "scene_center",
        min_points_per_instance: int = 16,
        max_instances: int = 128,
        centroid_distance_weight_min: float = 0.5,
        centroid_distance_weight_max: float = 1.5,
        augment: PointCloudAugmentationConfig | dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use PointNet2InstanceSegDataset.")
        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split
        self.use_normals = use_normals
        self.normalize = normalize
        self.min_points_per_instance = int(min_points_per_instance)
        self.max_instances = int(max_instances)
        self.centroid_distance_weight_min = float(centroid_distance_weight_min)
        self.centroid_distance_weight_max = float(centroid_distance_weight_max)
        self.augment_config = (
            augment
            if isinstance(augment, PointCloudAugmentationConfig)
            else PointCloudAugmentationConfig.from_mapping(augment)
        )
        self.rng = np.random.default_rng(seed)
        self.files = sorted(self.split_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No processed .npz samples found in {self.split_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.files[index]
        with np.load(path) as data:
            points_camera = np.asarray(data["points"], dtype=np.float32)
            semantic_labels = np.asarray(data["semantic_labels"], dtype=np.int64)
            instance_labels = np.asarray(data["instance_labels"], dtype=np.int64)
            point_pixels = np.asarray(data["point_pixels"], dtype=np.uint16)
            raw_sample = str(np.asarray(data["raw_sample"]).item())
            features = np.asarray(data["features"], dtype=np.float32) if self.use_normals and "features" in data.files else None

        points_normalized, normalization_center = self._normalize_points(points_camera)
        points_normalized, features, semantic_labels, instance_labels = augment_point_cloud(
            points_normalized,
            features,
            semantic_labels,
            instance_labels,
            self.augment_config,
            self.rng,
            camera_position=-normalization_center.reshape(3),
        )
        targets = self._build_instance_targets(points_normalized, semantic_labels, instance_labels)
        model_input = points_normalized if features is None else np.concatenate([points_normalized, features], axis=1)
        points_camera_augmented = points_normalized + normalization_center.reshape(1, 3)

        return {
            "points": torch.from_numpy(points_normalized.astype(np.float32)),
            "points_camera": torch.from_numpy(points_camera_augmented.astype(np.float32)),
            "points_camera_normalized": torch.from_numpy(points_normalized.astype(np.float32)),
            "features": None if features is None else torch.from_numpy(features.astype(np.float32)),
            "model_input": torch.from_numpy(model_input.astype(np.float32)),
            "semantic_labels": torch.from_numpy(semantic_labels.astype(np.int64)),
            "instance_labels": torch.from_numpy(instance_labels.astype(np.int64)),
            "centroid_offsets": torch.from_numpy(targets["centroid_offsets"]),
            "valid_instance_mask": torch.from_numpy(targets["valid_instance_mask"]),
            "object_mask": torch.from_numpy(targets["object_mask"]),
            "centroid_distance_weights": torch.from_numpy(targets["centroid_distance_weights"]),
            "instance_ids": torch.from_numpy(targets["instance_ids"]),
            "instance_point_counts": torch.from_numpy(targets["instance_point_counts"]),
            "num_instances": torch.tensor(targets["num_instances"], dtype=torch.int64),
            "normalization_center_camera": torch.from_numpy(normalization_center.astype(np.float32)),
            "point_pixels": torch.from_numpy(point_pixels.astype(np.int64)),
            "raw_sample": raw_sample,
            "path": str(path),
        }

    def _normalize_points(self, points_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.normalize == "none":
            center = np.zeros(3, dtype=np.float32)
            return points_camera.astype(np.float32), center
        if self.normalize == "scene_center":
            center = points_camera.mean(axis=0).astype(np.float32)
            return (points_camera - center.reshape(1, 3)).astype(np.float32), center
        raise ValueError(f"Unknown normalization mode: {self.normalize}")

    def _build_instance_targets(
        self,
        points: np.ndarray,
        semantic_labels: np.ndarray,
        instance_labels: np.ndarray,
    ) -> dict[str, np.ndarray]:
        centroid_offsets = np.zeros_like(points, dtype=np.float32)
        object_mask = (semantic_labels > 0) & (instance_labels > 0)
        valid_instance_mask = np.zeros(points.shape[0], dtype=np.bool_)
        centroid_distance_weights = np.zeros(points.shape[0], dtype=np.float32)
        instance_ids: list[int] = []
        instance_point_counts: list[int] = []

        for instance_id in sorted(int(value) for value in np.unique(instance_labels[object_mask])):
            mask = object_mask & (instance_labels == instance_id)
            count = int(mask.sum())
            if count == 0:
                continue
            instance_ids.append(instance_id)
            instance_point_counts.append(count)
            centroid = points[mask].mean(axis=0)
            offsets = centroid.reshape(1, 3) - points[mask]
            centroid_offsets[mask] = offsets.astype(np.float32)
            if count >= self.min_points_per_instance:
                valid_instance_mask[mask] = True
            centroid_distance_weights[mask] = self._centroid_distance_weights(offsets)

        compact_ids = np.asarray(instance_ids, dtype=np.int64)
        compact_counts = np.asarray(instance_point_counts, dtype=np.int64)
        padded_ids = np.full(self.max_instances, -1, dtype=np.int64)
        padded_counts = np.zeros(self.max_instances, dtype=np.int64)
        keep = min(compact_ids.shape[0], self.max_instances)
        if keep > 0:
            padded_ids[:keep] = compact_ids[:keep]
            padded_counts[:keep] = compact_counts[:keep]

        return {
            "centroid_offsets": centroid_offsets.astype(np.float32),
            "valid_instance_mask": valid_instance_mask,
            "object_mask": object_mask.astype(np.bool_),
            "centroid_distance_weights": centroid_distance_weights.astype(np.float32),
            "instance_ids": padded_ids,
            "instance_point_counts": padded_counts,
            "num_instances": np.asarray(keep, dtype=np.int64),
        }

    def _centroid_distance_weights(self, offsets: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(offsets, axis=1).astype(np.float32)
        if distances.size == 0:
            return distances
        max_distance = float(distances.max())
        if max_distance <= 1e-8:
            return np.full_like(distances, self.centroid_distance_weight_min, dtype=np.float32)
        normalized = distances / max_distance
        weight_range = self.centroid_distance_weight_max - self.centroid_distance_weight_min
        return (self.centroid_distance_weight_min + normalized * weight_range).astype(np.float32)
