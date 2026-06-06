"""PyTorch dataset wrapper for processed PointNet++ semantic data."""

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


class PointNet2SemSegDataset(Dataset):  # type: ignore[misc]
    """Load processed `.npz` samples for PointNet++ semantic segmentation."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        *,
        use_normals: bool = True,
        normalize: str = "scene_center",
        augment: PointCloudAugmentationConfig | dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use PointNet2SemSegDataset.")
        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split
        self.use_normals = use_normals
        self.normalize = normalize
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
            points = np.asarray(data["points"], dtype=np.float32)
            semantic_labels = np.asarray(data["semantic_labels"], dtype=np.int64)
            instance_labels = np.asarray(data["instance_labels"], dtype=np.int64)
            point_pixels = np.asarray(data["point_pixels"], dtype=np.uint16)
            raw_sample = str(np.asarray(data["raw_sample"]).item())
            features = np.asarray(data["features"], dtype=np.float32) if self.use_normals and "features" in data.files else None

        points = self._normalize_points(points)
        points, features, semantic_labels, instance_labels = augment_point_cloud(
            points,
            features,
            semantic_labels,
            instance_labels,
            self.augment_config,
            self.rng,
        )
        model_input = points if features is None else np.concatenate([points, features], axis=1).astype(np.float32)

        return {
            "points": torch.from_numpy(points.astype(np.float32)),
            "features": None if features is None else torch.from_numpy(features.astype(np.float32)),
            "model_input": torch.from_numpy(model_input.astype(np.float32)),
            "semantic_labels": torch.from_numpy(semantic_labels.astype(np.int64)),
            "instance_labels": torch.from_numpy(instance_labels.astype(np.int64)),
            "point_pixels": torch.from_numpy(point_pixels.astype(np.int64)),
            "raw_sample": raw_sample,
            "path": str(path),
        }

    def _normalize_points(self, points: np.ndarray) -> np.ndarray:
        if self.normalize == "none":
            return points.astype(np.float32)
        if self.normalize == "scene_center":
            center = points.mean(axis=0, keepdims=True)
            return (points - center).astype(np.float32)
        raise ValueError(f"Unknown normalization mode: {self.normalize}")
