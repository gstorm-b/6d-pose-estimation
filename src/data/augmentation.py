"""NumPy augmentations for processed point-cloud datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PointCloudAugmentationConfig:
    enabled: bool = False
    xyz_jitter_std: float = 0.0
    xyz_jitter_clip: float = 0.003
    depth_noise_std: float = 0.0
    point_dropout_prob: float = 0.0
    outlier_ratio: float = 0.0
    outlier_std: float = 0.02
    normal_jitter_std: float = 0.0
    random_z_rotation: bool = False

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "PointCloudAugmentationConfig":
        if not value:
            return cls()
        known = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {key: value[key] for key in known if key in value}
        return cls(**kwargs)


def augment_point_cloud(
    points: np.ndarray,
    features: np.ndarray | None,
    semantic_labels: np.ndarray,
    instance_labels: np.ndarray,
    config: PointCloudAugmentationConfig,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Apply conservative sensor-like augmentation to a fixed-size point cloud."""

    if not config.enabled:
        return points, features, semantic_labels, instance_labels

    rng = np.random.default_rng() if rng is None else rng
    aug_points = np.asarray(points, dtype=np.float32).copy()
    aug_features = None if features is None else np.asarray(features, dtype=np.float32).copy()
    aug_semantic = np.asarray(semantic_labels).copy()
    aug_instance = np.asarray(instance_labels).copy()

    if config.random_z_rotation:
        theta = float(rng.uniform(0.0, np.pi * 2.0))
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rot = np.array([[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        aug_points = aug_points @ rot.T
        if aug_features is not None and aug_features.shape[1] >= 3:
            aug_features[:, :3] = aug_features[:, :3] @ rot.T

    if config.xyz_jitter_std > 0:
        jitter = rng.normal(0.0, config.xyz_jitter_std, size=aug_points.shape).astype(np.float32)
        if config.xyz_jitter_clip > 0:
            jitter = np.clip(jitter, -config.xyz_jitter_clip, config.xyz_jitter_clip)
        aug_points += jitter

    if config.depth_noise_std > 0:
        aug_points[:, 2] += rng.normal(0.0, config.depth_noise_std, size=aug_points.shape[0]).astype(np.float32)

    if aug_features is not None and config.normal_jitter_std > 0 and aug_features.shape[1] >= 3:
        aug_features[:, :3] += rng.normal(0.0, config.normal_jitter_std, size=aug_features[:, :3].shape).astype(np.float32)
        norms = np.linalg.norm(aug_features[:, :3], axis=1, keepdims=True)
        aug_features[:, :3] = aug_features[:, :3] / np.maximum(norms, 1e-6)

    if config.point_dropout_prob > 0 and aug_points.shape[0] > 0:
        keep = rng.random(aug_points.shape[0]) >= config.point_dropout_prob
        if not np.any(keep):
            keep[int(rng.integers(0, aug_points.shape[0]))] = True
        replacement = rng.choice(np.flatnonzero(keep), size=aug_points.shape[0] - int(keep.sum()), replace=True)
        drop_indices = np.flatnonzero(~keep)
        aug_points[drop_indices] = aug_points[replacement]
        if aug_features is not None:
            aug_features[drop_indices] = aug_features[replacement]
        aug_semantic[drop_indices] = aug_semantic[replacement]
        aug_instance[drop_indices] = aug_instance[replacement]

    if config.outlier_ratio > 0 and aug_points.shape[0] > 0:
        count = int(round(aug_points.shape[0] * config.outlier_ratio))
        if count > 0:
            indices = rng.choice(aug_points.shape[0], size=min(count, aug_points.shape[0]), replace=False)
            center = aug_points.mean(axis=0, keepdims=True)
            aug_points[indices] = center + rng.normal(0.0, config.outlier_std, size=(len(indices), 3)).astype(np.float32)
            aug_semantic[indices] = 0
            aug_instance[indices] = 0
            if aug_features is not None:
                aug_features[indices] = 0.0

    return aug_points, aug_features, aug_semantic, aug_instance
