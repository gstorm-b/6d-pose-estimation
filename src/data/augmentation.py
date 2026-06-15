"""NumPy augmentations for processed point-cloud datasets.

Two tiers:
  - conservative geometric/sensor jitter (xyz/depth/normal jitter, random dropout,
    outliers, z-rotation) used since the first trainings, and
  - structured-light sim-to-real noise (P8): depth-quadratic along-ray noise +
    quantization, normal-incidence (grazing) dropout, depth-discontinuity (edge)
    dropout, and specular blob dropout. All sim-to-real terms default to off and
    only switch on once validated against the clean synthetic gates.

Dropout never shrinks the cloud: dropped slots are refilled by resampling the
survivors so the model keeps its fixed-size input. Camera-dependent terms need
the camera position in the SAME frame as `points`; pass it via `camera_position`
(the instance dataset normalizes to scene center, so it passes
`-normalization_center`). When unknown, a top-down standoff camera is assumed.
"""

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
    # --- structured-light sim-to-real noise (P8); all off by default ---
    depth_quadratic_noise: float = 0.0       # along-ray std = coef * range^2 (m per m^2)
    depth_quantization_m: float = 0.0        # quantize depth (camera z) to this step (m)
    incidence_dropout_max_prob: float = 0.0  # drop prob at full grazing (|n.v| -> 0)
    incidence_dropout_power: float = 2.0     # shaping exponent on (1 - |n.v|)
    edge_dropout_prob: float = 0.0           # drop prob for depth-discontinuity points
    edge_dropout_neighbors: int = 8          # k neighbors (in xy) for the depth-gap test
    edge_dropout_gap_m: float = 0.005        # neighbor z-gap that counts as an edge
    blob_dropout_count: int = 0              # number of specular holes per cloud
    blob_dropout_radius_m: float = 0.01      # hole radius (m)
    camera_fallback_standoff_m: float = 0.6  # used only when camera_position is unknown

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "PointCloudAugmentationConfig":
        if not value:
            return cls()
        known = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {key: value[key] for key in known if key in value}
        return cls(**kwargs)

    @property
    def uses_camera(self) -> bool:
        return self.depth_quadratic_noise > 0 or self.incidence_dropout_max_prob > 0


def _resample_refill(
    keep: np.ndarray,
    arrays: list[np.ndarray | None],
    rng: np.random.Generator,
) -> list[np.ndarray | None]:
    """Refill dropped slots (~keep) by resampling kept indices; keeps size fixed."""

    n = keep.shape[0]
    if keep.all():
        return arrays
    if not keep.any():
        keep = keep.copy()
        keep[int(rng.integers(0, n))] = True
    kept = np.flatnonzero(keep)
    drop = np.flatnonzero(~keep)
    replacement = rng.choice(kept, size=drop.shape[0], replace=True)
    out: list[np.ndarray | None] = []
    for arr in arrays:
        if arr is None:
            out.append(None)
            continue
        arr = arr.copy()
        arr[drop] = arr[replacement]
        out.append(arr)
    return out


def _camera_position(points: np.ndarray, camera_position: np.ndarray | None, standoff: float) -> np.ndarray:
    if camera_position is not None:
        return np.asarray(camera_position, dtype=np.float32).reshape(3)
    # Unknown camera: assume a top-down sensor a standoff in front of the nearest surface.
    return np.array(
        [float(points[:, 0].mean()), float(points[:, 1].mean()), float(points[:, 2].min()) - float(standoff)],
        dtype=np.float32,
    )


def _edge_keep(points: np.ndarray, config: PointCloudAugmentationConfig, rng: np.random.Generator) -> np.ndarray:
    """Drop points adjacent to a depth discontinuity (shadowing), via xy-neighbor z-gaps."""

    n = points.shape[0]
    k = min(int(config.edge_dropout_neighbors), n - 1)
    if k < 1:
        return np.ones(n, dtype=bool)
    try:
        from scipy.spatial import cKDTree
    except Exception:  # noqa: BLE001 - no scipy -> skip edge dropout
        return np.ones(n, dtype=bool)
    tree = cKDTree(points[:, :2])
    _dists, idx = tree.query(points[:, :2], k=k + 1)
    neighbor_z = points[idx[:, 1:], 2]
    zgap = np.abs(neighbor_z - points[:, 2:3]).max(axis=1)
    on_edge = zgap > float(config.edge_dropout_gap_m)
    drop = on_edge & (rng.random(n) < float(config.edge_dropout_prob))
    return ~drop


def _blob_keep(points: np.ndarray, config: PointCloudAugmentationConfig, rng: np.random.Generator) -> np.ndarray:
    """Drop all points inside a few random spheres (specular dropout holes)."""

    n = points.shape[0]
    count = int(config.blob_dropout_count)
    radius = float(config.blob_dropout_radius_m)
    if count <= 0 or radius <= 0 or n == 0:
        return np.ones(n, dtype=bool)
    seeds = rng.choice(n, size=min(count, n), replace=False)
    keep = np.ones(n, dtype=bool)
    for seed in seeds:
        keep &= np.linalg.norm(points - points[seed], axis=1) > radius
    return keep


def augment_point_cloud(
    points: np.ndarray,
    features: np.ndarray | None,
    semantic_labels: np.ndarray,
    instance_labels: np.ndarray,
    config: PointCloudAugmentationConfig,
    rng: np.random.Generator | None = None,
    *,
    camera_position: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Apply sensor-like + structured-light augmentation to a fixed-size point cloud."""

    if not config.enabled:
        return points, features, semantic_labels, instance_labels

    rng = np.random.default_rng() if rng is None else rng
    aug_points = np.asarray(points, dtype=np.float32).copy()
    aug_features = None if features is None else np.asarray(features, dtype=np.float32).copy()
    aug_semantic = np.asarray(semantic_labels).copy()
    aug_instance = np.asarray(instance_labels).copy()
    n = aug_points.shape[0]

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
        aug_points[:, 2] += rng.normal(0.0, config.depth_noise_std, size=n).astype(np.float32)

    # Structured-light depth model: along-ray noise that grows with range^2, then quantization.
    view_dir = None
    if config.uses_camera:
        cam = _camera_position(aug_points, camera_position, config.camera_fallback_standoff_m)
        ray = aug_points - cam
        ranges = np.linalg.norm(ray, axis=1)
        view_dir = ray / np.maximum(ranges[:, None], 1e-6)
        if config.depth_quadratic_noise > 0:
            std = config.depth_quadratic_noise * (ranges ** 2)
            aug_points += (view_dir * (rng.standard_normal(n) * std)[:, None]).astype(np.float32)

    if config.depth_quantization_m > 0:
        step = float(config.depth_quantization_m)
        aug_points[:, 2] = np.round(aug_points[:, 2] / step) * step

    if aug_features is not None and config.normal_jitter_std > 0 and aug_features.shape[1] >= 3:
        aug_features[:, :3] += rng.normal(0.0, config.normal_jitter_std, size=aug_features[:, :3].shape).astype(np.float32)
        norms = np.linalg.norm(aug_features[:, :3], axis=1, keepdims=True)
        aug_features[:, :3] = aug_features[:, :3] / np.maximum(norms, 1e-6)

    # Combine every dropout source into one keep mask, then refill once.
    keep = np.ones(n, dtype=bool)
    if config.point_dropout_prob > 0:
        keep &= rng.random(n) >= config.point_dropout_prob
    if config.incidence_dropout_max_prob > 0 and aug_features is not None and aug_features.shape[1] >= 3:
        if view_dir is None:
            cam = _camera_position(aug_points, camera_position, config.camera_fallback_standoff_m)
            ray = aug_points - cam
            view_dir = ray / np.maximum(np.linalg.norm(ray, axis=1, keepdims=True), 1e-6)
        cos_incidence = np.abs(np.sum(aug_features[:, :3] * view_dir, axis=1))
        drop_prob = config.incidence_dropout_max_prob * np.power(
            np.clip(1.0 - cos_incidence, 0.0, 1.0), config.incidence_dropout_power
        )
        keep &= rng.random(n) >= drop_prob
    if config.edge_dropout_prob > 0:
        keep &= _edge_keep(aug_points, config, rng)
    if config.blob_dropout_count > 0:
        keep &= _blob_keep(aug_points, config, rng)
    if not keep.all():
        aug_points, aug_features, aug_semantic, aug_instance = _resample_refill(
            keep, [aug_points, aug_features, aug_semantic, aug_instance], rng
        )

    if config.outlier_ratio > 0 and n > 0:
        count = int(round(n * config.outlier_ratio))
        if count > 0:
            indices = rng.choice(n, size=min(count, n), replace=False)
            center = aug_points.mean(axis=0, keepdims=True)
            aug_points[indices] = center + rng.normal(0.0, config.outlier_std, size=(len(indices), 3)).astype(np.float32)
            aug_semantic[indices] = 0
            aug_instance[indices] = 0
            if aug_features is not None:
                aug_features[indices] = 0.0

    return aug_points, aug_features, aug_semantic, aug_instance
