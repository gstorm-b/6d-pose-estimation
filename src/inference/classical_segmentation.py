"""Classical (no-learning) instance segmentation for bin scenes.

Plane-fit the bin floor, drop floor/wall points, Euclidean-cluster what
remains, and rank clusters topmost-first. For the industrial goal - find the
pickable parts on top of the pile - this is a robust baseline that does not
depend on the learned semantic head, works on any SKU without training, and
doubles as an honest reference when measuring the learned model on real data.

Returns the same result type as the learned ``InstanceSegmenter`` so viewers,
CLIs and downstream pose stages can consume either interchangeably.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.inference.instance_segmentation import InstanceSegmentationResult, SegmentedInstance
from src.inference.workspace_crop import (
    PlaneModel,
    _dominant_floor_component,
    _inside_eroded_hull,
    fit_floor_plane,
)


@dataclass(frozen=True)
class ClassicalSegmentationConfig:
    floor_clearance_m: float = 0.008    # points closer to the floor than this are background
    max_height_m: float = 0.30          # points higher above the floor than this are rig/noise
    wall_erosion_m: float = 0.03        # shrink the floor hull by this to drop the bin walls
    plane_distance_m: float = 0.004     # RANSAC inlier threshold for the floor fit
    plane_iterations: int = 256
    voxel_size_m: float = 0.002         # cluster on a voxel-downsampled cloud for speed (0 = off)
    cluster_eps_m: float = 0.004        # Euclidean clustering neighbour distance
    cluster_min_samples: int = 8
    min_cluster_points: int = 120       # reject clusters smaller than this (full-resolution points)
    max_instances: int = 50
    seed: int = 7
    # Touching parts merge under pure Euclidean clustering. Crest splitting
    # re-clusters only each cluster's highest points (the crests of lying parts
    # separate even when their flanks touch) and grows the split back out.
    crest_split: bool = True
    crest_quantile: float = 0.7         # per-cluster height quantile that defines the crest
    crest_eps_m: float = 0.005          # crest re-clustering neighbour distance
    crest_min_seed: int = 40            # minimum crest points for a valid sub-cluster seed

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "ClassicalSegmentationConfig":
        if not value:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: value[k] for k in known if k in value})


def _voxel_downsample_indices(points: np.ndarray, voxel_size: float, rng: np.random.Generator) -> np.ndarray:
    """One representative index per occupied voxel."""

    keys = np.floor(points / voxel_size).astype(np.int64)
    order = rng.permutation(points.shape[0])
    _uniq, first = np.unique(keys[order], axis=0, return_index=True)
    return np.sort(order[first])


def _crest_split(
    points: np.ndarray,
    heights: np.ndarray,
    labels: np.ndarray,
    cfg: "ClassicalSegmentationConfig",
) -> np.ndarray:
    """Split merged clusters by re-clustering their crests and growing back out."""

    from scipy.spatial import cKDTree

    from src.inference.instance_clustering import _dbscan

    out = labels.copy()
    next_id = int(out.max()) + 1
    for cluster_id in (int(v) for v in np.unique(labels) if int(v) >= 0):
        mask = labels == cluster_id
        if int(mask.sum()) < 2 * cfg.crest_min_seed:
            continue
        crest_threshold = float(np.quantile(heights[mask], cfg.crest_quantile))
        crest = mask & (heights >= crest_threshold)
        seed_raw = _dbscan(points[crest], cfg.crest_eps_m, cfg.cluster_min_samples)
        valid = [s for s in np.unique(seed_raw) if s >= 0 and int((seed_raw == s).sum()) >= cfg.crest_min_seed]
        if len(valid) < 2:
            continue
        seed_points = np.concatenate([points[crest][seed_raw == s] for s in valid])
        seed_ranks = np.concatenate(
            [np.full(int((seed_raw == s).sum()), rank, dtype=np.int64) for rank, s in enumerate(valid)]
        )
        _dist, nearest = cKDTree(seed_points).query(points[mask], k=1)
        assigned = seed_ranks[nearest]
        member_idx = np.flatnonzero(mask)
        for rank in range(1, len(valid)):
            out[member_idx[assigned == rank]] = next_id
            next_id += 1
    return out


def segment_scene_classical(
    points_camera: np.ndarray,
    config: ClassicalSegmentationConfig | dict[str, Any] | None = None,
    *,
    pixels: np.ndarray | None = None,
    plane: PlaneModel | None = None,
) -> InstanceSegmentationResult:
    """Segment a bin scene (positive-forward metres) without a learned model."""

    from src.inference.instance_clustering import _dbscan

    cfg = config if isinstance(config, ClassicalSegmentationConfig) else ClassicalSegmentationConfig.from_mapping(config)
    points = np.asarray(points_camera, dtype=np.float32)
    n = points.shape[0]
    rng = np.random.default_rng(cfg.seed)
    t0 = time.perf_counter()

    if plane is None:
        plane = fit_floor_plane(
            points,
            distance_threshold=cfg.plane_distance_m,
            iterations=cfg.plane_iterations,
            seed=cfg.seed,
        )
    height = plane.signed_height(points)

    candidate = (height >= cfg.floor_clearance_m) & (height <= cfg.max_height_m)
    floor_inliers = np.abs(height) < cfg.plane_distance_m
    if floor_inliers.sum() >= 16 and cfg.wall_erosion_m > 0:
        floor_patch = _dominant_floor_component(points[floor_inliers][:, :2], seed=cfg.seed)
        candidate &= _inside_eroded_hull(points[:, :2], floor_patch, cfg.wall_erosion_m)
    plane_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    labels = np.zeros(n, dtype=np.int64)
    candidate_idx = np.flatnonzero(candidate)
    if candidate_idx.size >= cfg.cluster_min_samples:
        cand_points = points[candidate_idx]
        cand_heights = height[candidate_idx]
        if cfg.voxel_size_m > 0 and cand_points.shape[0] > 20000:
            voxel_local = _voxel_downsample_indices(cand_points, cfg.voxel_size_m, rng)
            cluster_points = cand_points[voxel_local]
            raw = _dbscan(cluster_points, cfg.cluster_eps_m, cfg.cluster_min_samples)
            if cfg.crest_split:
                raw = _crest_split(cluster_points, cand_heights[voxel_local], raw, cfg)
            # Propagate voxel labels back to every candidate point.
            from scipy.spatial import cKDTree

            _d, nearest = cKDTree(cluster_points).query(cand_points, k=1)
            raw_full = raw[nearest]
        else:
            raw_full = _dbscan(cand_points, cfg.cluster_eps_m, cfg.cluster_min_samples)
            if cfg.crest_split:
                raw_full = _crest_split(cand_points, cand_heights, raw_full, cfg)

        next_id = 1
        for raw_label in sorted(int(v) for v in np.unique(raw_full) if int(v) >= 0):
            mask = raw_full == raw_label
            if int(mask.sum()) < cfg.min_cluster_points:
                continue
            labels[candidate_idx[mask]] = next_id
            next_id += 1
    clustering_ms = (time.perf_counter() - t1) * 1000.0

    instances: list[SegmentedInstance] = []
    for instance_id in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        idx = np.flatnonzero(labels == instance_id)
        cluster = points[idx]
        instances.append(
            SegmentedInstance(
                instance_id=instance_id,
                point_indices=idx,
                point_count=int(idx.shape[0]),
                centroid_camera=cluster.mean(axis=0),
                bbox_min_camera=cluster.min(axis=0),
                bbox_max_camera=cluster.max(axis=0),
                mean_object_probability=1.0,
            )
        )
    # Topmost first: pickable parts sit highest above the floor.
    instances.sort(key=lambda inst: float(np.max(height[inst.point_indices])), reverse=True)
    instances = instances[: cfg.max_instances]

    # Compact ids to the ranked order and rebuild the label image accordingly.
    ranked_labels = np.zeros(n, dtype=np.int64)
    for rank, inst in enumerate(instances, start=1):
        ranked_labels[inst.point_indices] = rank
        inst.instance_id = rank

    voted_centers = points.copy()
    for inst in instances:
        voted_centers[inst.point_indices] = inst.centroid_camera

    return InstanceSegmentationResult(
        points_camera=points,
        instance_labels=ranked_labels,
        object_probabilities=(ranked_labels > 0).astype(np.float32),
        voted_centers_camera=voted_centers,
        instances=instances,
        timings_ms={"plane_ms": plane_ms, "clustering_ms": clustering_ms},
        pixels=pixels,
        source_point_count=n,
    )
