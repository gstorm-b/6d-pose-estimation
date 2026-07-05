"""Workspace (bin ROI) cropping for real scenes.

Synthetic training scenes contain nothing but the bin and its parts, while a
real camera also sees the rig frame, cables, neighbouring bins and the floor.
Cropping the cloud to the bin workspace before inference removes that
out-of-distribution background - it is the cheap, deterministic half of closing
the sim-to-real gap, and on a fixed industrial mount the bounds only need to be
calibrated once.

Two modes:
- ``bounds``: an explicit axis-aligned box in the positive-forward camera frame
  (the production path: calibrate once per cell, store in config).
- ``auto_plane``: RANSAC-fit the bin floor (dominant far plane), keep points
  above it and inside the eroded xy hull of its inliers. Convenience for
  first-contact captures where no calibration exists yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PlaneModel:
    """Plane n.p + d = 0 with n oriented toward the camera (n_z < 0)."""

    normal: np.ndarray  # (3,) unit
    offset: float       # d

    def signed_height(self, points: np.ndarray) -> np.ndarray:
        """Height above the plane, positive toward the camera.

        With the unit normal oriented toward the camera, a point displaced by
        `h` along the normal satisfies n.p + d = h, so the signed height is
        simply the plane residual.
        """

        return points @ self.normal + self.offset


@dataclass(frozen=True)
class WorkspaceCropConfig:
    mode: str = "none"                    # none | bounds | auto_plane
    # bounds mode (metres, positive-forward camera frame; None = unbounded)
    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None
    z_min: float | None = None
    z_max: float | None = None
    # auto_plane mode
    plane_distance_m: float = 0.004       # RANSAC inlier threshold
    plane_iterations: int = 256
    min_height_m: float = 0.0             # keep points at least this far above the floor
    max_height_m: float = 0.35            # ...and no farther than this (drops the rig above the bin rim)
    hull_erosion_m: float = 0.02          # shrink the floor-inlier xy hull to drop the walls
    seed: int = 7

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "WorkspaceCropConfig":
        if not value:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: value[k] for k in known if k in value})


def fit_floor_plane(
    points: np.ndarray,
    *,
    distance_threshold: float = 0.004,
    iterations: int = 256,
    seed: int = 7,
) -> PlaneModel:
    """RANSAC-fit the bin floor: the dominant plane among the farthest points.

    Candidate triples are drawn from the far half of the cloud (largest z),
    which is where the floor lives in a top-down bin view; scoring counts
    inliers over the full cloud so a floor larger than the far half still wins.
    """

    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 3:
        raise ValueError("need at least 3 points to fit a plane")
    rng = np.random.default_rng(seed)
    # RANSAC scoring on a subsample keeps the fit fast on full-resolution clouds;
    # the final SVD refinement below still uses every inlier.
    score_pts = pts
    if score_pts.shape[0] > 60000:
        score_pts = pts[rng.choice(pts.shape[0], size=60000, replace=False)]
    z_median = float(np.median(score_pts[:, 2]))
    far = score_pts[score_pts[:, 2] >= z_median]
    if far.shape[0] < 3:
        far = score_pts

    best_inliers = -1
    best: tuple[np.ndarray, float] | None = None
    for _ in range(int(iterations)):
        sample = far[rng.choice(far.shape[0], size=3, replace=False)]
        edge_a = sample[1] - sample[0]
        edge_b = sample[2] - sample[0]
        normal = np.cross(edge_a, edge_b)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal = normal / norm
        offset = -float(normal @ sample[0])
        distances = np.abs(score_pts @ normal + offset)
        inliers = int((distances < distance_threshold).sum())
        if inliers > best_inliers:
            best_inliers = inliers
            best = (normal, offset)
    if best is None:
        raise ValueError("plane fit failed (degenerate cloud)")

    normal, offset = best
    # Orient the normal toward the camera (origin): floor normal has n_z < 0 in
    # a positive-forward frame, so height above the floor is -(n.p + d).
    if normal[2] > 0:
        normal = -normal
        offset = -offset
    # Least-squares refinement on the inliers.
    inlier_mask = np.abs(pts @ normal + offset) < distance_threshold
    inlier_pts = pts[inlier_mask]
    if inlier_pts.shape[0] >= 3:
        centroid = inlier_pts.mean(axis=0)
        _u, _s, vh = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
        refined = vh[2]
        if refined[2] > 0:
            refined = -refined
        normal = refined
        offset = -float(refined @ centroid)
    return PlaneModel(normal=normal.astype(np.float64), offset=float(offset))


def _dominant_floor_component(
    floor_xy: np.ndarray,
    *,
    eps_m: float = 0.03,
    seed: int = 7,
    max_points: int = 20000,
) -> np.ndarray:
    """Largest connected xy component of the floor inliers.

    Plane inliers live on the *infinite* plane, so coplanar surfaces away from
    the bin (a conveyor, a shelf at the same height) would otherwise inflate
    the floor hull. Keep only the dominant connected patch - the bin floor.
    """

    from src.inference.instance_clustering import _dbscan

    xy = np.asarray(floor_xy, dtype=np.float32)
    if xy.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        xy = xy[rng.choice(xy.shape[0], size=max_points, replace=False)]
    labels = _dbscan(xy, eps_m, 10)
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    if ids.size == 0:
        return xy
    return xy[labels == ids[np.argmax(counts)]]


def _inside_eroded_hull(points_xy: np.ndarray, hull_xy: np.ndarray, erosion_m: float) -> np.ndarray:
    """Test xy membership in the convex hull of `hull_xy` shrunk inward by ~erosion_m."""

    from scipy.spatial import ConvexHull, Delaunay

    hull = ConvexHull(hull_xy)
    vertices = hull_xy[hull.vertices]
    centroid = vertices.mean(axis=0)
    radii = np.linalg.norm(vertices - centroid, axis=1)
    scale = np.clip(1.0 - erosion_m / np.maximum(radii, 1e-9), 0.0, 1.0)
    eroded = centroid + (vertices - centroid) * scale[:, None]
    return Delaunay(eroded).find_simplex(points_xy) >= 0


def crop_workspace(
    points: np.ndarray,
    config: WorkspaceCropConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return (kept indices, info) for the configured workspace crop."""

    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    if config.mode == "none":
        return np.arange(n, dtype=np.int64), {"mode": "none", "kept": n, "total": n}

    if config.mode == "bounds":
        keep = np.ones(n, dtype=bool)
        for axis, low, high in (
            (0, config.x_min, config.x_max),
            (1, config.y_min, config.y_max),
            (2, config.z_min, config.z_max),
        ):
            if low is not None:
                keep &= pts[:, axis] >= float(low)
            if high is not None:
                keep &= pts[:, axis] <= float(high)
        return np.flatnonzero(keep), {"mode": "bounds", "kept": int(keep.sum()), "total": n}

    if config.mode == "auto_plane":
        plane = fit_floor_plane(
            pts,
            distance_threshold=config.plane_distance_m,
            iterations=config.plane_iterations,
            seed=config.seed,
        )
        height = plane.signed_height(pts)
        floor_inliers = np.abs(height) < config.plane_distance_m
        keep = (height >= float(config.min_height_m)) & (height <= float(config.max_height_m))
        if floor_inliers.sum() >= 16 and config.hull_erosion_m >= 0:
            floor_patch = _dominant_floor_component(pts[floor_inliers][:, :2], seed=config.seed)
            keep &= _inside_eroded_hull(pts[:, :2], floor_patch, config.hull_erosion_m)
        info = {
            "mode": "auto_plane",
            "kept": int(keep.sum()),
            "total": n,
            "plane_normal": plane.normal.tolist(),
            "plane_offset": plane.offset,
            "floor_inliers": int(floor_inliers.sum()),
        }
        return np.flatnonzero(keep), info

    raise ValueError(f"Unknown workspace crop mode: {config.mode}")
