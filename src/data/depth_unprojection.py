"""Depth-image to point-cloud helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def unproject_depth_to_camera(
    depth_m: np.ndarray,
    intrinsics: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject valid depth pixels into positive-forward camera coordinates.

    The Blender generator stores `points_camera` as `(x, y, z_forward)`, where
    `z_forward` is positive in front of the camera. This helper follows the same
    convention so processed data stays compatible with raw metadata.
    """

    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth_m must be 2D, got {depth.shape}")

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])

    valid_v, valid_u = np.nonzero(depth > 0)
    z = depth[valid_v, valid_u].astype(np.float32)
    x = ((valid_u.astype(np.float32) - cx) / fx) * z
    y = -((valid_v.astype(np.float32) - cy) / fy) * z

    points = np.stack((x, y, z), axis=1).astype(np.float32)
    pixels = np.stack((valid_u, valid_v), axis=1).astype(np.uint16)
    return points, pixels
