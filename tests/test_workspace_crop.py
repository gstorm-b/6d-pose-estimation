"""Tests for workspace cropping and the floor-plane fit."""

from __future__ import annotations

import numpy as np
import pytest

from src.inference.workspace_crop import (
    WorkspaceCropConfig,
    crop_workspace,
    fit_floor_plane,
)


def _bin_scene(rng: np.random.Generator, *, floor_z: float = 0.6) -> dict[str, np.ndarray]:
    """A synthetic top-down bin: floor plane, one raised part, far outliers."""

    floor = np.stack(
        [
            rng.uniform(-0.2, 0.2, size=4000),
            rng.uniform(-0.15, 0.15, size=4000),
            np.full(4000, floor_z) + rng.normal(0.0, 0.0005, size=4000),
        ],
        axis=1,
    )
    part = np.stack(
        [
            rng.uniform(-0.03, 0.03, size=800),
            rng.uniform(-0.03, 0.03, size=800),
            np.full(800, floor_z - 0.03) + rng.normal(0.0, 0.0005, size=800),
        ],
        axis=1,
    )
    outside = np.stack(
        [
            rng.uniform(0.4, 0.5, size=300),
            rng.uniform(0.4, 0.5, size=300),
            rng.uniform(0.3, 0.9, size=300),
        ],
        axis=1,
    )
    return {"floor": floor, "part": part, "outside": outside}


def test_fit_floor_plane_recovers_top_down_floor():
    rng = np.random.default_rng(3)
    scene = _bin_scene(rng)
    points = np.concatenate([scene["floor"], scene["part"]])
    plane = fit_floor_plane(points, seed=3)

    # Normal oriented toward the camera means n_z < 0 in a positive-forward frame.
    assert plane.normal[2] < 0
    assert abs(abs(plane.normal[2]) - 1.0) < 0.01

    # Signed height: ~0 on the floor, positive (toward camera) for the raised part.
    floor_height = plane.signed_height(scene["floor"])
    part_height = plane.signed_height(scene["part"])
    assert abs(float(np.median(floor_height))) < 0.002
    assert float(np.median(part_height)) == pytest.approx(0.03, abs=0.005)


def test_crop_bounds_mode_keeps_only_box():
    points = np.array(
        [
            [0.0, 0.0, 0.5],
            [0.3, 0.0, 0.5],   # outside x_max
            [0.0, -0.4, 0.5],  # outside y_min
            [0.0, 0.0, 0.9],   # outside z_max
        ]
    )
    config = WorkspaceCropConfig(mode="bounds", x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z_min=0.3, z_max=0.7)
    keep, info = crop_workspace(points, config)
    assert keep.tolist() == [0]
    assert info["kept"] == 1 and info["total"] == 4


def test_crop_auto_plane_drops_floor_and_out_of_hull():
    rng = np.random.default_rng(11)
    scene = _bin_scene(rng)
    points = np.concatenate([scene["floor"], scene["part"], scene["outside"]])
    config = WorkspaceCropConfig(mode="auto_plane", min_height_m=0.01, max_height_m=0.2, hull_erosion_m=0.01, seed=11)
    keep, info = crop_workspace(points, config)

    kept = set(keep.tolist())
    part_indices = set(range(len(scene["floor"]), len(scene["floor"]) + len(scene["part"])))
    outside_indices = set(range(len(points) - len(scene["outside"]), len(points)))
    # The raised part survives; the floor (height ~0 < min_height) and the
    # out-of-hull clutter do not.
    assert len(kept & part_indices) > 0.9 * len(part_indices)
    assert not kept & outside_indices
    assert info["kept"] < len(scene["floor"])


def test_crop_none_keeps_everything():
    points = np.zeros((5, 3))
    keep, info = crop_workspace(points, WorkspaceCropConfig(mode="none"))
    assert keep.shape[0] == 5
    assert info["mode"] == "none"
