"""Tests for the classical (plane removal + clustering) baseline segmenter."""

from __future__ import annotations

import numpy as np

from src.inference.classical_segmentation import (
    ClassicalSegmentationConfig,
    segment_scene_classical,
)


def _cylinder_on_floor(
    rng: np.random.Generator,
    *,
    center_xy: tuple[float, float],
    axis: str = "y",
    radius: float = 0.012,
    length: float = 0.12,
    floor_z: float = 0.6,
    points: int = 3000,
) -> np.ndarray:
    """Top surface of a cylinder lying on the floor along `axis` (camera view)."""

    along = rng.uniform(-length / 2, length / 2, size=points)
    theta = rng.uniform(-np.pi / 2, np.pi / 2, size=points)  # visible upper half
    across = radius * np.sin(theta)
    lift = radius + radius * np.cos(theta)  # 0..2r above the floor
    x = center_xy[0] + (across if axis == "y" else along)
    y = center_xy[1] + (along if axis == "y" else across)
    z = floor_z - lift
    return np.stack([x, y, z], axis=1) + rng.normal(0.0, 0.0004, size=(points, 3))


def _floor(rng: np.random.Generator, *, floor_z: float = 0.6, points: int = 12000) -> np.ndarray:
    return np.stack(
        [
            rng.uniform(-0.2, 0.2, size=points),
            rng.uniform(-0.15, 0.15, size=points),
            np.full(points, floor_z) + rng.normal(0.0, 0.0005, size=points),
        ],
        axis=1,
    )


def test_separated_parts_become_instances_ranked_topmost_first():
    rng = np.random.default_rng(5)
    part_low = _cylinder_on_floor(rng, center_xy=(-0.08, 0.0))
    # Raise the second part as if resting on another: it must rank first.
    part_high = _cylinder_on_floor(rng, center_xy=(0.08, 0.0)) - np.array([0.0, 0.0, 0.02])
    scene = np.concatenate([_floor(rng), part_low, part_high]).astype(np.float32)

    config = ClassicalSegmentationConfig(min_cluster_points=200, voxel_size_m=0.0)
    result = segment_scene_classical(scene, config)

    assert result.instance_count == 2
    top = result.instances[0]
    assert top.instance_id == 1
    # Topmost = closest to the camera = smallest z centroid.
    assert top.centroid_camera[2] < result.instances[1].centroid_camera[2]
    # Floor stays background.
    floor_labels = result.instance_labels[: 12000]
    assert (floor_labels == 0).mean() > 0.99


def test_touching_parallel_parts_split_by_crest():
    rng = np.random.default_rng(9)
    radius = 0.012
    # Two cylinders side by side, flanks touching (centers 2r apart).
    part_a = _cylinder_on_floor(rng, center_xy=(-radius, 0.0), radius=radius)
    part_b = _cylinder_on_floor(rng, center_xy=(radius, 0.0), radius=radius)
    scene = np.concatenate([_floor(rng), part_a, part_b]).astype(np.float32)

    with_split = segment_scene_classical(
        scene, ClassicalSegmentationConfig(min_cluster_points=200, voxel_size_m=0.0, crest_split=True)
    )
    without_split = segment_scene_classical(
        scene, ClassicalSegmentationConfig(min_cluster_points=200, voxel_size_m=0.0, crest_split=False)
    )
    assert without_split.instance_count == 1  # Euclidean clustering merges touching parts
    assert with_split.instance_count == 2

    # The split must follow the parts: each cylinder's points map to one instance.
    labels = with_split.instance_labels
    labels_a = labels[12000 : 12000 + 3000]
    labels_b = labels[12000 + 3000 :]
    majority_a = np.bincount(labels_a[labels_a > 0]).argmax()
    majority_b = np.bincount(labels_b[labels_b > 0]).argmax()
    assert majority_a != majority_b
    assert (labels_a == majority_a).mean() > 0.8
    assert (labels_b == majority_b).mean() > 0.8


def test_empty_above_floor_returns_no_instances():
    rng = np.random.default_rng(2)
    result = segment_scene_classical(_floor(rng).astype(np.float32), ClassicalSegmentationConfig(voxel_size_m=0.0))
    assert result.instance_count == 0
    assert (result.instance_labels == 0).all()
