"""Tests for the natural/voxel sampling modes used to build processed datasets."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.pointnet2_processing import (
    PointNet2ConversionConfig,
    natural_sample_indices,
    voxel_sample_indices,
)


def test_natural_sampling_is_uniform_and_unique():
    rng = np.random.default_rng(0)
    indices = natural_sample_indices(100_000, num_points=4096, rng=rng)
    assert indices.shape == (4096,)
    assert np.unique(indices).size == 4096  # no duplicated points when enough exist


def test_natural_sampling_preserves_class_prior():
    rng = np.random.default_rng(1)
    labels = np.zeros(50_000, dtype=np.int64)
    labels[:5_000] = 1  # 10% object
    indices = natural_sample_indices(labels.size, num_points=8192, rng=rng)
    fraction = float((labels[indices] == 1).mean())
    assert fraction == pytest.approx(0.10, abs=0.02)


def test_voxel_sampling_normalizes_density():
    rng = np.random.default_rng(2)
    # A dense patch (0.4 mm pitch, like a real sensor) and a sparse patch
    # (2 mm pitch): after 2 mm voxel sampling both should contribute roughly
    # proportionally to their *area*, not their raw point count.
    dense = np.stack(
        [v.ravel() for v in np.meshgrid(np.arange(0, 0.1, 0.0004), np.arange(0, 0.1, 0.0004))]
        + [np.full(250 * 250, 0.5).ravel()],
        axis=1,
    )
    sparse = np.stack(
        [v.ravel() + 0.2 for v in np.meshgrid(np.arange(0, 0.1, 0.002), np.arange(0, 0.1, 0.002))]
        + [np.full(50 * 50, 0.5).ravel()],
        axis=1,
    )
    points = np.concatenate([dense, sparse])
    indices = voxel_sample_indices(points, num_points=4000, voxel_size_m=0.002, rng=rng)
    from_dense = int((indices < dense.shape[0]).sum())
    # Equal areas -> roughly half the sample each, despite dense having 25x the points.
    assert 0.4 < from_dense / 4000 < 0.6
    assert indices.shape == (4000,)


def test_voxel_sampling_fills_when_short():
    rng = np.random.default_rng(3)
    points = rng.uniform(0, 0.01, size=(500, 3))  # fewer voxels than requested
    indices = voxel_sample_indices(points, num_points=1024, voxel_size_m=0.002, rng=rng)
    assert indices.shape == (1024,)
    assert indices.max() < 500


def test_config_rejects_unknown_modes():
    with pytest.raises(ValueError, match="sampling_mode"):
        PointNet2ConversionConfig(sampling_mode="bogus")
    with pytest.raises(ValueError, match="normals_source"):
        PointNet2ConversionConfig(normals_source="bogus")
