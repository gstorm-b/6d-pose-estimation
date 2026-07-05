"""Tests for real-camera .pcd loading and convention conversion."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.pcd_loading import convert_to_positive_forward, read_pcd


def _write_binary_pcd(path, xyz: np.ndarray, width: int, height: int) -> None:
    n = xyz.shape[0]
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F U\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {width}\n"
        f"HEIGHT {height}\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    )
    record = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<u4")])
    record["x"], record["y"], record["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    record["intensity"] = np.arange(n, dtype=np.uint32)
    with open(path, "wb") as handle:
        handle.write(header.encode("ascii"))
        record.tofile(handle)


def test_read_pcd_roundtrip_organized(tmp_path):
    rng = np.random.default_rng(1)
    width, height = 8, 4
    xyz = rng.normal(0.0, 100.0, size=(width * height, 3))
    xyz[5] = np.nan  # one invalid pixel
    xyz[9] = 0.0     # one all-zero pixel
    path = tmp_path / "scene.pcd"
    _write_binary_pcd(path, xyz, width, height)

    cloud = read_pcd(path)
    assert cloud.organized
    assert cloud.width == width and cloud.height == height
    np.testing.assert_allclose(cloud.xyz[0], xyz[0], rtol=1e-6)
    assert "intensity" in cloud.fields

    valid = cloud.valid_mask()
    assert not valid[5] and not valid[9]
    assert valid.sum() == width * height - 2

    pixels = cloud.pixel_grid()
    assert pixels[9].tolist() == [9 % width, 9 // width]


def test_convert_millimetre_negative_z_cloud():
    rng = np.random.default_rng(4)
    # Basler-style: millimetres, camera looking down -z (points at z ~ -600 mm).
    points_mm = np.stack(
        [
            rng.uniform(-150.0, 150.0, size=500),
            rng.uniform(-100.0, 100.0, size=500),
            rng.uniform(-650.0, -550.0, size=500),
        ],
        axis=1,
    )
    converted, info = convert_to_positive_forward(points_mm)
    assert info["in_mm"] and info["flip_z"]
    assert converted[:, 2].min() > 0.5 and converted[:, 2].max() < 0.7
    np.testing.assert_allclose(converted[:, 0], points_mm[:, 0] / 1000.0, rtol=1e-9)


def test_convert_metre_positive_forward_cloud_untouched():
    rng = np.random.default_rng(6)
    points = np.stack(
        [
            rng.uniform(-0.2, 0.2, size=300),
            rng.uniform(-0.2, 0.2, size=300),
            rng.uniform(0.4, 0.7, size=300),
        ],
        axis=1,
    )
    converted, info = convert_to_positive_forward(points)
    assert not info["in_mm"] and not info["flip_z"]
    np.testing.assert_allclose(converted, points, rtol=1e-12)


def test_read_pcd_rejects_truncated_file(tmp_path):
    xyz = np.zeros((10, 3))
    path = tmp_path / "broken.pcd"
    _write_binary_pcd(path, xyz, 10, 1)
    data = path.read_bytes()
    path.write_bytes(data[:-20])
    with pytest.raises(ValueError, match="truncated"):
        read_pcd(path)
