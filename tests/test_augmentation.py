"""Unit tests for the structured-light sim-to-real augmentation (P8).

Pure numpy, no models or data needed. Runs under pytest or standalone:

    python tests/test_augmentation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augmentation import PointCloudAugmentationConfig, augment_point_cloud  # noqa: E402


def _scene(n: int = 2048, seed: int = 0):
    rng = np.random.default_rng(seed)
    points = (rng.random((n, 3)).astype(np.float32) * 0.2) + np.array([0, 0, 0.5], dtype=np.float32)
    normals = rng.standard_normal((n, 3)).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    semantic = (rng.random(n) > 0.5).astype(np.int64)
    instance = rng.integers(0, 5, size=n).astype(np.int64)
    return points, normals, semantic, instance


def test_disabled_is_passthrough() -> None:
    points, normals, sem, inst = _scene()
    cfg = PointCloudAugmentationConfig(enabled=False, depth_quadratic_noise=0.01)
    p, f, s, i = augment_point_cloud(points, normals, sem, inst, cfg)
    assert p is points and f is normals and s is sem and i is inst


def test_fixed_size_and_finite() -> None:
    points, normals, sem, inst = _scene()
    cfg = PointCloudAugmentationConfig(
        enabled=True,
        depth_quadratic_noise=0.01,
        depth_quantization_m=0.001,
        incidence_dropout_max_prob=0.5,
        edge_dropout_prob=0.3,
        edge_dropout_gap_m=0.01,
        blob_dropout_count=3,
        blob_dropout_radius_m=0.02,
        outlier_ratio=0.01,
    )
    p, f, s, i = augment_point_cloud(points, normals, sem, inst, cfg, np.random.default_rng(1),
                                     camera_position=np.array([0, 0, 0], dtype=np.float32))
    assert p.shape == points.shape and f.shape == normals.shape
    assert s.shape == sem.shape and i.shape == inst.shape
    assert np.all(np.isfinite(p)) and np.all(np.isfinite(f))
    assert s.dtype == sem.dtype and i.dtype == inst.dtype


def test_determinism() -> None:
    points, normals, sem, inst = _scene()
    cfg = PointCloudAugmentationConfig(enabled=True, depth_quadratic_noise=0.02, blob_dropout_count=2,
                                       blob_dropout_radius_m=0.03, incidence_dropout_max_prob=0.4)
    cam = np.array([0, 0, 0], dtype=np.float32)
    a = augment_point_cloud(points, normals, sem, inst, cfg, np.random.default_rng(7), camera_position=cam)[0]
    b = augment_point_cloud(points, normals, sem, inst, cfg, np.random.default_rng(7), camera_position=cam)[0]
    c = augment_point_cloud(points, normals, sem, inst, cfg, np.random.default_rng(8), camera_position=cam)[0]
    assert np.array_equal(a, b), "same seed must be deterministic"
    assert not np.array_equal(a, c), "different seed must differ"


def test_quantization_snaps_depth() -> None:
    points, normals, sem, inst = _scene()
    step = 0.01
    cfg = PointCloudAugmentationConfig(enabled=True, depth_quantization_m=step)
    p, _f, _s, _i = augment_point_cloud(points, normals, sem, inst, cfg, np.random.default_rng(0))
    remainder = np.abs(p[:, 2] / step - np.round(p[:, 2] / step))
    assert float(remainder.max()) < 1e-4, "depth must be quantized to the step grid"


def test_incidence_dropout_targets_grazing() -> None:
    # Facing points (normal toward camera) in x<0.075; grazing points (normal _|_ ray) in x>0.075.
    cam = np.array([0, 0, 0], dtype=np.float32)
    n_each = 1000
    rng = np.random.default_rng(0)
    facing_xyz = np.column_stack([
        rng.uniform(0.0, 0.05, n_each), rng.uniform(0.0, 0.05, n_each), np.full(n_each, 0.5)
    ]).astype(np.float32)
    grazing_xyz = np.column_stack([
        rng.uniform(0.10, 0.15, n_each), rng.uniform(0.0, 0.05, n_each), np.full(n_each, 0.5)
    ]).astype(np.float32)
    facing_n = -facing_xyz / np.linalg.norm(facing_xyz, axis=1, keepdims=True)  # points back at camera
    ray_g = grazing_xyz / np.linalg.norm(grazing_xyz, axis=1, keepdims=True)
    grazing_n = np.cross(ray_g, np.array([0, 0, 1.0]))                          # perpendicular to the ray
    grazing_n /= np.linalg.norm(grazing_n, axis=1, keepdims=True)

    points = np.vstack([facing_xyz, grazing_xyz]).astype(np.float32)
    normals = np.vstack([facing_n, grazing_n]).astype(np.float32)
    sem = np.ones(2 * n_each, dtype=np.int64)
    inst = np.ones(2 * n_each, dtype=np.int64)

    cfg = PointCloudAugmentationConfig(enabled=True, incidence_dropout_max_prob=1.0, incidence_dropout_power=1.0)
    p, _f, _s, _i = augment_point_cloud(points, normals, sem, inst, cfg, np.random.default_rng(3), camera_position=cam)
    grazing_remaining = int(np.sum(p[:, 0] > 0.075))
    facing_remaining = int(np.sum(p[:, 0] <= 0.075))
    print(f"  incidence: facing_remaining={facing_remaining} grazing_remaining={grazing_remaining}")
    assert grazing_remaining < n_each * 0.1, "grazing points should be almost entirely dropped"
    assert facing_remaining > n_each * 1.5, "facing points should dominate after refill"


def test_blob_dropout_changes_cloud() -> None:
    points, normals, sem, inst = _scene()
    cfg = PointCloudAugmentationConfig(enabled=True, blob_dropout_count=1, blob_dropout_radius_m=0.05)
    p, _f, _s, _i = augment_point_cloud(points, normals, sem, inst, cfg, np.random.default_rng(0))
    assert p.shape == points.shape
    assert not np.array_equal(p, points), "a specular blob hole should change the cloud"


def _run_all() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
