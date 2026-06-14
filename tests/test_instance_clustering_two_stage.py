"""Tests for stage-2 merge/split clustering. Runs under pytest or standalone:

    python tests/test_instance_clustering_two_stage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.instance_clustering import (  # noqa: E402
    TwoStageClusteringConfig,
    refine_instance_labels_two_stage,
)

DIAMETER = 0.14


def test_merge_oversegmented_object() -> None:
    rng = np.random.default_rng(0)
    pa = rng.normal([0.0, 0.0, 0.0], 0.01, size=(80, 3))
    pb = rng.normal([0.02, 0.0, 0.0], 0.01, size=(80, 3))
    points = np.vstack([pa, pb]).astype(np.float32)
    votes = rng.normal([0.01, 0.0, 0.0], 0.005, size=(160, 3)).astype(np.float32)
    labels = np.array([1] * 80 + [2] * 80, dtype=np.int64)
    config = TwoStageClusteringConfig(enabled=True, merge_center_fraction=0.5, merge_contiguity_fraction=0.3, split_separation_fraction=0.0)
    out = refine_instance_labels_two_stage(points, votes, labels, DIAMETER, config)
    assert len(set(int(v) for v in np.unique(out) if v > 0)) == 1


def test_split_two_objects_in_one_cluster() -> None:
    rng = np.random.default_rng(1)
    v1 = rng.normal([0.0, 0.0, 0.0], 0.005, size=(80, 3))
    v2 = rng.normal([0.13, 0.0, 0.0], 0.005, size=(80, 3))
    votes = np.vstack([v1, v2]).astype(np.float32)
    points = votes.copy()
    labels = np.ones(160, dtype=np.int64)
    config = TwoStageClusteringConfig(enabled=True, merge_center_fraction=0.0, split_separation_fraction=0.7, split_min_points=20)
    out = refine_instance_labels_two_stage(points, votes, labels, DIAMETER, config)
    assert len(set(int(v) for v in np.unique(out) if v > 0)) == 2


def test_disabled_is_noop() -> None:
    points = np.zeros((10, 3), dtype=np.float32)
    labels = np.array([1, 1, 2, 2, 0, 0, 1, 2, 1, 2], dtype=np.int64)
    out = refine_instance_labels_two_stage(points, points, labels, DIAMETER, TwoStageClusteringConfig(enabled=False))
    assert np.array_equal(out, labels)


def test_config_from_mapping() -> None:
    cfg = TwoStageClusteringConfig.from_mapping({"enabled": True, "merge_center_fraction": 0.4, "split_min_points": 50})
    assert cfg.enabled and cfg.merge_center_fraction == 0.4 and cfg.split_min_points == 50


def _run_all() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
