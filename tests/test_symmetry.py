"""Tests for config-driven symmetry. Runs under pytest or standalone:

    python tests/test_symmetry.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.pose_geometry import (  # noqa: E402
    axis_angle_to_matrix,
    load_symmetry,
    resolve_symmetry_matrices,
    rotation_y_np,
    symmetry_matrices_for_name,
)


def test_json_matches_builtin_for_existing_skus() -> None:
    # Gate: switching to JSON-driven symmetry must not change the matrices.
    for name in ("k41144", "bending_pipe"):
        builtin = symmetry_matrices_for_name(name)
        loaded = load_symmetry(name).matrices
        assert builtin.shape == loaded.shape, f"{name} shape: {builtin.shape} vs {loaded.shape}"
        assert np.allclose(builtin, loaded, atol=1e-6), f"{name} matrices differ"


def test_axis_angle_y180() -> None:
    rod = axis_angle_to_matrix([0, 1, 0], np.pi)
    assert np.allclose(rod, np.diag([-1.0, 1.0, -1.0]), atol=1e-6)
    assert np.allclose(rod, rotation_y_np(np.pi), atol=1e-6)


def test_identity_is_first_and_k41144_has_two() -> None:
    matrices = load_symmetry("k41144").matrices
    assert matrices.shape == (2, 3, 3)
    assert np.allclose(matrices[0], np.eye(3), atol=1e-7)


def test_continuous_expansion() -> None:
    definition = load_symmetry(
        {"name": "revolute_z", "finite_rotations": [], "continuous_axes": [{"axis": [0, 0, 1], "samples": 24}]}
    )
    assert definition.has_continuous
    assert definition.matrices.shape[0] == 24  # identity + 23 non-identity samples
    assert np.allclose(definition.matrices[0], np.eye(3), atol=1e-7)


def test_resolve_from_dict_and_array() -> None:
    matrices = resolve_symmetry_matrices({"name": "x", "finite_rotations": [{"axis": [1, 0, 0], "angle_deg": 180}]})
    assert matrices.shape == (2, 3, 3)
    passthrough = resolve_symmetry_matrices(np.eye(3).reshape(1, 3, 3))
    assert passthrough.shape == (1, 3, 3)


def test_unknown_name_raises() -> None:
    try:
        resolve_symmetry_matrices("totally_unknown_object")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown symmetry name")


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
