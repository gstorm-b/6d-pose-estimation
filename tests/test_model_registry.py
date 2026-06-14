"""Tests for the model registry: discovery, resolution, loading, LRU eviction.

Requires the packaged bundles under models/ (scripts/package_model_bundle.py)
and torch. Runs under pytest or standalone:

    python tests/test_model_registry.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.registry.bundle_schema import BundleValidationError  # noqa: E402
from src.registry.model_registry import ModelRegistry  # noqa: E402

MODELS_ROOT = PROJECT_ROOT / "models"


def _registry(**kwargs) -> ModelRegistry:
    return ModelRegistry(MODELS_ROOT, **kwargs)


def _have_bundles() -> bool:
    return MODELS_ROOT.exists() and any((MODELS_ROOT / sku).is_dir() for sku in ("K41144", "bending_pipe"))


def test_list_skus() -> None:
    if not _have_bundles():
        print("  (skip: no packaged bundles under models/)")
        return
    skus = _registry().list_skus()
    assert "K41144" in skus, skus
    assert "bending_pipe" in skus, skus


def test_resolve_latest_and_manifest() -> None:
    if not _have_bundles():
        print("  (skip: no packaged bundles)")
        return
    bundle = _registry().resolve("K41144")
    assert bundle.sku == "K41144"
    assert bundle.version  # resolved to a concrete version
    assert bundle.manifest.model.symmetry_name == "k41144"
    assert bundle.manifest.model.diameter_m > 0


def test_resolve_unknown_sku_rejected() -> None:
    try:
        _registry().resolve("does_not_exist")
    except BundleValidationError:
        return
    raise AssertionError("expected BundleValidationError for unknown sku")


def test_load_on_cpu() -> None:
    if not _have_bundles():
        print("  (skip: no packaged bundles)")
        return
    registry = _registry()
    bundle = registry.resolve("K41144")
    loaded = registry.load(bundle, device="cpu")
    assert loaded.device == "cpu"
    assert loaded.instance_model is not None
    assert loaded.pose_model is not None
    assert loaded.model_points_object.ndim == 2 and loaded.model_points_object.shape[1] == 3
    assert loaded.keypoints_object.shape[0] == bundle.manifest.model.keypoint_count
    assert loaded.diameter_m > 0
    # Cache hit returns the same object.
    again = registry.load(bundle, device="cpu")
    assert again is loaded


def test_lru_eviction() -> None:
    if not _have_bundles():
        print("  (skip: no packaged bundles)")
        return
    registry = _registry(vram_budget_bundles=1)
    registry.load(registry.resolve("K41144"), device="cpu")
    registry.load(registry.resolve("bending_pipe"), device="cpu")
    assert len(registry._cache) == 1, f"expected 1 cached bundle, got {len(registry._cache)}"


def _run_all() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
