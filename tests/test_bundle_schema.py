"""Tests for the model bundle manifest schema. Runs under pytest or standalone:

    python tests/test_bundle_schema.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.registry.bundle_schema import (  # noqa: E402
    SCHEMA_VERSION,
    BundleManifest,
    BundleValidationError,
    ModelInfo,
    read_manifest,
    write_manifest,
)


def _model_info() -> ModelInfo:
    return ModelInfo(
        class_name="K41144",
        model_path="object-model/K41144.stl",
        model_scale=1.0,
        symmetry_name="k41144",
        diameter_m=0.1003,
        model_point_count=1024,
        keypoint_count=8,
    )


def _manifest() -> BundleManifest:
    return BundleManifest(
        schema_version=SCHEMA_VERSION,
        sku="K41144",
        version="v1",
        created="2026-06-14T00:00:00",
        model=_model_info(),
        files={
            "instance_seg": "instance_seg.pt",
            "pose_voting": "pose_voting.pt",
            "model_points": "model_points.npy",
            "keypoints": "keypoints.npy",
            "model_stl": "model.stl",
        },
        inference={"num_points": 16384, "dbscan_eps_m": 0.004},
    )


def _make_bundle(tmp: Path) -> Path:
    bundle = tmp / "K41144" / "v1"
    bundle.mkdir(parents=True)
    for name in ("instance_seg.pt", "pose_voting.pt", "model_points.npy", "keypoints.npy", "model.stl"):
        (bundle / name).write_bytes(b"stub")
    return bundle


def test_roundtrip_to_from_dict() -> None:
    manifest = _manifest()
    restored = BundleManifest.from_dict(manifest.to_dict())
    assert restored.sku == "K41144"
    assert restored.version == "v1"
    assert restored.model.diameter_m == manifest.model.diameter_m
    assert restored.model.symmetry_name == "k41144"
    assert restored.files["model_stl"] == "model.stl"


def test_validate_and_write_read(tmp_path: Path | None = None) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        bundle = _make_bundle(tmp_dir)
        write_manifest(bundle, _manifest())
        loaded = read_manifest(bundle, check_files=True)
        assert loaded.model.class_name == "K41144"
        assert loaded.inference["dbscan_eps_m"] == 0.004


def test_missing_required_file_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "K41144" / "v1"
        bundle.mkdir(parents=True)
        # Only some of the required files exist.
        (bundle / "instance_seg.pt").write_bytes(b"stub")
        try:
            _manifest().validate(bundle, check_files=True)
        except BundleValidationError:
            return
        raise AssertionError("expected BundleValidationError for missing files")


def test_bad_schema_version_rejected() -> None:
    manifest = _manifest()
    manifest.schema_version = 999
    try:
        manifest.validate(check_files=False)
    except BundleValidationError:
        return
    raise AssertionError("expected BundleValidationError for bad schema_version")


def test_missing_manifest_field_rejected() -> None:
    data = _manifest().to_dict()
    del data["sku"]
    try:
        BundleManifest.from_dict(data)
    except BundleValidationError:
        return
    raise AssertionError("expected BundleValidationError for missing sku")


def test_invalid_model_info_rejected() -> None:
    manifest = _manifest()
    manifest.model.model_scale = 0.0
    try:
        manifest.validate(check_files=False)
    except BundleValidationError:
        return
    raise AssertionError("expected BundleValidationError for model_scale=0")


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
