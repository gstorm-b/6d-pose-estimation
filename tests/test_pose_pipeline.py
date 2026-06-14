"""Golden-consistency tests for the in-memory pose pipeline.

Requires packaged bundles (models/), a processed K41144 sample, and its raw
sample. Verifies that the pipeline's GT-label path reproduces the offline pose
quality, and that the depth -> poses predicted path runs end to end. Runs under
pytest or standalone:

    python tests/test_pose_pipeline.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODELS_ROOT = PROJECT_ROOT / "models"
PROCESSED = PROJECT_ROOT / "processed-data" / "pointnet2_semseg_k41144" / "test"
RAW_ROOT = PROJECT_ROOT / "synthetic-data" / "K41144"


def _prereqs() -> bool:
    return (MODELS_ROOT / "K41144").exists() and PROCESSED.exists() and bool(glob.glob(str(PROCESSED / "*.npz")))


def _load_pipeline():
    from src.inference.pose_pipeline import PosePipeline
    from src.registry.model_registry import ModelRegistry

    registry = ModelRegistry(MODELS_ROOT)
    loaded = registry.load(registry.resolve("K41144"), device="cpu")
    return PosePipeline.from_loaded_bundle(loaded), loaded


def _symmetry_aware_add_mm(model_points, pred, gt, symmetry):
    Rp, tp = pred[:3, :3], pred[:3, 3]
    Rg, tg = gt[:3, :3], gt[:3, 3]
    transformed_pred = model_points @ Rp.T + tp
    best = min(
        float(np.mean(np.linalg.norm(transformed_pred - (model_points @ (Rg @ S).T + tg), axis=1)))
        for S in symmetry
    )
    return best * 1000.0


def test_gt_label_path_matches_offline_quality() -> None:
    if not _prereqs():
        print("  (skip: missing bundles or processed K41144 sample)")
        return
    from src.training.pose_geometry import resolve_symmetry_matrices

    pipeline, loaded = _load_pipeline()
    sample = sorted(glob.glob(str(PROCESSED / "*.npz")))[0]
    data = np.load(sample)
    points = data["points"].astype(np.float32)
    features = data["features"].astype(np.float32)
    labels = data["instance_labels"].astype(np.int64)
    raw = str(data["raw_sample"])
    metadata = json.loads((RAW_ROOT / raw / "metadata.json").read_text(encoding="utf-8"))

    result = pipeline.infer_from_points(points, features, gt_instance_labels=labels, metadata=metadata)
    gt_count = len([v for v in np.unique(labels) if int(v) > 0])
    assert len(result.instances) == gt_count, f"instance count {len(result.instances)} != gt {gt_count}"
    assert all(np.all(np.isfinite(inst.object_to_camera)) for inst in result.instances)

    model_points = loaded.model_points_object.astype(np.float64)
    symmetry = resolve_symmetry_matrices("k41144").astype(np.float64)
    gt_by_id = {int(i["instance_id"]): np.asarray(i["object_to_camera"], dtype=np.float64) for i in metadata["instances"]}
    adds = []
    for inst in result.instances:
        gt = gt_by_id.get(inst.instance_id)
        if gt is not None:
            adds.append(_symmetry_aware_add_mm(model_points, inst.object_to_camera, gt, symmetry))
    adds = np.asarray(adds)
    print(f"  GT-path ADD mean={adds.mean():.2f} mm median={np.median(adds):.2f} mm (offline GT-crop ref ~5.6 mm)")
    # Loose gate: in-memory pipeline must reach offline-comparable quality.
    assert adds.mean() < 15.0, f"ADD mean {adds.mean():.2f} mm too high vs offline ~5.6 mm"


def test_depth_predicted_path_runs() -> None:
    if not _prereqs():
        print("  (skip: missing prerequisites)")
        return
    pipeline, _loaded = _load_pipeline()
    sample = sorted(glob.glob(str(PROCESSED / "*.npz")))[0]
    raw = str(np.load(sample)["raw_sample"])
    raw_npz = RAW_ROOT / raw / "sensor_data.npz"
    metadata = json.loads((RAW_ROOT / raw / "metadata.json").read_text(encoding="utf-8"))
    if not raw_npz.exists():
        print("  (skip: raw sensor_data.npz not present)")
        return
    sensor = np.load(raw_npz)
    depth = sensor["depth_m"]
    normal_camera = sensor["normal_camera"] if "normal_camera" in sensor.files else None
    intrinsics = metadata["camera_intrinsics"]

    result = pipeline.infer_scene(depth, intrinsics, normal_camera)
    print(f"  depth->poses predicted instances={len(result.instances)} timings={ {k: round(v,1) for k,v in result.timings_ms.items()} }")
    assert all(np.all(np.isfinite(inst.object_to_camera)) for inst in result.instances)


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
