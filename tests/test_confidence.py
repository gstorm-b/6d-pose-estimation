"""Tests for per-instance confidence and top-K pick metrics. Runs under pytest or:

    python tests/test_confidence.py
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

from src.inference.confidence import (  # noqa: E402
    ConfidenceConfig,
    cluster_isolation,
    compute_instance_confidence,
    top_k_pick_success,
)

MODELS_ROOT = PROJECT_ROOT / "models"
PROCESSED = PROJECT_ROOT / "processed-data" / "pointnet2_semseg_k41144" / "test"
RAW_ROOT = PROJECT_ROOT / "synthetic-data" / "K41144"


def test_confidence_monotonic_in_dispersion() -> None:
    config = ConfidenceConfig()
    low, _ = compute_instance_confidence({"center_vote_dispersion": 0.01, "keypoint_dispersion": 0.02, "point_count": 500}, config)
    high, _ = compute_instance_confidence({"center_vote_dispersion": 0.20, "keypoint_dispersion": 0.40, "point_count": 500}, config)
    assert low > high, f"low-dispersion confidence {low} should exceed high-dispersion {high}"
    assert 0.0 <= high <= low <= 1.0


def test_point_count_raises_confidence() -> None:
    config = ConfidenceConfig()
    few, _ = compute_instance_confidence({"center_vote_dispersion": 0.05, "keypoint_dispersion": 0.05, "point_count": 20}, config)
    many, _ = compute_instance_confidence({"center_vote_dispersion": 0.05, "keypoint_dispersion": 0.05, "point_count": 1000}, config)
    assert many > few


def test_model_fit_term_dominates() -> None:
    config = ConfidenceConfig()
    base = {"center_vote_dispersion": 0.05, "keypoint_dispersion": 0.05, "point_count": 500}
    good_fit, _ = compute_instance_confidence({**base, "model_fit": 0.02}, config)
    bad_fit, _ = compute_instance_confidence({**base, "model_fit": 0.5}, config)
    assert good_fit > bad_fit, "lower model-fit chamfer must raise confidence"

    # A well-fitting pose with otherwise weak signals must still outrank a poorly-
    # fitting pose with strong other signals (model-fit dominates the geometric mean).
    well_fit_weak, _ = compute_instance_confidence(
        {"center_vote_dispersion": 0.15, "keypoint_dispersion": 0.20, "point_count": 60, "model_fit": 0.02}, config
    )
    bad_fit_strong, _ = compute_instance_confidence(
        {"center_vote_dispersion": 0.01, "keypoint_dispersion": 0.01, "point_count": 2000, "model_fit": 0.6}, config
    )
    assert well_fit_weak > bad_fit_strong, "model-fit must dominate the other signals"


def test_cluster_isolation_values() -> None:
    centroids = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.0, 0.0, 0.0]])
    iso = cluster_isolation(centroids, diameter_m=0.1)
    assert np.isclose(iso[0], 1.0)  # nearest is 0.1 m away = 1 diameter
    assert np.isclose(iso[1], 1.0)
    assert iso[2] > iso[0]  # the far one is more isolated


def test_top_k_pick_success_logic() -> None:
    # Scene 1: top confidence is correct. Scene 2: top wrong but a top-3 is correct.
    scenes = [
        [(0.9, True), (0.5, False), (0.2, True)],
        [(0.9, False), (0.6, False), (0.3, True)],
    ]
    result = top_k_pick_success(scenes, k=3)
    assert result["scenes"] == 2.0
    assert result["top1_pick_success"] == 0.5  # scene1 yes, scene2 no
    assert result["top3_pick_success"] == 1.0  # both have a correct in top-3


def _prereqs() -> bool:
    return (MODELS_ROOT / "K41144").exists() and PROCESSED.exists() and bool(glob.glob(str(PROCESSED / "*.npz")))


def test_confidence_is_informative_on_sample() -> None:
    if not _prereqs():
        print("  (skip: missing bundles or processed sample)")
        return
    from src.inference.device import resolve_device
    from src.inference.pose_pipeline import PosePipeline
    from src.registry.model_registry import ModelRegistry
    from src.training.pose_geometry import resolve_symmetry_matrices

    registry = ModelRegistry(MODELS_ROOT)
    loaded = registry.load(registry.resolve("K41144"), device=resolve_device("cuda"))
    pipeline = PosePipeline.from_loaded_bundle(loaded)

    sample = sorted(glob.glob(str(PROCESSED / "*.npz")))[0]
    data = np.load(sample)
    points, features = data["points"].astype(np.float32), data["features"].astype(np.float32)
    labels = data["instance_labels"].astype(np.int64)
    raw = str(data["raw_sample"])
    metadata = json.loads((RAW_ROOT / raw / "metadata.json").read_text(encoding="utf-8"))

    result = pipeline.infer_from_points(points, features, gt_instance_labels=labels, metadata=metadata)

    # Instances must be ranked highest-confidence first.
    confs = [inst.confidence for inst in result.instances]
    assert all(c is not None for c in confs)
    assert confs == sorted(confs, reverse=True), "instances are not sorted by confidence"

    # Confidence should be informative: higher-confidence half has lower ADD error.
    model_points = loaded.model_points_object.astype(np.float64)
    symmetry = resolve_symmetry_matrices("k41144").astype(np.float64)
    gt_by_id = {int(i["instance_id"]): np.asarray(i["object_to_camera"], dtype=np.float64) for i in metadata["instances"]}
    adds = []
    for inst in result.instances:
        gt = gt_by_id.get(inst.instance_id)
        if gt is None:
            adds.append(np.nan)
            continue
        Rp, tp = inst.object_to_camera[:3, :3], inst.object_to_camera[:3, 3]
        Rg, tg = gt[:3, :3], gt[:3, 3]
        pred = model_points @ Rp.T + tp
        adds.append(min(float(np.mean(np.linalg.norm(pred - (model_points @ (Rg @ S).T + tg), axis=1))) for S in symmetry) * 1000.0)
    adds = np.asarray(adds)
    half = len(adds) // 2
    top_half_add = np.nanmean(adds[:half])
    bottom_half_add = np.nanmean(adds[half:])
    print(f"  ADD top-confidence half={top_half_add:.2f} mm  bottom half={bottom_half_add:.2f} mm")
    assert top_half_add <= bottom_half_add + 1.0, "confidence ranking is not informative (top half not better)"

    # Top-1 / top-3 pick success on this scene (correct = ADD < 0.1 diameter).
    threshold = 0.1 * loaded.diameter_m * 1000.0
    scene = [(inst.confidence, adds[i] < threshold) for i, inst in enumerate(result.instances) if np.isfinite(adds[i])]
    metric = top_k_pick_success([scene], k=3)
    print(f"  top1={metric['top1_pick_success']:.0f} top3={metric['top3_pick_success']:.0f} (threshold {threshold:.1f} mm)")


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
