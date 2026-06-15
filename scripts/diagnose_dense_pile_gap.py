"""Diagnose the predicted-path pose gap on dense piles and measure the operational
bin-picking metric (top-K pick success).

The GT-crop pose eval gates releases at ADD ~4.6 mm, but the full predicted path
degrades on dense ~40-object piles. The pose model is sound (GT crops reproduce
~4 mm in-memory), so the loss is crop quality. Findings: predicted clusters are
pure but the pose still flips on a fraction of them; the single strongest
correct-vs-wrong signal is model-fit (does the posed model land on its own crop),
which the confidence score now weights heavily.

Correctness here is matching-free and join-free (avoids GPU non-determinism
between separate clustering calls): a predicted pose is correct iff its
symmetry-aware ADD to the *nearest* GT object is below 0.1 diameter. We then
report top-1/top-3 pick success when ranking by confidence vs by raw model-fit.

    python scripts/diagnose_dense_pile_gap.py --sku bending_pipe --limit 25
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _best_add_mm(model_points, pred, gt_list, symmetry) -> float:
    """Minimum symmetry-aware ADD (mm) from one predicted pose to any GT object."""
    transformed = model_points @ pred[:3, :3].T + pred[:3, 3]
    best = np.inf
    for gt in gt_list:
        for S in symmetry:
            ref = model_points @ (gt[:3, :3] @ S).T + gt[:3, 3]
            best = min(best, float(np.mean(np.linalg.norm(transformed - ref, axis=1))))
    return best * 1000.0


def _topk(scene_instances, key, ascending, k, correct_key="correct"):
    ranked = sorted(scene_instances, key=lambda e: e[key], reverse=not ascending)
    top1 = bool(ranked and ranked[0][correct_key])
    topk = any(e[correct_key] for e in ranked[:k])
    return top1, topk


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose dense-pile predicted-path pose gap")
    parser.add_argument("--sku", default="bending_pipe")
    parser.add_argument("--processed", default="processed-data/pointnet2_semseg_bending_pipe_wave2/test")
    parser.add_argument("--raw-root", default="synthetic-data")
    parser.add_argument("--registry-root", default="models")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--out", default="experiments/dense_pile_gap")
    args = parser.parse_args()

    from src.data.pose_crop_dataset import rigid_object_to_camera_from_scaled
    from src.inference.device import resolve_device
    from src.inference.pose_bridge import load_raw_metadata
    from src.inference.pose_pipeline import PosePipeline
    from src.registry.model_registry import ModelRegistry
    from src.training.pose_geometry import resolve_symmetry_matrices

    registry = ModelRegistry(args.registry_root)
    loaded = registry.load(registry.resolve(args.sku), device=resolve_device(args.device))
    pipeline = PosePipeline.from_loaded_bundle(loaded)
    model_points = loaded.model_points_object.astype(np.float64)
    diameter = float(loaded.diameter_m)
    symmetry = resolve_symmetry_matrices(loaded.bundle.manifest.model.symmetry_name).astype(np.float64)
    add_thr = 0.1 * diameter * 1000.0

    samples = sorted(glob.glob(str(Path(args.processed) / "*.npz")))[: args.limit]
    rows, scenes = [], []
    for sample_path in samples:
        data = np.load(sample_path)
        points = data["points"].astype(np.float32)
        features = data["features"].astype(np.float32)
        gt_labels = data["instance_labels"].astype(np.int64)
        meta = load_raw_metadata(args.raw_root, str(data["raw_sample"]))
        if meta is None:
            continue
        visible = {int(v) for v in np.unique(gt_labels) if int(v) > 0}
        gt_list = [
            rigid_object_to_camera_from_scaled(np.asarray(i["object_to_camera"], dtype=np.float64))
            for i in meta["instances"] if int(i["instance_id"]) in visible
        ]
        if not gt_list:
            continue

        result = pipeline.infer_from_points(points, features)
        scene_instances = []
        for inst in result.instances:
            add_mm = _best_add_mm(model_points, inst.object_to_camera, gt_list, symmetry)
            entry = {
                "add_mm": add_mm, "correct": bool(add_mm < add_thr),
                "confidence": float(inst.confidence or 0.0),
                "model_fit": float(inst.diagnostics.get("model_fit", 9.9)),
                "point_count": int(inst.point_count),
            }
            rows.append(entry)
            scene_instances.append(entry)
        if not scene_instances:
            continue
        t1_conf, t3_conf = _topk(scene_instances, "confidence", False, 3)
        t1_fit, t3_fit = _topk(scene_instances, "model_fit", True, 3)
        scenes.append({
            "sample": Path(sample_path).stem, "n_instances": len(scene_instances),
            "n_correct": sum(e["correct"] for e in scene_instances),
            "top1_conf": t1_conf, "top3_conf": t3_conf, "top1_fit": t1_fit, "top3_fit": t3_fit,
        })

    add = np.array([r["add_mm"] for r in rows])
    correct = np.array([r["correct"] for r in rows])
    fit = np.array([r["model_fit"] for r in rows])
    gate_rows = {}
    for g in (0.05, 0.08, 0.10, 0.15):
        keep = fit < g
        gate_rows[f"fit<{g}"] = {
            "kept": int(keep.sum()),
            "precision": round(float(correct[keep].mean()), 3) if keep.any() else None,
            "recall_of_correct": round(float(correct[keep].sum() / max(correct.sum(), 1)), 3),
        }
    report = {
        "sku": args.sku, "n_scenes": len(scenes), "n_instances": len(rows),
        "add_thr_mm": round(add_thr, 2),
        "raw_precision_all_instances": round(float(correct.mean()), 3) if rows else None,
        "pick_success": {
            "top1_by_confidence": round(float(np.mean([s["top1_conf"] for s in scenes])), 3) if scenes else None,
            "top3_by_confidence": round(float(np.mean([s["top3_conf"] for s in scenes])), 3) if scenes else None,
            "top1_by_fit": round(float(np.mean([s["top1_fit"] for s in scenes])), 3) if scenes else None,
            "top3_by_fit": round(float(np.mean([s["top3_fit"] for s in scenes])), 3) if scenes else None,
        },
        "fit_gate": gate_rows,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps({"summary": report, "scenes": scenes}, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
