"""Real-data evaluation harness for the pose backend (P8).

A real-eval set is a folder of frames in a camera-agnostic format:

    <set>/<frame>/depth.npy        # (H, W) float32 depth in metres, 0 = invalid
    <set>/<frame>/intrinsics.json  # {fx, fy, cx, cy, width, height}
    <set>/<frame>/normal.npy       # optional (H, W, 3) camera-frame normals
    <set>/<frame>/labels.json      # optional GT: {sku, instances:[{instance_id, object_to_camera, quality}]}
    <set>/<frame>/init_labels.json # optional operator rough inits (for `label`)

Subcommands:
  export-synthetic  turn raw synthetic samples into this format (validate the
                    harness before a real camera exists; labels = visible GT).
  evaluate          run the registry pipeline on every frame, optionally ICP-
                    refine, and (when labels exist) report symmetry-aware metrics.
  label             ICP-refine operator rough inits into GT labels with a quality
                    score (semi-automatic labeling for real captures).

    python scripts/prepare_real_eval_set.py export-synthetic --raw synthetic-data/bending_pipe_4 \
        --out real-data --set bending_synth --sku bending_pipe --limit 5
    python scripts/prepare_real_eval_set.py evaluate --set real-data/bending_synth --sku bending_pipe --refine
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------- io
def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _frames(set_dir: Path) -> list[Path]:
    return sorted(p for p in set_dir.iterdir() if p.is_dir() and (p / "depth.npy").exists())


# --------------------------------------------------------------- geometry utils
def _symmetry_aware_add_mm(model_points: np.ndarray, pred: np.ndarray, gt: np.ndarray, symmetry: np.ndarray) -> float:
    Rp, tp = pred[:3, :3], pred[:3, 3]
    Rg, tg = gt[:3, :3], gt[:3, 3]
    transformed_pred = model_points @ Rp.T + tp
    best = min(
        float(np.mean(np.linalg.norm(transformed_pred - (model_points @ (Rg @ S).T + tg), axis=1)))
        for S in symmetry
    )
    return best * 1000.0


def _scene_points_metadata_frame(depth: np.ndarray, intrinsics: dict) -> np.ndarray:
    from src.data.depth_unprojection import unproject_depth_to_camera
    from src.data.pose_crop_dataset import positive_forward_depth_to_metadata_camera

    points_pf, _pixels = unproject_depth_to_camera(depth, intrinsics)
    return positive_forward_depth_to_metadata_camera(points_pf)


def _crop_around(points: np.ndarray, center: np.ndarray, radius: float, min_points: int = 64) -> np.ndarray:
    distances = np.linalg.norm(points - center.reshape(1, 3), axis=1)
    mask = distances <= radius
    if int(mask.sum()) < min_points:
        order = np.argsort(distances)[: max(min_points, 1)]
        return points[order]
    return points[mask]


def _icp_refine(pose: np.ndarray, crop_points: np.ndarray, model_points: np.ndarray, diameter: float, device: str):
    import torch

    from src.inference.pose_refinement import PoseRefinementConfig, refine_pose_batch

    rotation = torch.tensor(pose[:3, :3], dtype=torch.float32, device=device).unsqueeze(0)
    origin = torch.tensor(pose[:3, 3], dtype=torch.float32, device=device).unsqueeze(0)
    batch = {
        "sampled_points_camera": torch.tensor(crop_points, dtype=torch.float32, device=device).unsqueeze(0),
        "model_points_object": torch.tensor(model_points, dtype=torch.float32, device=device),
        "model_diameter_m": torch.tensor([diameter], dtype=torch.float32, device=device),
    }
    out = refine_pose_batch(
        rotation_object_to_camera=rotation,
        origin_camera=origin,
        batch=batch,
        config=PoseRefinementConfig(method="hybrid"),
    )
    refined = np.eye(4, dtype=np.float64)
    refined[:3, :3] = out["rotation_object_to_camera"][0].detach().cpu().numpy()
    refined[:3, 3] = out["origin_camera"][0].detach().cpu().numpy()
    return refined, float(out["alignment_after_m"][0].detach().cpu())


def _load_pipeline(registry_root: str, sku: str, device: str):
    from src.inference.device import resolve_device
    from src.inference.pose_pipeline import PosePipeline
    from src.registry.model_registry import ModelRegistry

    registry = ModelRegistry(registry_root)
    loaded = registry.load(registry.resolve(sku), device=resolve_device(device))
    return PosePipeline.from_loaded_bundle(loaded), loaded


# ------------------------------------------------------------ export-synthetic
def cmd_export_synthetic(args: argparse.Namespace) -> int:
    from src.data.pose_crop_dataset import rigid_object_to_camera_from_scaled

    raw_root = Path(args.raw)
    sample_dirs = [raw_root] if (raw_root / "metadata.json").exists() else sorted(
        p for p in raw_root.iterdir() if p.is_dir() and (p / "metadata.json").exists()
    )
    if args.limit:
        sample_dirs = sample_dirs[: args.limit]
    if not sample_dirs:
        print(f"no samples with metadata.json under {raw_root}")
        return 1
    out_set = Path(args.out) / args.set
    written = 0
    for sample_dir in sample_dirs:
        metadata = _read_json(sample_dir / "metadata.json")
        sensor = np.load(sample_dir / "sensor_data.npz")
        depth = np.asarray(sensor["depth_m"], dtype=np.float32)
        intrinsics = {k: metadata["camera_intrinsics"][k] for k in ("fx", "fy", "cx", "cy", "width", "height")}
        visible = set(int(v) for v in np.unique(sensor["point_instance_ids"]) if int(v) > 0) if "point_instance_ids" in sensor.files else None
        instances = []
        for inst in metadata.get("instances", []):
            iid = int(inst["instance_id"])
            if visible is not None and iid not in visible:
                continue
            # Metadata object_to_camera bakes model_scale into the rotation columns;
            # rigidify to a unit-rotation metres-space pose so it matches pipeline output.
            rigid = rigid_object_to_camera_from_scaled(np.asarray(inst["object_to_camera"], dtype=np.float64))
            instances.append(
                {"instance_id": iid, "object_to_camera": rigid.tolist(),
                 "class_name": metadata.get("class_name", args.sku), "quality": 1.0}
            )
        frame_dir = out_set / sample_dir.name
        frame_dir.mkdir(parents=True, exist_ok=True)
        np.save(frame_dir / "depth.npy", depth)
        _write_json(frame_dir / "intrinsics.json", intrinsics)
        _write_json(frame_dir / "labels.json", {"sku": args.sku, "source": "synthetic_gt", "instances": instances})
        if args.save_normal and "normal_camera" in sensor.files:
            np.save(frame_dir / "normal.npy", np.asarray(sensor["normal_camera"], dtype=np.float32))
        written += 1
        print(f"  {frame_dir.relative_to(Path(args.out))}: {len(instances)} visible instances")
    print(f"exported {written} frames to {out_set}")
    return 0


# --------------------------------------------------------------------- evaluate
def _match_and_score(predictions, labels, model_points, symmetry, diameter, match_threshold_fraction=0.3):
    """Optimal (Hungarian) GT<->prediction assignment by centroid distance, tightly gated.

    Greedy matching mismatches predictions to neighbouring GT in dense piles (where
    parts are closer together than they are long), which inflates ADD/translation.
    """

    gt = [(int(g["instance_id"]), np.asarray(g["object_to_camera"], dtype=np.float64)) for g in labels]
    pred = [(int(p["instance_id"]), np.asarray(p["object_to_camera"], dtype=np.float64)) for p in predictions]
    rows = [{"instance_id": gid, "matched": False} for gid, _ in gt]
    gate = match_threshold_fraction * diameter
    if gt and pred:
        from scipy.optimize import linear_sum_assignment

        cost = np.zeros((len(gt), len(pred)), dtype=np.float64)
        for gi, (_gid, gpose) in enumerate(gt):
            for pi, (_pid, ppose) in enumerate(pred):
                cost[gi, pi] = float(np.linalg.norm(ppose[:3, 3] - gpose[:3, 3]))
        gi_idx, pi_idx = linear_sum_assignment(cost)
        for gi, pi in zip(gi_idx, pi_idx):
            if cost[gi, pi] > gate:
                continue
            gpose, ppose = gt[gi][1], pred[pi][1]
            rows[gi] = {
                "instance_id": gt[gi][0], "matched": True,
                "add_mm": _symmetry_aware_add_mm(model_points, ppose, gpose, symmetry),
                "translation_mm": float(cost[gi, pi]) * 1000.0,
            }
    return rows


def cmd_evaluate(args: argparse.Namespace) -> int:
    from src.service.schemas import InferenceOptions
    from src.training.pose_geometry import resolve_symmetry_matrices

    set_dir = Path(args.set)
    frames = _frames(set_dir)
    if not frames:
        print(f"no frames with depth.npy under {set_dir}")
        return 1
    pipeline, loaded = _load_pipeline(args.registry_root, args.sku, args.device)
    model_points = loaded.model_points_object.astype(np.float64)
    diameter = float(loaded.diameter_m)
    symmetry = resolve_symmetry_matrices(loaded.bundle.manifest.model.symmetry_name).astype(np.float64)
    options = InferenceOptions(max_instances=args.max_instances, min_confidence=args.min_confidence)

    all_add, all_trans, total_gt, total_matched, per_frame = [], [], 0, 0, []
    for frame in frames:
        depth = np.load(frame / "depth.npy").astype(np.float32)
        intrinsics = _read_json(frame / "intrinsics.json")
        normal = np.load(frame / "normal.npy") if (frame / "normal.npy").exists() else None
        result = pipeline.infer_scene(depth, intrinsics, normal, options=options)
        predictions = [
            {"instance_id": int(inst.instance_id), "object_to_camera": inst.object_to_camera.tolist(),
             "confidence": float(inst.confidence or 0.0)}
            for inst in result.instances
        ]
        if args.refine and result.instances:
            scene_meta = _scene_points_metadata_frame(depth, intrinsics)
            for pred, inst in zip(predictions, result.instances):
                crop = _crop_around(scene_meta, inst.crop_centroid_camera, args.crop_radius_fraction * diameter)
                refined, _align = _icp_refine(np.asarray(pred["object_to_camera"]), crop, loaded.model_points_object, diameter, loaded.device)
                pred["object_to_camera"] = refined.tolist()

        labels_path = frame / "labels.json"
        frame_report = {"frame": frame.name, "n_predictions": len(predictions)}
        if labels_path.exists():
            labels = _read_json(labels_path)["instances"]
            rows = _match_and_score(predictions, labels, model_points, symmetry, diameter)
            matched = [r for r in rows if r["matched"]]
            total_gt += len(rows)
            total_matched += len(matched)
            all_add.extend(r["add_mm"] for r in matched)
            all_trans.extend(r["translation_mm"] for r in matched)
            frame_report.update({"n_gt": len(rows), "n_matched": len(matched)})
        per_frame.append(frame_report)
        _write_json(frame / "predictions.json", {"sku": args.sku, "model_version": loaded.bundle.version, "instances": predictions, "timings_ms": result.timings_ms})

    report = {"sku": args.sku, "model_version": loaded.bundle.version, "n_frames": len(frames),
              "refined": bool(args.refine), "per_frame": per_frame}
    if all_add:
        add = np.asarray(all_add)
        thr = 0.1 * diameter * 1000.0
        report["metrics"] = {
            "n_matched": int(total_matched), "n_gt": int(total_gt),
            "recall": round(total_matched / max(total_gt, 1), 4),
            "add_mm_mean": round(float(add.mean()), 3), "add_mm_median": round(float(np.median(add)), 3),
            "translation_mm_mean": round(float(np.mean(all_trans)), 3),
            "add_0.1d_success": round(float(np.mean(add < thr)), 4),
        }
        print(f"  ADD {report['metrics']['add_mm_mean']} mm | trans {report['metrics']['translation_mm_mean']} mm | "
              f"ADD_0.1d {report['metrics']['add_0.1d_success']} | recall {report['metrics']['recall']} "
              f"(matched {total_matched}/{total_gt}){' [refined]' if args.refine else ''}")
    else:
        print(f"  ran {len(frames)} frames, no GT labels -> predictions only")
    _write_json(set_dir / "eval_report.json", report)
    print(f"wrote {set_dir / 'eval_report.json'}")
    return 0


# ------------------------------------------------------------------------ label
def cmd_label(args: argparse.Namespace) -> int:
    set_dir = Path(args.set)
    frames = [f for f in _frames(set_dir) if (f / "init_labels.json").exists()]
    if not frames:
        print(f"no frames with init_labels.json under {set_dir}")
        return 1
    _pipeline, loaded = _load_pipeline(args.registry_root, args.sku, args.device)
    diameter = float(loaded.diameter_m)
    for frame in frames:
        depth = np.load(frame / "depth.npy").astype(np.float32)
        intrinsics = _read_json(frame / "intrinsics.json")
        scene_meta = _scene_points_metadata_frame(depth, intrinsics)
        inits = _read_json(frame / "init_labels.json")["instances"]
        labeled = []
        for inst in inits:
            pose = np.asarray(inst["object_to_camera"], dtype=np.float64)
            crop = _crop_around(scene_meta, pose[:3, 3], args.crop_radius_fraction * diameter)
            refined, align = _icp_refine(pose, crop, loaded.model_points_object, diameter, loaded.device)
            quality = round(float(1.0 / (1.0 + align / max(diameter, 1e-6))), 4)  # 1 = perfect alignment
            labeled.append({"instance_id": int(inst["instance_id"]), "object_to_camera": refined.tolist(),
                            "alignment_m": round(align, 5), "quality": quality})
        _write_json(frame / "labels.json", {"sku": args.sku, "source": "icp_refined_init", "instances": labeled})
        mean_q = np.mean([r["quality"] for r in labeled]) if labeled else 0.0
        print(f"  {frame.name}: labeled {len(labeled)} instances, mean quality {mean_q:.3f}")
    print(f"labeled {len(frames)} frames")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Real-data pose evaluation harness")
    parser.add_argument("--registry-root", default="models")
    parser.add_argument("--device", default="cuda")
    sub = parser.add_subparsers(dest="command", required=True)

    p_exp = sub.add_parser("export-synthetic", help="export raw synthetic samples to the real-eval format")
    p_exp.add_argument("--raw", required=True)
    p_exp.add_argument("--out", default="real-data")
    p_exp.add_argument("--set", required=True)
    p_exp.add_argument("--sku", required=True)
    p_exp.add_argument("--limit", type=int, default=0)
    p_exp.add_argument("--save-normal", action="store_true")
    p_exp.set_defaults(func=cmd_export_synthetic)

    p_eval = sub.add_parser("evaluate", help="run the pipeline on a real-eval set and score against labels")
    p_eval.add_argument("--set", required=True)
    p_eval.add_argument("--sku", required=True)
    p_eval.add_argument("--refine", action="store_true", help="ICP-refine each prediction against its crop")
    p_eval.add_argument("--max-instances", type=int, default=50)
    p_eval.add_argument("--min-confidence", type=float, default=0.0)
    p_eval.add_argument("--crop-radius-fraction", type=float, default=0.75)
    p_eval.set_defaults(func=cmd_evaluate)

    p_lab = sub.add_parser("label", help="ICP-refine operator rough inits into GT labels")
    p_lab.add_argument("--set", required=True)
    p_lab.add_argument("--sku", required=True)
    p_lab.add_argument("--crop-radius-fraction", type=float, default=0.75)
    p_lab.set_defaults(func=cmd_label)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
