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
  import-pcd        turn organized camera .pcd frames (e.g. Basler stereo) into
                    this format: depth.npy in metres positive-forward + fitted
                    pinhole intrinsics (+ optional estimated normal.npy).
  evaluate          run the registry pipeline on every frame, optionally ICP-
                    refine, and (when labels exist) report symmetry-aware metrics.
  eval-seg          run instance segmentation only (model or classical) on every
                    frame and report segmentation-level stats (foreground
                    fraction, instance count/sizes; IoU when seg GT exists).
  label             ICP-refine operator rough inits into GT labels with a quality
                    score (semi-automatic labeling for real captures).

    python scripts/prepare_real_eval_set.py export-synthetic --raw synthetic-data/bending_pipe_4 \
        --out real-data --set bending_synth --sku bending_pipe --limit 5
    python scripts/prepare_real_eval_set.py import-pcd --pcd real-data/bending \
        --out real-data --set bending_real --save-normal
    python scripts/prepare_real_eval_set.py evaluate --set real-data/bending_synth --sku bending_pipe --refine
    python scripts/prepare_real_eval_set.py eval-seg --set real-data/bending_real --sku bending_pipe --crop auto
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


# ------------------------------------------------------------------- import-pcd
def _fit_pipeline_intrinsics(points: np.ndarray, pixels: np.ndarray) -> dict:
    """Fit fx, fy, cx, cy so `unproject_depth_to_camera` reproduces `points`.

    The pipeline convention is x = (u - cx) / fx * z and y = -(v - cy) / fy * z,
    so we solve the two independent least-squares systems
        u = fx * (x / z) + cx        v = -fy * (y / z) + cy
    over every valid pixel. Works whatever the source camera's axis conventions
    were, because the fit happens after conversion to positive-forward metres.
    """

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    u, v = pixels[:, 0].astype(np.float64), pixels[:, 1].astype(np.float64)
    ax = np.stack([x / z, np.ones_like(z)], axis=1)
    fx, cx = np.linalg.lstsq(ax, u, rcond=None)[0]
    ay = np.stack([-y / z, np.ones_like(z)], axis=1)
    fy, cy = np.linalg.lstsq(ay, v, rcond=None)[0]
    residual_u = float(np.sqrt(np.mean((ax @ np.array([fx, cx]) - u) ** 2)))
    residual_v = float(np.sqrt(np.mean((ay @ np.array([fy, cy]) - v) ** 2)))
    return {
        "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
        "fit_rmse_u_px": round(residual_u, 4), "fit_rmse_v_px": round(residual_v, 4),
    }


def cmd_import_pcd(args: argparse.Namespace) -> int:
    from src.data.pcd_loading import convert_to_positive_forward, estimate_camera_normals, read_pcd

    pcd_dir = Path(args.pcd)
    pcd_paths = sorted(pcd_dir.glob(args.pattern))
    if args.limit:
        pcd_paths = pcd_paths[: args.limit]
    if not pcd_paths:
        print(f"no .pcd files matching '{args.pattern}' under {pcd_dir}")
        return 1

    out_set = Path(args.out) / args.set
    written = 0
    for pcd_path in pcd_paths:
        cloud = read_pcd(pcd_path)
        if not cloud.organized:
            print(f"  skip {pcd_path.name}: not an organized cloud (need one point per pixel)")
            continue
        valid = cloud.valid_mask()
        points, conversions = convert_to_positive_forward(cloud.xyz[valid])
        pixels = cloud.pixel_grid()[valid]

        intrinsics = _fit_pipeline_intrinsics(points, pixels)
        intrinsics.update({"width": cloud.width, "height": cloud.height})

        depth = np.zeros((cloud.height, cloud.width), dtype=np.float32)
        depth[pixels[:, 1], pixels[:, 0]] = points[:, 2].astype(np.float32)

        frame_dir = out_set / pcd_path.stem
        frame_dir.mkdir(parents=True, exist_ok=True)
        np.save(frame_dir / "depth.npy", depth)
        _write_json(frame_dir / "intrinsics.json", {**intrinsics, **conversions, "source_pcd": str(pcd_path)})
        if args.save_normal:
            normals = estimate_camera_normals(points)
            normal_map = np.zeros((cloud.height, cloud.width, 3), dtype=np.float32)
            normal_map[pixels[:, 1], pixels[:, 0]] = normals
            np.save(frame_dir / "normal.npy", normal_map)
        if "intensity" in cloud.fields:
            try:
                from PIL import Image

                intensity = np.zeros(cloud.width * cloud.height, dtype=np.float64)
                intensity[valid.nonzero()[0]] = cloud.fields["intensity"][valid].astype(np.float64)
                grid = intensity.reshape(cloud.height, cloud.width)
                peak = np.percentile(grid[grid > 0], 99) if (grid > 0).any() else 1.0
                preview = np.clip(grid / max(peak, 1e-9) * 255.0, 0, 255).astype(np.uint8)
                Image.fromarray(preview).save(frame_dir / "preview.png")
            except Exception:  # noqa: BLE001 - preview is best-effort
                pass
        written += 1
        print(
            f"  {frame_dir.relative_to(Path(args.out))}: {int(valid.sum())} valid px "
            f"({100.0 * valid.sum() / valid.size:.0f}%), fit rmse u/v = "
            f"{intrinsics['fit_rmse_u_px']}/{intrinsics['fit_rmse_v_px']} px"
            f"{' [mm->m]' if conversions['in_mm'] else ''}{' [flip-z]' if conversions['flip_z'] else ''}"
        )
    print(f"imported {written} frames to {out_set}")
    return 0


# --------------------------------------------------------------------- eval-seg
def cmd_eval_seg(args: argparse.Namespace) -> int:
    from src.inference.workspace_crop import WorkspaceCropConfig, crop_workspace

    set_dir = Path(args.set)
    frames = _frames(set_dir)
    if not frames:
        print(f"no frames with depth.npy under {set_dir}")
        return 1

    segmenter = None
    if args.method == "model":
        from src.inference.device import resolve_device
        from src.inference.instance_segmentation import InstanceSegmenter
        from src.registry.model_registry import ModelRegistry

        if args.checkpoint:
            segmenter = InstanceSegmenter.from_checkpoint(args.checkpoint, device=args.device)
            model_id = f"checkpoint:{Path(args.checkpoint).name}"
        else:
            registry = ModelRegistry(args.registry_root)
            bundle = registry.resolve(args.sku)
            segmenter = InstanceSegmenter.from_loaded_bundle(registry.load(bundle, device=resolve_device(args.device)))
            model_id = f"{args.sku}:{bundle.version}"
    else:
        model_id = "classical"

    per_frame = []
    for frame in frames:
        from src.data.depth_unprojection import unproject_depth_to_camera

        depth = np.load(frame / "depth.npy").astype(np.float32)
        intrinsics = _read_json(frame / "intrinsics.json")
        points, pixels = unproject_depth_to_camera(depth, intrinsics)
        features = None
        if (frame / "normal.npy").exists():
            normal = np.asarray(np.load(frame / "normal.npy"), dtype=np.float32)
            features = normal[pixels[:, 1].astype(np.int64), pixels[:, 0].astype(np.int64)]

        pixels = pixels.astype(np.int64)
        if args.crop == "auto":
            keep, _info = crop_workspace(points, WorkspaceCropConfig(mode="auto_plane"))
            points, pixels = points[keep], pixels[keep]
            features = None if features is None else features[keep]

        if args.method == "classical":
            from src.inference.classical_segmentation import segment_scene_classical

            result = segment_scene_classical(points, None, pixels=pixels)
        else:
            result = segmenter.segment_points(points, features, pixels=pixels)

        labels = result.instance_labels
        foreground = float((labels > 0).mean()) if labels.size else 0.0
        sizes = sorted((inst.point_count for inst in result.instances), reverse=True)
        report = {
            "frame": frame.name,
            "points_in": int(points.shape[0]),
            "points_segmented": int(labels.shape[0]),
            "foreground_fraction": round(foreground, 4),
            "instance_count": result.instance_count,
            "instance_sizes": sizes[:20],
        }

        gt_path = frame / "seg_labels.npy"
        if gt_path.exists() and result.pixels is not None:
            from src.inference.instance_clustering import evaluate_instance_predictions

            gt_grid = np.asarray(np.load(gt_path))
            gt = gt_grid[result.pixels[:, 1], result.pixels[:, 0]].astype(np.int64)
            semantic_pred = labels > 0
            semantic_gt = gt > 0
            intersection = int((semantic_pred & semantic_gt).sum())
            union = int((semantic_pred | semantic_gt).sum())
            instance_metrics = evaluate_instance_predictions(gt, labels)
            report.update(
                {
                    "semantic_iou": round(intersection / union, 4) if union else None,
                    "instance_precision": round(instance_metrics["instance_precision"], 4),
                    "instance_recall": round(instance_metrics["instance_recall"], 4),
                    "instance_mean_iou": round(instance_metrics["instance_mean_iou"], 4),
                }
            )
        per_frame.append(report)
        print(
            f"  {frame.name}: fg {report['foreground_fraction']:.2%}, "
            f"{report['instance_count']} instances, sizes {report['instance_sizes'][:6]}"
        )

    summary = {
        "set": str(set_dir),
        "method": model_id,
        "crop": args.crop,
        "n_frames": len(per_frame),
        "mean_foreground_fraction": round(float(np.mean([r["foreground_fraction"] for r in per_frame])), 4),
        "mean_instance_count": round(float(np.mean([r["instance_count"] for r in per_frame])), 2),
        "per_frame": per_frame,
    }
    scored = [r for r in per_frame if "semantic_iou" in r]
    if scored:
        summary["mean_semantic_iou"] = round(float(np.mean([r["semantic_iou"] or 0.0 for r in scored])), 4)
        summary["mean_instance_recall"] = round(float(np.mean([r["instance_recall"] for r in scored])), 4)
        summary["mean_instance_precision"] = round(float(np.mean([r["instance_precision"] for r in scored])), 4)
    report_path = set_dir / f"seg_report_{args.method}.json"
    _write_json(report_path, summary)
    print(f"wrote {report_path}")
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


def _pick_success(predictions, labels, model_points, symmetry, diameter, k=3):
    """Top-1/top-k pick success: ranked by confidence, is a pick a graspable pose?

    A pose is graspable if its symmetry-aware ADD to the nearest GT object is below
    0.1 diameter. Predictions arrive already confidence-ranked from the pipeline.
    """
    thr = 0.1 * diameter * 1000.0
    gts = [np.asarray(g["object_to_camera"], dtype=np.float64) for g in labels]
    if not gts:
        return False, False
    flags = []
    for p in predictions:
        P = np.asarray(p["object_to_camera"], dtype=np.float64)
        transformed = model_points @ P[:3, :3].T + P[:3, 3]
        best = min(
            float(np.mean(np.linalg.norm(transformed - (model_points @ (g[:3, :3] @ S).T + g[:3, 3]), axis=1)))
            for g in gts for S in symmetry
        ) * 1000.0
        flags.append(best < thr)
    return (bool(flags and flags[0]), any(flags[:k]))


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
    options = InferenceOptions(
        max_instances=args.max_instances, min_confidence=args.min_confidence, max_model_fit=args.max_model_fit
    )

    all_add, all_trans, total_gt, total_matched, per_frame = [], [], 0, 0, []
    pick_top1, pick_top3 = [], []
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
            # Operational bin-picking metric: rank picks by confidence, is the top
            # (or one of top-3) a graspable pose (ADD < 0.1d to some object)?
            t1, t3 = _pick_success(predictions, labels, model_points, symmetry, diameter)
            pick_top1.append(t1)
            pick_top3.append(t3)
            frame_report.update({"n_gt": len(rows), "n_matched": len(matched), "top1_pick": t1, "top3_pick": t3})
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
            "top1_pick_success": round(float(np.mean(pick_top1)), 4) if pick_top1 else None,
            "top3_pick_success": round(float(np.mean(pick_top3)), 4) if pick_top3 else None,
        }
        print(f"  ADD {report['metrics']['add_mm_mean']} mm | ADD_0.1d {report['metrics']['add_0.1d_success']} | "
              f"recall {report['metrics']['recall']} | top1_pick {report['metrics']['top1_pick_success']} | "
              f"top3_pick {report['metrics']['top3_pick_success']}"
              f"{' [refined]' if args.refine else ''}{' [fit-gated]' if args.max_model_fit else ''}")
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

    p_pcd = sub.add_parser("import-pcd", help="convert organized camera .pcd frames to the real-eval format")
    p_pcd.add_argument("--pcd", required=True, help="Folder containing organized .pcd files.")
    p_pcd.add_argument("--pattern", default="*.pcd", help="Glob pattern for the .pcd files.")
    p_pcd.add_argument("--out", default="real-data")
    p_pcd.add_argument("--set", required=True)
    p_pcd.add_argument("--limit", type=int, default=0)
    p_pcd.add_argument("--save-normal", action="store_true", help="Estimate + save normal.npy per frame.")
    p_pcd.set_defaults(func=cmd_import_pcd)

    p_seg = sub.add_parser("eval-seg", help="segmentation-only metrics on a real-eval set")
    p_seg.add_argument("--set", required=True)
    p_seg.add_argument("--method", choices=("model", "classical"), default="model")
    p_seg.add_argument("--sku", default=None, help="Registry SKU (with --method model).")
    p_seg.add_argument("--checkpoint", default=None, help="Instance-seg checkpoint (alternative to --sku).")
    p_seg.add_argument("--crop", choices=("none", "auto"), default="none", help="Workspace crop before segmentation.")
    p_seg.set_defaults(func=cmd_eval_seg)

    p_eval = sub.add_parser("evaluate", help="run the pipeline on a real-eval set and score against labels")
    p_eval.add_argument("--set", required=True)
    p_eval.add_argument("--sku", required=True)
    p_eval.add_argument("--refine", action="store_true", help="ICP-refine each prediction against its crop")
    p_eval.add_argument("--max-instances", type=int, default=50)
    p_eval.add_argument("--min-confidence", type=float, default=0.0)
    p_eval.add_argument("--max-model-fit", type=float, default=None, help="Reject poses whose model->crop chamfer/diameter exceeds this.")
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
