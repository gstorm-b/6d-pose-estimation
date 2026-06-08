"""Sweep predicted-crop export parameters and evaluate downstream pose quality."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_float_list(value: str) -> list[float]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("expected at least one float value")
    return [float(item) for item in items]


def parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("expected at least one integer value")
    return [int(item) for item in items]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-checkpoint", required=True)
    parser.add_argument("--instance-data", required=True, help="Processed PointNet++ instance dataset root.")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--pose-config", required=True)
    parser.add_argument("--pose-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--work-dir", help="Defaults to <out stem>_crops next to --out.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ops-backend", default="pytorch3d")
    parser.add_argument("--object-probability-thresholds", default="0.45,0.50,0.55,0.60")
    parser.add_argument("--dbscan-eps-m", default="0.003,0.004,0.005,0.006,0.007")
    parser.add_argument("--min-cluster-points", default="64,96,128,160")
    parser.add_argument("--dbscan-min-samples", type=int)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--min-matched-gt-iou", type=float, default=0.5)
    parser.add_argument("--min-retained-matched-fraction", type=float, default=0.90)
    parser.add_argument("--refine-pose", action="store_true")
    parser.add_argument("--refinement-method", default="hybrid")
    parser.add_argument("--refinement-distance-threshold-fraction", type=float, default=0.16)
    parser.add_argument("--refinement-max-translation-delta-fraction", type=float, default=0.20)
    parser.add_argument("--refinement-max-rotation-delta-deg", type=float, default=20.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def safe_float(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def run_command(command: list[str], *, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"returncode": 0, "command": command, "stdout": "", "stderr": "", "dry_run": True}
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
    return {
        "returncode": int(completed.returncode),
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "dry_run": False,
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_block(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_count": payload.get("sample_count"),
        "raw_metrics": payload.get("raw_metrics", payload.get("metrics")),
        "refined_metrics": payload.get("refined_metrics"),
        "refinement_summary": (payload.get("refinement") or {}).get("summary"),
    }


def score_metrics(metrics: dict[str, Any] | None) -> tuple[float, float, float]:
    if not metrics:
        return (0.0, float("inf"), float("inf"))
    return (
        float(metrics.get("pose_success_add_0.1d", 0.0)),
        -float(metrics.get("add_mm", float("inf"))),
        -float(metrics.get("translation_error_mm", float("inf"))),
    )


def choose_recommended(rows: list[dict[str, Any]], *, min_retained_matched_fraction: float) -> dict[str, Any] | None:
    successful = [row for row in rows if row.get("status") == "ok"]
    if not successful:
        return None
    baseline_matched = max(float(row["crop_summary"].get("matched_gt_iou_at_least_0.5", 0.0)) for row in successful)
    min_matched = baseline_matched * float(min_retained_matched_fraction)

    def row_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
        all_metrics = row["pose_all"].get("refined_metrics") or row["pose_all"].get("raw_metrics")
        matched_metrics = row["pose_matched"].get("refined_metrics") or row["pose_matched"].get("raw_metrics")
        all_success, all_add_neg, all_trans_neg = score_metrics(all_metrics)
        matched_success, _matched_add_neg, _matched_trans_neg = score_metrics(matched_metrics)
        crop_summary = row["crop_summary"]
        return (
            all_success,
            all_add_neg,
            matched_success,
            float(crop_summary.get("matched_gt_iou_mean", 0.0)),
            all_trans_neg,
        )

    candidates = [
        row
        for row in successful
        if float(row["crop_summary"].get("matched_gt_iou_at_least_0.5", 0.0)) >= min_matched
    ]
    if not candidates:
        candidates = successful
    return sorted(candidates, key=row_key, reverse=True)[0]


def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    work_dir = Path(args.work_dir) if args.work_dir else out_path.with_suffix("").parent / f"{out_path.with_suffix('').name}_crops"
    work_dir.mkdir(parents=True, exist_ok=True)
    probabilities = parse_float_list(args.object_probability_thresholds)
    eps_values = parse_float_list(args.dbscan_eps_m)
    min_cluster_values = parse_int_list(args.min_cluster_points)
    rows: list[dict[str, Any]] = []
    export_script = PROJECT_ROOT / "scripts" / "export_pose_instance_crops.py"
    eval_script = PROJECT_ROOT / "scripts" / "eval_pointnet2_pose.py"

    for probability in probabilities:
        for eps in eps_values:
            for min_cluster_points in min_cluster_values:
                name = f"prob_{safe_float(probability)}_eps_{safe_float(eps)}_min_{min_cluster_points}"
                crop_dir = work_dir / name
                pose_all_path = work_dir / f"{name}_pose_all.json"
                pose_matched_path = work_dir / f"{name}_pose_iou{safe_float(args.min_matched_gt_iou)}.json"
                row: dict[str, Any] = {
                    "name": name,
                    "params": {
                        "object_probability_threshold": probability,
                        "dbscan_eps_m": eps,
                        "min_cluster_points": min_cluster_points,
                        "dbscan_min_samples": args.dbscan_min_samples,
                    },
                    "crop_dir": str(crop_dir),
                    "status": "ok",
                }
                export_cmd = [
                    sys.executable,
                    str(export_script),
                    "--data",
                    args.instance_data,
                    "--split",
                    args.split,
                    "--source",
                    "predicted",
                    "--checkpoint",
                    args.instance_checkpoint,
                    "--raw-root",
                    args.raw_root,
                    "--out",
                    str(crop_dir),
                    "--device",
                    args.device,
                    "--ops-backend",
                    args.ops_backend,
                    "--object-probability-threshold",
                    str(probability),
                    "--dbscan-eps-m",
                    str(eps),
                    "--min-cluster-points",
                    str(min_cluster_points),
                ]
                if args.dbscan_min_samples is not None:
                    export_cmd.extend(["--dbscan-min-samples", str(args.dbscan_min_samples)])
                if args.limit_samples is not None:
                    export_cmd.extend(["--limit-samples", str(args.limit_samples)])

                if not args.skip_existing or not (crop_dir / "manifest.json").exists():
                    export_result = run_command(export_cmd, dry_run=args.dry_run)
                    row["export_command"] = export_result["command"]
                    row["export_returncode"] = export_result["returncode"]
                    if export_result["returncode"] != 0:
                        row["status"] = "export_failed"
                        row["stderr"] = export_result["stderr"][-4000:]
                        rows.append(row)
                        print(f"[pose-sweep] {name}: export failed")
                        continue

                if args.dry_run:
                    row["crop_summary"] = {}
                    rows.append(row)
                    print(f"[pose-sweep] {name}: dry run")
                    continue

                manifest = load_json(crop_dir / "manifest.json")
                matched_stats = manifest.get("matched_gt_iou_stats", {})
                row["crop_summary"] = {
                    "total_crops": manifest.get("total_crops", 0),
                    "crops_with_object_to_camera": manifest.get("crops_with_object_to_camera", 0),
                    "matched_gt_iou_mean": matched_stats.get("mean", 0.0),
                    "matched_gt_iou_at_least_0.5": matched_stats.get("at_least_0_5", 0.0),
                    "matched_gt_iou_at_least_0.75": matched_stats.get("at_least_0_75", 0.0),
                }

                base_eval_cmd = [
                    sys.executable,
                    str(eval_script),
                    "--config",
                    args.pose_config,
                    "--checkpoint",
                    args.pose_checkpoint,
                    "--data",
                    str(crop_dir),
                    "--device",
                    args.device,
                ]
                if args.refine_pose:
                    base_eval_cmd.extend(
                        [
                            "--refine-pose",
                            "--refinement-method",
                            args.refinement_method,
                            "--refinement-distance-threshold-fraction",
                            str(args.refinement_distance_threshold_fraction),
                            "--refinement-max-translation-delta-fraction",
                            str(args.refinement_max_translation_delta_fraction),
                            "--refinement-max-rotation-delta-deg",
                            str(args.refinement_max_rotation_delta_deg),
                        ]
                    )

                eval_all_cmd = base_eval_cmd + ["--out", str(pose_all_path)]
                all_result = run_command(eval_all_cmd, dry_run=False)
                row["pose_all_command"] = all_result["command"]
                row["pose_all_returncode"] = all_result["returncode"]
                if all_result["returncode"] != 0:
                    row["status"] = "pose_all_failed"
                    row["stderr"] = all_result["stderr"][-4000:]
                    rows.append(row)
                    print(f"[pose-sweep] {name}: all-pose eval failed")
                    continue
                row["pose_all"] = metric_block(load_json(pose_all_path))

                eval_matched_cmd = base_eval_cmd + [
                    "--out",
                    str(pose_matched_path),
                    "--min-matched-gt-iou",
                    str(args.min_matched_gt_iou),
                ]
                matched_result = run_command(eval_matched_cmd, dry_run=False)
                row["pose_matched_command"] = matched_result["command"]
                row["pose_matched_returncode"] = matched_result["returncode"]
                if matched_result["returncode"] != 0:
                    row["status"] = "pose_matched_failed"
                    row["stderr"] = matched_result["stderr"][-4000:]
                    rows.append(row)
                    print(f"[pose-sweep] {name}: matched-pose eval failed")
                    continue
                row["pose_matched"] = metric_block(load_json(pose_matched_path))
                rows.append(row)
                all_metrics = row["pose_all"].get("refined_metrics") or row["pose_all"].get("raw_metrics")
                print(
                    f"[pose-sweep] {name}: crops={row['crop_summary']['total_crops']} "
                    f"iou05={row['crop_summary']['matched_gt_iou_at_least_0.5']} "
                    f"all_add01d={all_metrics.get('pose_success_add_0.1d', 0.0):.3f}"
                )

    recommended = choose_recommended(rows, min_retained_matched_fraction=args.min_retained_matched_fraction)
    summary = {
        "instance_checkpoint": args.instance_checkpoint,
        "instance_data": args.instance_data,
        "raw_root": args.raw_root,
        "pose_config": args.pose_config,
        "pose_checkpoint": args.pose_checkpoint,
        "split": args.split,
        "device": args.device,
        "work_dir": str(work_dir),
        "min_matched_gt_iou": args.min_matched_gt_iou,
        "min_retained_matched_fraction": args.min_retained_matched_fraction,
        "refine_pose": bool(args.refine_pose),
        "recommended": recommended,
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[pose-sweep] wrote summary: {out_path}")
    if recommended is not None:
        print(f"[pose-sweep] recommended: {recommended['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
