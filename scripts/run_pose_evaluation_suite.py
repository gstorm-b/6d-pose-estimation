"""Run the reproducible pose evaluation suite for raw and optional refined pose."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


POSE_COMPILE_TARGETS = [
    "src/data/pose_crop_dataset.py",
    "src/models/pointnet2_pose.py",
    "src/models/pointnet2_pose_voting.py",
    "src/models/pointnet2_pose_refiner.py",
    "src/training/pose_losses.py",
    "src/training/pose_metrics.py",
    "src/training/pose_refiner_losses.py",
    "src/inference/pose_refinement.py",
    "scripts/train_pointnet2_pose.py",
    "scripts/train_pointnet2_pose_refiner.py",
    "scripts/eval_pointnet2_pose.py",
    "scripts/export_pose_instance_crops.py",
    "scripts/analyze_pose_errors.py",
    "scripts/sweep_pose_crop_export_params.py",
    "scripts/run_pose_evaluation_suite.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", help="Defaults to experiments/pose_eval_suite_<timestamp>.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--refine-pose", action="store_true")
    parser.add_argument("--refinement-method", default="hybrid")
    parser.add_argument("--refinement-distance-threshold-fraction", type=float, default=0.16)
    parser.add_argument("--refinement-max-translation-delta-fraction", type=float, default=0.20)
    parser.add_argument("--refinement-max-rotation-delta-deg", type=float, default=20.0)
    parser.add_argument("--k41144-translation-refiner-checkpoint")
    parser.add_argument("--bending-translation-refiner-checkpoint")

    parser.add_argument("--k41144-pose-config", default="configs/train/pointnet2_pose_voting_k41144.yaml")
    parser.add_argument("--k41144-pose-checkpoint", default="experiments/pointnet2_pose_k41144_20260608_023605/checkpoints/best_add.pt")
    parser.add_argument("--k41144-gt-val", default="experiments/phase14_pose_crops_k41144_gt_val_xyznormal")
    parser.add_argument("--k41144-gt-test", default="experiments/phase14_pose_crops_k41144_gt_test_xyznormal")
    parser.add_argument("--k41144-pred-test", default="experiments/phase17_pose_crops_k41144_pred_test_xyznormal")

    parser.add_argument("--bending-pose-config", default="configs/train/pointnet2_pose_voting_bending_pipe.yaml")
    parser.add_argument("--bending-pose-checkpoint", default="experiments/pointnet2_pose_bending_pipe_20260608_021309/checkpoints/best_add.pt")
    parser.add_argument("--bending-gt-val", default="experiments/phase14_pose_crops_bending_pipe_gt_val_xyznormal")
    parser.add_argument("--bending-gt-test", default="experiments/phase14_pose_crops_bending_pipe_gt_test_xyznormal")
    parser.add_argument("--bending-pred-test", default="experiments/phase17_pose_crops_bending_pipe_pred_test_xyznormal")
    return parser.parse_args()


def run_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
    return {
        "command": command,
        "returncode": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def run_checked(command: list[str]) -> dict[str, Any]:
    result = run_command(command)
    if result["returncode"] != 0:
        raise RuntimeError(
            "command failed:\n"
            + " ".join(command)
            + "\nSTDOUT:\n"
            + result["stdout"][-4000:]
            + "\nSTDERR:\n"
            + result["stderr"][-4000:]
        )
    return result


def eval_command(
    *,
    config: str,
    data: str,
    out: Path,
    checkpoint: str | None,
    device: str,
    identity_baseline: bool = False,
    min_matched_gt_iou: float | None = None,
    limit_samples: int | None = None,
    refine_args: list[str] | None = None,
    translation_refiner_checkpoint: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "eval_pointnet2_pose.py"),
        "--config",
        config,
        "--data",
        data,
        "--out",
        str(out),
        "--device",
        device,
    ]
    if checkpoint is not None:
        command.extend(["--checkpoint", checkpoint])
    if identity_baseline:
        command.append("--identity-baseline")
    if min_matched_gt_iou is not None:
        command.extend(["--min-matched-gt-iou", str(min_matched_gt_iou)])
    if limit_samples is not None:
        command.extend(["--limit-samples", str(limit_samples)])
    if refine_args:
        command.extend(refine_args)
    if translation_refiner_checkpoint:
        command.extend(["--translation-refiner-checkpoint", translation_refiner_checkpoint])
    return command


def load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def row_from_metrics(*, object_name: str, dataset_name: str, path: Path) -> dict[str, Any]:
    payload = load_metrics(path)
    raw = payload.get("raw_metrics", payload.get("metrics", {}))
    refined = payload.get("refined_metrics")
    row = {
        "object": object_name,
        "dataset": dataset_name,
        "path": str(path),
        "sample_count": payload.get("sample_count", 0),
        "matched_crop_iou": raw.get("matched_crop_iou"),
        "raw_add_mm": raw.get("add_mm"),
        "raw_translation_mm": raw.get("translation_error_mm"),
        "raw_add_0.1d": raw.get("pose_success_add_0.1d"),
        "refined_add_mm": None,
        "refined_translation_mm": None,
        "refined_add_0.1d": None,
        "delta_add_mm": None,
        "delta_translation_mm": None,
    }
    if refined is not None:
        row.update(
            {
                "refined_add_mm": refined.get("add_mm"),
                "refined_translation_mm": refined.get("translation_error_mm"),
                "refined_add_0.1d": refined.get("pose_success_add_0.1d"),
                "delta_add_mm": refined.get("add_mm", 0.0) - raw.get("add_mm", 0.0),
                "delta_translation_mm": refined.get("translation_error_mm", 0.0) - raw.get("translation_error_mm", 0.0),
            }
        )
    return row


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]], commands: list[dict[str, Any]]) -> None:
    headers = [
        "object",
        "dataset",
        "samples",
        "iou",
        "raw_add",
        "raw_trans",
        "raw_0.1d",
        "ref_add",
        "ref_trans",
        "ref_0.1d",
        "d_add",
        "d_trans",
    ]
    lines = ["# Pose Evaluation Suite", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = [
            row["object"],
            row["dataset"],
            row["sample_count"],
            fmt(row["matched_crop_iou"]),
            fmt(row["raw_add_mm"]),
            fmt(row["raw_translation_mm"]),
            fmt(row["raw_add_0.1d"]),
            fmt(row["refined_add_mm"]),
            fmt(row["refined_translation_mm"]),
            fmt(row["refined_add_0.1d"]),
            fmt(row["delta_add_mm"]),
            fmt(row["delta_translation_mm"]),
        ]
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    lines.extend(["", "## Commands", ""])
    for record in commands:
        status = "ok" if record["returncode"] == 0 else f"failed:{record['returncode']}"
        lines.append(f"- `{status}` `{' '.join(record['command'])}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "experiments" / f"pose_eval_suite_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    commands: list[dict[str, Any]] = []
    if not args.skip_compile:
        compile_cmd = [sys.executable, "-m", "py_compile", *POSE_COMPILE_TARGETS]
        commands.append(run_checked(compile_cmd))

    refine_args: list[str] = []
    if args.refine_pose:
        refine_args = [
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

    objects = [
        {
            "name": "K41144",
            "config": args.k41144_pose_config,
            "checkpoint": args.k41144_pose_checkpoint,
            "gt_val": args.k41144_gt_val,
            "gt_test": args.k41144_gt_test,
            "pred_test": args.k41144_pred_test,
            "translation_refiner": args.k41144_translation_refiner_checkpoint,
        },
        {
            "name": "bending_pipe",
            "config": args.bending_pose_config,
            "checkpoint": args.bending_pose_checkpoint,
            "gt_val": args.bending_gt_val,
            "gt_test": args.bending_gt_test,
            "pred_test": args.bending_pred_test,
            "translation_refiner": args.bending_translation_refiner_checkpoint,
        },
    ]
    rows: list[dict[str, Any]] = []
    for obj in objects:
        identity_path = out_dir / f"{obj['name']}_identity_val.json"
        command = eval_command(
            config=obj["config"],
            data=obj["gt_val"],
            out=identity_path,
            checkpoint=None,
            device=args.device,
            identity_baseline=True,
            limit_samples=args.limit_samples,
        )
        commands.append(run_checked(command))
        rows.append(row_from_metrics(object_name=obj["name"], dataset_name="identity_val", path=identity_path))

        gt_path = out_dir / f"{obj['name']}_gt_test.json"
        command = eval_command(
            config=obj["config"],
            data=obj["gt_test"],
            out=gt_path,
            checkpoint=obj["checkpoint"],
            device=args.device,
            limit_samples=args.limit_samples,
            refine_args=refine_args,
            translation_refiner_checkpoint=obj["translation_refiner"],
        )
        commands.append(run_checked(command))
        rows.append(row_from_metrics(object_name=obj["name"], dataset_name="gt_test", path=gt_path))

        pred_iou_path = out_dir / f"{obj['name']}_pred_test_iou05.json"
        command = eval_command(
            config=obj["config"],
            data=obj["pred_test"],
            out=pred_iou_path,
            checkpoint=obj["checkpoint"],
            device=args.device,
            min_matched_gt_iou=0.5,
            limit_samples=args.limit_samples,
            refine_args=refine_args,
            translation_refiner_checkpoint=obj["translation_refiner"],
        )
        commands.append(run_checked(command))
        rows.append(row_from_metrics(object_name=obj["name"], dataset_name="pred_test_iou05", path=pred_iou_path))

        pred_all_path = out_dir / f"{obj['name']}_pred_test_all.json"
        command = eval_command(
            config=obj["config"],
            data=obj["pred_test"],
            out=pred_all_path,
            checkpoint=obj["checkpoint"],
            device=args.device,
            limit_samples=args.limit_samples,
            refine_args=refine_args,
            translation_refiner_checkpoint=obj["translation_refiner"],
        )
        commands.append(run_checked(command))
        rows.append(row_from_metrics(object_name=obj["name"], dataset_name="pred_test_all", path=pred_all_path))

    summary = {
        "out_dir": str(out_dir),
        "device": args.device,
        "limit_samples": args.limit_samples,
        "refine_pose": bool(args.refine_pose),
        "rows": rows,
        "commands": commands,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(out_dir / "summary.md", rows, commands)
    print(f"[pose-suite] wrote summary: {out_dir / 'summary.json'}")
    print(f"[pose-suite] wrote markdown: {out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
