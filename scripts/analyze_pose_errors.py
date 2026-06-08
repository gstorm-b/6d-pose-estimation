"""Analyze per-crop pose errors and optionally export worst-case PLY previews."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pose_crop_dataset import build_pose_crop_dataset_from_config  # noqa: E402
from src.models.pointnet2_pose import build_pointnet2_pose_from_config  # noqa: E402
from src.training.checkpoint import load_checkpoint  # noqa: E402
from src.training.config import load_config  # noqa: E402
from src.training.pose_metrics import (  # noqa: E402
    CenterVoteAggregationConfig,
    compute_pose_metrics_per_sample,
    pose_predictions_to_object_to_camera,
)
from src.training.seed import seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--min-matched-gt-iou", type=float)
    parser.add_argument("--center-vote-aggregation", default="mean", choices=("mean", "median", "trimmed_mean", "weighted_trimmed_mean", "geometric_median"))
    parser.add_argument("--center-vote-trim-fraction", type=float, default=0.1)
    parser.add_argument("--center-vote-max-residual-fraction", type=float)
    parser.add_argument("--center-vote-geometric-median-iterations", type=int, default=8)
    parser.add_argument("--translation-threshold-mm", type=float, default=5.0)
    parser.add_argument("--rotation-threshold-deg", type=float, default=20.0)
    parser.add_argument("--low-iou-threshold", type=float, default=0.5)
    parser.add_argument("--tiny-crop-points", type=int, default=96)
    parser.add_argument("--export-worst-ply", type=int, default=0)
    parser.add_argument("--ply-dir", help="Defaults to <out stem>_ply next to --out.")
    return parser.parse_args()


def move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    import torch

    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def tensor_at(tensor: Any, index: int) -> Any:
    if hasattr(tensor, "detach"):
        value = tensor[index].detach().cpu()
        if value.ndim == 0:
            return value.item()
        return value.numpy().tolist()
    return tensor[index]


def metric_at(metrics: dict[str, Any], index: int) -> dict[str, float]:
    return {key: float(value[index].detach().cpu()) for key, value in metrics.items()}


def transform(points_object: np.ndarray, rotation_object_to_camera: np.ndarray, origin_camera: np.ndarray) -> np.ndarray:
    return points_object @ rotation_object_to_camera.T + origin_camera.reshape(1, 3)


def write_colored_ply(path: Path, groups: list[tuple[np.ndarray, tuple[int, int, int]]]) -> None:
    vertices = []
    for points, color in groups:
        color_arr = np.asarray(color, dtype=np.uint8)
        for point in np.asarray(points, dtype=np.float32):
            vertices.append((float(point[0]), float(point[1]), float(point[2]), int(color_arr[0]), int(color_arr[1]), int(color_arr[2])))
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for vertex in vertices:
            f.write(f"{vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f} {vertex[3]} {vertex[4]} {vertex[5]}\n")


def origin_marker(origin: np.ndarray, diameter: float) -> np.ndarray:
    radius = max(float(diameter) * 0.02, 0.001)
    offsets = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [radius, 0.0, 0.0],
            [-radius, 0.0, 0.0],
            [0.0, radius, 0.0],
            [0.0, -radius, 0.0],
            [0.0, 0.0, radius],
            [0.0, 0.0, -radius],
        ],
        dtype=np.float32,
    )
    return origin.reshape(1, 3).astype(np.float32) + offsets


def bucket_record(record: dict[str, Any], args: argparse.Namespace) -> list[str]:
    buckets = []
    large_t = record["translation_error_mm"] > args.translation_threshold_mm
    large_r = record["rotation_error_deg_symmetry_aware"] > args.rotation_threshold_deg
    if large_t and not large_r:
        buckets.append("large_translation_small_rotation")
    if large_r and not large_t:
        buckets.append("small_translation_large_rotation")
    if large_t and large_r:
        buckets.append("large_translation_large_rotation")
    if record["matched_gt_iou"] < args.low_iou_threshold:
        buckets.append("low_matched_iou")
    if record["crop_point_count"] < args.tiny_crop_points:
        buckets.append("tiny_crop")
    if not buckets:
        buckets.append("ok_or_mild_error")
    return buckets


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", config.get("train", {}).get("seed", 7)))
    seed_everything(seed)

    import torch
    from torch.utils.data import DataLoader

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    center_vote_config = CenterVoteAggregationConfig(
        method=args.center_vote_aggregation,
        trim_fraction=args.center_vote_trim_fraction,
        max_residual_fraction=args.center_vote_max_residual_fraction,
        geometric_median_iterations=args.center_vote_geometric_median_iterations,
    )
    dataset = build_pose_crop_dataset_from_config(config, args.data, min_matched_gt_iou=args.min_matched_gt_iou)
    if args.limit_samples is not None:
        if args.limit_samples <= 0:
            raise ValueError("--limit-samples must be positive")
        dataset.records = dataset.records[: args.limit_samples]
    batch_size = int(args.batch_size or config.get("train", {}).get("batch_size", 16))
    workers = int(args.num_workers if args.num_workers is not None else config.get("train", {}).get("num_workers", 0))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, drop_last=False)

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    checkpoint_config = checkpoint.get("config", config)
    model = build_pointnet2_pose_from_config(checkpoint_config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    records: list[dict[str, Any]] = []
    preview_payloads: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch["points_pose_input"], batch.get("crop_aux_features"))
            pred = pose_predictions_to_object_to_camera(outputs, batch, center_vote_config=center_vote_config)
            metrics = compute_pose_metrics_per_sample(outputs, batch, center_vote_config=center_vote_config)
            for index, sample_name in enumerate(batch["sample_name"]):
                row_metrics = metric_at(metrics, index)
                row = {
                    "sample_name": sample_name,
                    "crop_index": int(tensor_at(batch["crop_index"], index)),
                    "matched_gt_iou": float(tensor_at(batch["matched_gt_iou"], index)),
                    "crop_point_count": int(tensor_at(batch["crop_point_count"], index)),
                    "translation_error_mm": row_metrics["translation_error_mm"],
                    "rotation_error_deg_symmetry_aware": row_metrics["rotation_error_deg_symmetry_aware"],
                    "add_mm": row_metrics["add_mm"],
                    "pose_success_add_0.1d": row_metrics["pose_success_add_0.1d"],
                    "pose_success_add_0.05d": row_metrics["pose_success_add_0.05d"],
                    "predicted_origin_camera": tensor_at(pred["object_origin_camera"], index),
                    "gt_origin_camera": tensor_at(batch["object_origin_camera"], index),
                }
                row["origin_delta_camera_mm"] = (
                    (np.asarray(row["predicted_origin_camera"], dtype=np.float32) - np.asarray(row["gt_origin_camera"], dtype=np.float32)) * 1000.0
                ).tolist()
                row["buckets"] = bucket_record(row, args)
                records.append(row)
                if args.export_worst_ply > 0:
                    model_points = np.asarray(tensor_at(batch["model_points_object"], index), dtype=np.float32)
                    if model_points.ndim == 3:
                        model_points = model_points[0]
                    payload = {
                        "record": row,
                        "crop_points": np.asarray(tensor_at(batch["sampled_points_camera"], index), dtype=np.float32),
                        "pred_model_points": transform(
                            model_points,
                            np.asarray(tensor_at(pred["rotation_object_to_camera"], index), dtype=np.float32),
                            np.asarray(row["predicted_origin_camera"], dtype=np.float32),
                        ),
                        "gt_model_points": transform(
                            model_points,
                            np.asarray(tensor_at(batch["rotation_object_to_camera"], index), dtype=np.float32),
                            np.asarray(row["gt_origin_camera"], dtype=np.float32),
                        ),
                        "pred_origin": np.asarray(row["predicted_origin_camera"], dtype=np.float32),
                        "gt_origin": np.asarray(row["gt_origin_camera"], dtype=np.float32),
                        "diameter": float(tensor_at(batch["model_diameter_m"], index)),
                    }
                    preview_payloads.append(payload)
            print(f"[pose-audit] processed batch, total_records={len(records)}")

    bucket_counts: dict[str, int] = {}
    for record in records:
        for bucket in record["buckets"]:
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    sorted_records = sorted(records, key=lambda item: (item["add_mm"], item["translation_error_mm"]), reverse=True)
    output_path = Path(args.out)
    ply_dir = Path(args.ply_dir) if args.ply_dir else output_path.with_suffix("").parent / f"{output_path.with_suffix('').name}_ply"
    preview_files = []
    if args.export_worst_ply > 0:
        ply_dir.mkdir(parents=True, exist_ok=True)
        rank_by_key = {
            (record["sample_name"], record["crop_index"]): rank
            for rank, record in enumerate(sorted_records[: args.export_worst_ply], start=1)
        }
        for payload in preview_payloads:
            key = (payload["record"]["sample_name"], payload["record"]["crop_index"])
            if key not in rank_by_key:
                continue
            path = ply_dir / f"worst_{rank_by_key[key]:03d}_{payload['record']['sample_name']}_crop_{payload['record']['crop_index']}.ply"
            write_colored_ply(
                path,
                [
                    (payload["crop_points"], (180, 180, 180)),
                    (payload["pred_model_points"], (230, 60, 60)),
                    (payload["gt_model_points"], (60, 210, 80)),
                    (origin_marker(payload["pred_origin"], payload["diameter"]), (30, 80, 255)),
                    (origin_marker(payload["gt_origin"], payload["diameter"]), (255, 220, 40)),
                ],
            )
            preview_files.append(str(path))

    metrics_mean = {
        "translation_error_mm": float(np.mean([row["translation_error_mm"] for row in records])) if records else 0.0,
        "rotation_error_deg_symmetry_aware": float(np.mean([row["rotation_error_deg_symmetry_aware"] for row in records])) if records else 0.0,
        "add_mm": float(np.mean([row["add_mm"] for row in records])) if records else 0.0,
        "pose_success_add_0.1d": float(np.mean([row["pose_success_add_0.1d"] for row in records])) if records else 0.0,
        "pose_success_add_0.05d": float(np.mean([row["pose_success_add_0.05d"] for row in records])) if records else 0.0,
        "matched_crop_iou": float(np.mean([row["matched_gt_iou"] for row in records])) if records else 0.0,
    }
    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "data": args.data,
        "device": device,
        "sample_count": len(records),
        "center_vote_aggregation": {
            "method": center_vote_config.method,
            "trim_fraction": center_vote_config.trim_fraction,
            "max_residual_fraction": center_vote_config.max_residual_fraction,
            "geometric_median_iterations": center_vote_config.geometric_median_iterations,
        },
        "thresholds": {
            "translation_threshold_mm": args.translation_threshold_mm,
            "rotation_threshold_deg": args.rotation_threshold_deg,
            "low_iou_threshold": args.low_iou_threshold,
            "tiny_crop_points": args.tiny_crop_points,
        },
        "metrics_mean": metrics_mean,
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "worst_records": sorted_records[:50],
        "preview_files": preview_files,
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[pose-audit] wrote audit: {output_path}")
    print(f"[pose-audit] metrics_mean: {metrics_mean}")
    print(f"[pose-audit] bucket_counts: {summary['bucket_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
