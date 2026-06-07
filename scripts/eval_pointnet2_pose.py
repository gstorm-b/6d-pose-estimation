"""Evaluate PointNet++ pose estimation on pose crop manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pose_crop_dataset import build_pose_crop_dataset_from_config  # noqa: E402
from src.models.pointnet2_pose import build_pointnet2_pose_from_config  # noqa: E402
from src.training.checkpoint import load_checkpoint  # noqa: E402
from src.training.config import load_config  # noqa: E402
from src.training.pose_geometry import matrix_to_rotation_6d  # noqa: E402
from src.training.pose_metrics import PoseMetricMeter, compute_pose_metrics  # noqa: E402
from src.training.seed import seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True, help="Pose bridge crop export root containing manifest.json.")
    parser.add_argument("--checkpoint", help="Pose checkpoint. Omit with --identity-baseline.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--limit-samples", type=int, help="Limit number of crop records.")
    parser.add_argument("--min-matched-gt-iou", type=float, help="Filter predicted crops by matched GT IoU.")
    parser.add_argument("--identity-baseline", action="store_true", help="Use GT pose as prediction to validate metrics.")
    return parser.parse_args()


def move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    import torch

    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def identity_outputs(batch: dict[str, Any]) -> dict[str, Any]:
    rotation_6d = matrix_to_rotation_6d(batch["rotation_object_to_camera"])
    return {
        "rotation_6d": rotation_6d,
        "object_origin_offset_from_crop_centroid_camera_normalized": batch[
            "object_origin_offset_from_crop_centroid_camera_normalized"
        ],
    }


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

    dataset = build_pose_crop_dataset_from_config(config, args.data, min_matched_gt_iou=args.min_matched_gt_iou)
    if args.limit_samples is not None:
        if args.limit_samples <= 0:
            raise ValueError("--limit-samples must be positive")
        dataset.records = dataset.records[: args.limit_samples]
    batch_size = int(args.batch_size or config.get("train", {}).get("batch_size", 16))
    workers = int(args.num_workers if args.num_workers is not None else config.get("train", {}).get("num_workers", 0))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, drop_last=False)

    model = None
    checkpoint_config = None
    if not args.identity_baseline:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required unless --identity-baseline is set")
        checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
        checkpoint_config = checkpoint.get("config", config)
        model = build_pointnet2_pose_from_config(checkpoint_config)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        model.eval()

    meter = PoseMetricMeter()
    samples: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = identity_outputs(batch) if args.identity_baseline else model(batch["points_pose_input"], batch.get("crop_aux_features"))
            meter.update(outputs, batch)
            batch_metrics = compute_pose_metrics(outputs, batch)
            for idx, sample_name in enumerate(batch["sample_name"]):
                samples.append(
                    {
                        "sample_name": sample_name,
                        "crop_index": int(batch["crop_index"][idx].detach().cpu()),
                        "matched_gt_iou": float(batch["matched_gt_iou"][idx].detach().cpu()),
                        "batch_translation_error_mm": batch_metrics["translation_error_mm"],
                        "batch_add_mm": batch_metrics["add_mm"],
                    }
                )

    summary = {
        "config": args.config,
        "data": args.data,
        "checkpoint": args.checkpoint,
        "identity_baseline": args.identity_baseline,
        "device": device,
        "sample_count": len(dataset),
        "min_matched_gt_iou": args.min_matched_gt_iou,
        "metrics": meter.compute(),
        "dataset_source": dataset.manifest.get("source"),
        "model_diameter_m": dataset.geometry.model_diameter_m,
        "symmetry_name": dataset.geometry.symmetry_name,
        "checkpoint_config": checkpoint_config,
        "samples": samples[:100],
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[pose-eval] wrote metrics: {output_path}")
    print(f"[pose-eval] metrics: {summary['metrics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
