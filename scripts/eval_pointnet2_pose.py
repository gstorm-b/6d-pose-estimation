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
from src.models.pointnet2_pose_refiner import (  # noqa: E402
    PoseRefinerFeatureConfig,
    build_pointnet2_pose_refiner_from_config,
    build_pose_refiner_features,
)
from src.training.checkpoint import load_checkpoint  # noqa: E402
from src.training.config import load_config  # noqa: E402
from src.training.pose_geometry import matrix_to_rotation_6d  # noqa: E402
from src.training.pose_metrics import (  # noqa: E402
    CenterVoteAggregationConfig,
    PoseMetricMeter,
    compute_pose_metrics,
    compute_pose_metrics_per_sample,
    compute_pose_metrics_per_sample_from_pose,
    pose_predictions_to_object_to_camera,
)
from src.training.seed import seed_everything  # noqa: E402
from src.inference.pose_refinement import PoseRefinementConfig, refine_pose_batch  # noqa: E402


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
    parser.add_argument("--center-vote-aggregation", default="mean", choices=("mean", "median", "trimmed_mean", "weighted_trimmed_mean", "geometric_median"))
    parser.add_argument("--center-vote-trim-fraction", type=float, default=0.1)
    parser.add_argument("--center-vote-max-residual-fraction", type=float)
    parser.add_argument("--center-vote-geometric-median-iterations", type=int, default=8)
    parser.add_argument("--refine-pose", action="store_true", help="Report optional refined-pose metrics in addition to raw metrics.")
    parser.add_argument("--refinement-method", default="translation_only_icp", choices=("translation_only_icp", "point_to_point_icp", "hybrid"))
    parser.add_argument("--refinement-max-iterations", type=int, default=10)
    parser.add_argument("--refinement-distance-threshold-fraction", type=float, default=0.08)
    parser.add_argument("--refinement-max-translation-delta-fraction", type=float, default=0.10)
    parser.add_argument("--refinement-max-rotation-delta-deg", type=float, default=10.0)
    parser.add_argument("--refinement-accept-if-improved", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--translation-refiner-checkpoint", help="Optional learned translation-refiner checkpoint.")
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


def average_tensor_metrics(totals: dict[str, float], weight: int) -> dict[str, float]:
    divisor = max(weight, 1)
    return {key: value / divisor for key, value in sorted(totals.items())}


def update_tensor_metric_totals(totals: dict[str, float], metrics: dict[str, Any]) -> int:
    first = next(iter(metrics.values()))
    batch_size = int(first.shape[0])
    for key, value in metrics.items():
        totals[key] = totals.get(key, 0.0) + float(value.detach().cpu().sum()) / max(batch_size, 1) * batch_size
    return batch_size


def tensor_metric_at(metrics: dict[str, Any], index: int) -> dict[str, float]:
    return {key: float(value[index].detach().cpu()) for key, value in metrics.items()}


def update_refined_comparison_totals(
    totals: dict[str, float],
    *,
    raw_metrics: dict[str, Any],
    refined_metrics: dict[str, Any],
    batch: dict[str, Any],
) -> None:
    add_delta_m = refined_metrics["add_m"] - raw_metrics["add_m"]
    translation_delta_mm = refined_metrics["translation_error_mm"] - raw_metrics["translation_error_mm"]
    rotation_delta_deg = refined_metrics["rotation_error_deg_symmetry_aware"] - raw_metrics["rotation_error_deg_symmetry_aware"]
    diameter = batch["model_diameter_m"].to(device=add_delta_m.device, dtype=add_delta_m.dtype).view(-1)
    totals["delta_add_m_sum"] = totals.get("delta_add_m_sum", 0.0) + float(add_delta_m.detach().cpu().sum())
    totals["delta_add_mm_sum"] = totals.get("delta_add_mm_sum", 0.0) + float((add_delta_m * 1000.0).detach().cpu().sum())
    totals["delta_translation_mm_sum"] = totals.get("delta_translation_mm_sum", 0.0) + float(translation_delta_mm.detach().cpu().sum())
    totals["delta_rotation_deg_sum"] = totals.get("delta_rotation_deg_sum", 0.0) + float(rotation_delta_deg.detach().cpu().sum())
    totals["improved_add_count"] = totals.get("improved_add_count", 0.0) + float((add_delta_m < 0).detach().cpu().sum())
    totals["worsened_add_count"] = totals.get("worsened_add_count", 0.0) + float((add_delta_m > 0).detach().cpu().sum())
    totals["unchanged_add_count"] = totals.get("unchanged_add_count", 0.0) + float((add_delta_m == 0).detach().cpu().sum())
    totals["catastrophic_add_worse_0.05d_count"] = totals.get("catastrophic_add_worse_0.05d_count", 0.0) + float(
        (add_delta_m > (0.05 * diameter)).detach().cpu().sum()
    )


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
    refinement_config = PoseRefinementConfig(
        method=args.refinement_method,
        max_iterations=args.refinement_max_iterations,
        distance_threshold_fraction=args.refinement_distance_threshold_fraction,
        max_translation_delta_fraction=args.refinement_max_translation_delta_fraction,
        max_rotation_delta_deg=args.refinement_max_rotation_delta_deg,
        accept_if_improved=bool(args.refinement_accept_if_improved),
    )

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
    translation_refiner = None
    translation_refiner_checkpoint_config = None
    if not args.identity_baseline:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required unless --identity-baseline is set")
        checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
        checkpoint_config = checkpoint.get("config", config)
        model = build_pointnet2_pose_from_config(checkpoint_config)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        model.eval()
        if args.translation_refiner_checkpoint:
            refiner_checkpoint = load_checkpoint(args.translation_refiner_checkpoint, map_location="cpu")
            translation_refiner_checkpoint_config = refiner_checkpoint.get("config", {})
            translation_refiner = build_pointnet2_pose_refiner_from_config(translation_refiner_checkpoint_config)
            translation_refiner.load_state_dict(refiner_checkpoint["model_state"])
            translation_refiner.to(device)
            translation_refiner.eval()

    meter = PoseMetricMeter(center_vote_config=center_vote_config)
    refined_totals: dict[str, float] = {}
    refinement_totals: dict[str, float] = {}
    translation_refiner_totals: dict[str, float] = {}
    refined_comparison_totals: dict[str, float] = {}
    refined_weight = 0
    feature_config = PoseRefinerFeatureConfig(center_vote_config=center_vote_config)
    samples: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = identity_outputs(batch) if args.identity_baseline else model(batch["points_pose_input"], batch.get("crop_aux_features"))
            meter.update(outputs, batch)
            raw_metrics = compute_pose_metrics_per_sample(outputs, batch, center_vote_config=center_vote_config)
            refined_metrics = None
            refinement = None
            translation_refiner_row = None
            if (args.refine_pose or translation_refiner is not None) and not args.identity_baseline:
                raw_pose = pose_predictions_to_object_to_camera(outputs, batch, center_vote_config=center_vote_config)
                current_rotation = raw_pose["rotation_object_to_camera"]
                current_origin = raw_pose["object_origin_camera"]
                if translation_refiner is not None:
                    features = build_pose_refiner_features(
                        pose_outputs=outputs,
                        batch=batch,
                        raw_pose=raw_pose,
                        feature_config=feature_config,
                    )
                    refiner_outputs = translation_refiner(features)
                    diameter = batch["model_diameter_m"].to(device=current_origin.device, dtype=current_origin.dtype).view(-1, 1)
                    learned_delta = refiner_outputs["origin_delta_camera_normalized"] * diameter
                    current_origin = current_origin + learned_delta
                    learned_delta_norm = torch.linalg.norm(learned_delta, dim=-1)
                    translation_refiner_row = {
                        "translation_delta_m": learned_delta_norm,
                        "translation_delta_mm": learned_delta_norm * 1000.0,
                        "translation_delta_fraction": learned_delta_norm / diameter.view(-1).clamp_min(1e-8),
                    }
                    for key, value in translation_refiner_row.items():
                        translation_refiner_totals[key] = translation_refiner_totals.get(key, 0.0) + float(value.detach().cpu().sum())
                if args.refine_pose:
                    refinement = refine_pose_batch(
                        rotation_object_to_camera=current_rotation,
                        origin_camera=current_origin,
                        batch=batch,
                        config=refinement_config,
                    )
                else:
                    refinement = {
                        "rotation_object_to_camera": current_rotation,
                        "origin_camera": current_origin,
                    }
                    if translation_refiner_row is not None:
                        refinement.update(translation_refiner_row)
                        refinement["accepted"] = torch.ones(current_origin.shape[0], dtype=current_origin.dtype, device=current_origin.device)
                refined_metrics = compute_pose_metrics_per_sample_from_pose(
                    rotation_pred_object_to_camera=refinement["rotation_object_to_camera"],
                    origin_pred_camera=refinement["origin_camera"],
                    batch=batch,
                )
                refined_weight += update_tensor_metric_totals(refined_totals, refined_metrics)
                update_refined_comparison_totals(
                    refined_comparison_totals,
                    raw_metrics=raw_metrics,
                    refined_metrics=refined_metrics,
                    batch=batch,
                )
                for key, value in refinement.items():
                    if key in {"rotation_object_to_camera", "origin_camera"}:
                        continue
                    refinement_totals[key] = refinement_totals.get(key, 0.0) + float(value.detach().cpu().sum())
            for idx, sample_name in enumerate(batch["sample_name"]):
                row = {
                    "sample_name": sample_name,
                    "crop_index": int(batch["crop_index"][idx].detach().cpu()),
                    "matched_gt_iou": float(batch["matched_gt_iou"][idx].detach().cpu()),
                    "raw_metrics": tensor_metric_at(raw_metrics, idx),
                }
                if refined_metrics is not None and refinement is not None:
                    row["refined_metrics"] = tensor_metric_at(refined_metrics, idx)
                    row["refinement"] = {
                        key: float(value[idx].detach().cpu())
                        for key, value in refinement.items()
                        if key not in {"rotation_object_to_camera", "origin_camera"}
                    }
                    if translation_refiner_row is not None:
                        row["translation_refiner"] = {
                            key: float(value[idx].detach().cpu())
                            for key, value in translation_refiner_row.items()
                        }
                samples.append(row)

    raw_summary_metrics = meter.compute()
    refined_summary_metrics = average_tensor_metrics(refined_totals, refined_weight) if refined_weight else None
    refinement_summary = None
    if refined_weight:
        refinement_summary = {key: value / refined_weight for key, value in sorted(refinement_totals.items())}
        accepted_count = int(round(refinement_totals.get("accepted", 0.0)))
        refinement_summary.update(
            {
                "accepted_count": accepted_count,
                "rejected_count": int(refined_weight - accepted_count),
                "acceptance_rate": accepted_count / max(refined_weight, 1),
                "mean_delta_add_mm": refined_comparison_totals.get("delta_add_mm_sum", 0.0) / refined_weight,
                "mean_delta_translation_mm": refined_comparison_totals.get("delta_translation_mm_sum", 0.0) / refined_weight,
                "mean_delta_rotation_deg": refined_comparison_totals.get("delta_rotation_deg_sum", 0.0) / refined_weight,
                "improved_add_count": int(round(refined_comparison_totals.get("improved_add_count", 0.0))),
                "worsened_add_count": int(round(refined_comparison_totals.get("worsened_add_count", 0.0))),
                "unchanged_add_count": int(round(refined_comparison_totals.get("unchanged_add_count", 0.0))),
                "catastrophic_add_worse_0.05d_count": int(
                    round(refined_comparison_totals.get("catastrophic_add_worse_0.05d_count", 0.0))
                ),
            }
        )
    translation_refiner_summary = None
    if refined_weight and translation_refiner is not None:
        translation_refiner_summary = {
            key: value / refined_weight
            for key, value in sorted(translation_refiner_totals.items())
        }
    summary = {
        "config": args.config,
        "data": args.data,
        "checkpoint": args.checkpoint,
        "identity_baseline": args.identity_baseline,
        "device": device,
        "sample_count": len(dataset),
        "min_matched_gt_iou": args.min_matched_gt_iou,
        "center_vote_aggregation": {
            "method": center_vote_config.method,
            "trim_fraction": center_vote_config.trim_fraction,
            "max_residual_fraction": center_vote_config.max_residual_fraction,
            "geometric_median_iterations": center_vote_config.geometric_median_iterations,
        },
        "metrics": raw_summary_metrics,
        "raw_metrics": raw_summary_metrics,
        "refined_metrics": refined_summary_metrics,
        "refinement": {
            "enabled": bool(refined_weight),
            "geometry_enabled": bool(args.refine_pose and not args.identity_baseline),
            "config": {
                "method": refinement_config.method,
                "max_iterations": refinement_config.max_iterations,
                "distance_threshold_fraction": refinement_config.distance_threshold_fraction,
                "max_translation_delta_fraction": refinement_config.max_translation_delta_fraction,
                "max_rotation_delta_deg": refinement_config.max_rotation_delta_deg,
                "accept_if_improved": refinement_config.accept_if_improved,
            },
            "summary": refinement_summary,
        },
        "translation_refiner": {
            "enabled": bool(translation_refiner is not None),
            "checkpoint": args.translation_refiner_checkpoint,
            "summary": translation_refiner_summary,
        },
        "dataset_source": dataset.manifest.get("source"),
        "model_diameter_m": dataset.geometry.model_diameter_m,
        "symmetry_name": dataset.geometry.symmetry_name,
        "checkpoint_config": checkpoint_config,
        "translation_refiner_checkpoint_config": translation_refiner_checkpoint_config,
        "samples": samples[:100],
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[pose-eval] wrote metrics: {output_path}")
    print(f"[pose-eval] raw metrics: {summary['raw_metrics']}")
    if refined_summary_metrics is not None:
        print(f"[pose-eval] refined metrics: {refined_summary_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
