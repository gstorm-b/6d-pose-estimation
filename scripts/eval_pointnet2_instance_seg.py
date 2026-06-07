"""Evaluate PointNet++ instance segmentation on a held-out split."""

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
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from preview_pointnet2_instance_predictions import (  # noqa: E402
    build_clustering_config,
    default_output_dir,
    error_colors,
    instance_labels_to_colors,
    parse_eps_sweep,
    resolve_data_root,
    safe_sample_name,
    write_ply,
)
from src.data.augmentation import PointCloudAugmentationConfig  # noqa: E402
from src.data.pointnet2_instance_dataset import PointNet2InstanceSegDataset  # noqa: E402
from src.inference.instance_clustering import (  # noqa: E402
    InstanceMetricConfig,
    VotedCenterClusteringConfig,
    cluster_voted_centers,
    evaluate_instance_predictions,
)
from src.models.pointnet2_instance_seg import build_pointnet2_instance_seg_from_config  # noqa: E402
from src.training.checkpoint import load_checkpoint  # noqa: E402
from src.training.metrics import SegmentationMeter  # noqa: E402
from src.training.seed import seed_everything  # noqa: E402


OFFSET_KEYS = [
    "offset_l1_object",
    "offset_l1_near_centroid",
    "offset_l1_far_centroid",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a PointNet++ instance checkpoint.")
    parser.add_argument("--data", help="Processed dataset root. Defaults to checkpoint runtime/config data.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--out", required=True, help="Output JSON metrics path.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--normalize", help="Override dataset normalization mode.")
    parser.add_argument("--ops-backend", default="pytorch3d", help="Override PointNet++ ops backend for eval.")
    parser.add_argument("--seed", type=int, help="Override eval RNG seed. Defaults to checkpoint config seed.")
    parser.add_argument("--limit-samples", type=int, help="Evaluate only the first N samples.")
    parser.add_argument("--preview-out", help="Optional folder for PLY previews.")
    parser.add_argument("--preview-samples", type=int, default=0, help="Number of leading samples to export as PLY previews.")
    parser.add_argument("--object-probability-threshold", type=float, help="Override object probability threshold.")
    parser.add_argument("--dbscan-eps-m", type=float, help="Override DBSCAN eps in meters.")
    parser.add_argument("--dbscan-eps-sweep-m", help="Comma-separated DBSCAN eps sweep in meters.")
    parser.add_argument("--dbscan-min-samples", type=int, help="Override DBSCAN min_samples.")
    parser.add_argument("--min-cluster-points", type=int, help="Override minimum accepted cluster points.")
    return parser.parse_args()


def build_dataset(config: dict[str, Any], args: argparse.Namespace, seed: int) -> tuple[PointNet2InstanceSegDataset, Path]:
    data_root = resolve_data_root(config, args.data)
    dataset_config = config.get("dataset", {})
    use_normals = config.get("model", {}).get("input_features", "xyz_normal") == "xyz_normal"
    normalize = args.normalize or dataset_config.get("normalize", "scene_center")
    min_points_per_instance = int(config.get("loss", {}).get("min_points_per_instance", dataset_config.get("min_points_per_instance", 16)))
    dataset = PointNet2InstanceSegDataset(
        data_root,
        args.split,
        use_normals=use_normals,
        normalize=normalize,
        min_points_per_instance=min_points_per_instance,
        max_instances=int(dataset_config.get("max_instances", 128)),
        augment=PointCloudAugmentationConfig(enabled=False),
        seed=seed,
    )
    if args.limit_samples is not None:
        if args.limit_samples <= 0:
            raise ValueError("--limit-samples must be positive")
        dataset.files = dataset.files[: args.limit_samples]
    if not dataset.files:
        raise ValueError(f"No samples found for split {args.split!r} under {data_root}")
    return dataset, data_root


def offset_metrics(predicted_offsets: np.ndarray, target_offsets: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    valid = np.asarray(valid_mask, dtype=np.bool_)
    if not np.any(valid):
        return {key: 0.0 for key in OFFSET_KEYS}
    absolute_error = np.abs(predicted_offsets - target_offsets).mean(axis=-1)
    distances = np.linalg.norm(target_offsets, axis=-1)
    valid_distances = distances[valid]
    threshold = float(np.median(valid_distances))
    near_mask = valid & (distances <= threshold)
    far_mask = valid & (distances > threshold)
    return {
        "offset_l1_object": float(absolute_error[valid].mean()),
        "offset_l1_near_centroid": float(absolute_error[near_mask].mean()) if np.any(near_mask) else 0.0,
        "offset_l1_far_centroid": float(absolute_error[far_mask].mean()) if np.any(far_mask) else 0.0,
    }


def aggregate_offset_metrics(samples: list[dict[str, Any]]) -> dict[str, float]:
    if not samples:
        return {key: 0.0 for key in OFFSET_KEYS}
    return {key: float(np.mean([float(sample["offset_metrics"].get(key, 0.0)) for sample in samples])) for key in OFFSET_KEYS}


def aggregate_sweep_metrics(samples: list[dict[str, Any]]) -> list[dict[str, float]]:
    rows_by_eps: dict[float, list[dict[str, Any]]] = {}
    for sample in samples:
        for row in sample.get("dbscan_eps_sweep", []):
            rows_by_eps.setdefault(float(row["dbscan_eps_m"]), []).append(row)

    result: list[dict[str, float]] = []
    for eps, rows in sorted(rows_by_eps.items()):
        result.append(
            {
                "dbscan_eps_m": float(eps),
                "cluster_count_pred": float(np.mean([float(row["cluster_count_pred"]) for row in rows])),
                "instance_precision": float(np.mean([float(row["instance_precision"]) for row in rows])),
                "instance_recall": float(np.mean([float(row["instance_recall"]) for row in rows])),
                "instance_mean_iou": float(np.mean([float(row["instance_mean_iou"]) for row in rows])),
                "merge_count": float(np.mean([float(row["merge_count"]) for row in rows])),
                "split_count": float(np.mean([float(row["split_count"]) for row in rows])),
            }
        )
    return result


def mean_instance_metrics(samples: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "cluster_count_gt",
        "cluster_count_pred",
        "ignored_gt_instances",
        "instance_precision",
        "instance_recall",
        "instance_mean_iou",
        "matched_instance_count",
        "unmatched_gt_count",
        "unmatched_pred_count",
        "merge_count",
        "split_count",
    ]
    if not samples:
        return {key: 0.0 for key in keys}
    return {key: float(np.mean([float(sample["instance_metrics"].get(key, 0.0)) for sample in samples])) for key in keys}


def export_preview_ply(
    output_dir: Path,
    raw_sample: str,
    points_camera: np.ndarray,
    target_instances: np.ndarray,
    predicted_instances: np.ndarray,
    voted_centers: np.ndarray,
    instance_metrics: dict[str, Any],
) -> dict[str, str]:
    safe_name = safe_sample_name(raw_sample)
    files = {
        "gt_instances": f"{safe_name}_gt_instances.ply",
        "pred_instances": f"{safe_name}_pred_instances.ply",
        "error": f"{safe_name}_error.ply",
        "voted_centers": f"{safe_name}_voted_centers.ply",
    }
    write_ply(output_dir / files["gt_instances"], points_camera, instance_labels_to_colors(target_instances))
    write_ply(output_dir / files["pred_instances"], points_camera, instance_labels_to_colors(predicted_instances))
    write_ply(output_dir / files["error"], points_camera, error_colors(target_instances, predicted_instances, instance_metrics))
    voted_colors = instance_labels_to_colors(target_instances)
    voted_colors[target_instances <= 0] = instance_labels_to_colors(predicted_instances)[target_instances <= 0]
    write_ply(output_dir / files["voted_centers"], voted_centers, voted_colors)
    return files


def evaluate_sample(
    model: Any,
    item: dict[str, Any],
    device: str,
    clustering_config: VotedCenterClusteringConfig,
    metric_config: InstanceMetricConfig,
    eps_sweep: list[float],
    num_classes: int,
    preview_dir: Path | None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    import torch

    raw_sample = str(item["raw_sample"])
    points_camera = item["points_camera"].numpy()
    target_semantic = item["semantic_labels"].numpy().astype(np.int64)
    target_instances = item["instance_labels"].numpy().astype(np.int64)
    target_offsets = item["centroid_offsets"].numpy().astype(np.float32)
    valid_instance_mask = item["valid_instance_mask"].numpy().astype(np.bool_)
    inputs = item["model_input"].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(inputs)
        probabilities = torch.softmax(outputs["semantic_logits"], dim=-1).squeeze(0).cpu().numpy()
        predicted_offsets = outputs["offsets"].squeeze(0).cpu().numpy().astype(np.float32)

    object_probabilities = probabilities[:, 1] if num_classes > 1 else np.zeros(points_camera.shape[0], dtype=np.float32)
    predicted_semantic = (object_probabilities >= clustering_config.object_probability_threshold).astype(np.int64)
    cluster_result = cluster_voted_centers(points_camera, object_probabilities, predicted_offsets, clustering_config)
    predicted_instances = np.asarray(cluster_result["instance_labels"], dtype=np.int64)
    voted_centers = np.asarray(cluster_result["voted_centers"], dtype=np.float32)
    instance_metrics = evaluate_instance_predictions(target_instances, predicted_instances, metric_config)
    sample_offset_metrics = offset_metrics(predicted_offsets, target_offsets, valid_instance_mask)

    semantic_meter = SegmentationMeter(num_classes)
    semantic_meter.update(predicted_semantic, target_semantic)
    sample_summary: dict[str, Any] = {
        "raw_sample": raw_sample,
        "source_path": item["path"],
        "point_count": int(points_camera.shape[0]),
        "object_probability_min": float(object_probabilities.min()),
        "object_probability_mean": float(object_probabilities.mean()),
        "object_probability_max": float(object_probabilities.max()),
        "predicted_object_points": int(np.count_nonzero(predicted_semantic == 1)),
        "gt_object_points": int(np.count_nonzero(target_semantic == 1)),
        "semantic_metrics": semantic_meter.compute(),
        "offset_metrics": sample_offset_metrics,
        "instance_metrics": instance_metrics,
        "dbscan_eps_m": clustering_config.dbscan_eps_m,
    }

    if preview_dir is not None:
        sample_summary["preview_files"] = export_preview_ply(
            preview_dir,
            raw_sample,
            points_camera,
            target_instances,
            predicted_instances,
            voted_centers,
            instance_metrics,
        )

    sweep_rows = []
    for eps in eps_sweep:
        sweep_config = VotedCenterClusteringConfig(
            object_probability_threshold=clustering_config.object_probability_threshold,
            dbscan_eps_m=float(eps),
            dbscan_min_samples=clustering_config.dbscan_min_samples,
            min_cluster_points=clustering_config.min_cluster_points,
        )
        sweep_cluster = cluster_voted_centers(points_camera, object_probabilities, predicted_offsets, sweep_config)
        sweep_metrics = evaluate_instance_predictions(
            target_instances,
            np.asarray(sweep_cluster["instance_labels"], dtype=np.int64),
            metric_config,
        )
        sweep_rows.append(
            {
                "dbscan_eps_m": float(eps),
                "cluster_count_pred": sweep_metrics["cluster_count_pred"],
                "instance_precision": sweep_metrics["instance_precision"],
                "instance_recall": sweep_metrics["instance_recall"],
                "instance_mean_iou": sweep_metrics["instance_mean_iou"],
                "merge_count": sweep_metrics["merge_count"],
                "split_count": sweep_metrics["split_count"],
            }
        )
    sample_summary["dbscan_eps_sweep"] = sweep_rows
    return sample_summary, predicted_semantic, target_semantic


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    if args.ops_backend:
        config.setdefault("model", {})["ops_backend"] = args.ops_backend
    num_classes = int(config.get("model", {}).get("num_classes", 2))
    seed = int(args.seed if args.seed is not None else config.get("seed", config.get("train", {}).get("seed", 7)))
    seed_everything(seed)

    import torch

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = build_pointnet2_instance_seg_from_config(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    dataset, data_root = build_dataset(config, args, seed)
    clustering_config = build_clustering_config(config, args)
    metric_config = InstanceMetricConfig.from_mapping(config.get("metrics", {}))
    eps_sweep = parse_eps_sweep(args.dbscan_eps_sweep_m, config, clustering_config.dbscan_eps_m)
    preview_dir: Path | None = None
    if args.preview_out:
        preview_dir = Path(args.preview_out)
    elif args.preview_samples > 0:
        preview_dir = default_output_dir(checkpoint_path, args.split) / "eval_previews"
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    overall_semantic = SegmentationMeter(num_classes)
    samples: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        item = dataset[index]
        sample_preview_dir = preview_dir if index < args.preview_samples else None
        sample_summary, predicted_semantic, target_semantic = evaluate_sample(
            model,
            item,
            device,
            clustering_config,
            metric_config,
            eps_sweep,
            num_classes,
            sample_preview_dir,
        )
        samples.append(sample_summary)
        overall_semantic.update(predicted_semantic, target_semantic)
        semantic = sample_summary["semantic_metrics"]
        instance = sample_summary["instance_metrics"]
        offsets = sample_summary["offset_metrics"]
        print(
            f"[eval] {sample_summary['raw_sample']}: "
            f"pred_clusters={instance['cluster_count_pred']} "
            f"gt_clusters={instance['cluster_count_gt']} "
            f"instance_miou={instance['instance_mean_iou']:.4f} "
            f"instance_recall={instance['instance_recall']:.4f} "
            f"semantic_object_iou={semantic.get('object_iou', 0.0):.4f} "
            f"offset_l1={offsets['offset_l1_object']:.4f}"
        )

    summary = {
        "checkpoint": str(checkpoint_path),
        "data_root": str(data_root),
        "split": args.split,
        "device": device,
        "ops_backend_requested": args.ops_backend,
        "ops_backend_effective": getattr(model, "ops_backend_name", "unknown"),
        "seed": seed,
        "sample_count": len(samples),
        "preview_dir": str(preview_dir) if preview_dir is not None else None,
        "preview_samples": int(min(args.preview_samples, len(samples))) if preview_dir is not None else 0,
        "clustering": {
            "object_probability_threshold": clustering_config.object_probability_threshold,
            "dbscan_eps_m": clustering_config.dbscan_eps_m,
            "dbscan_eps_sweep_m": eps_sweep,
            "dbscan_min_samples": clustering_config.dbscan_min_samples,
            "min_cluster_points": clustering_config.min_cluster_points,
        },
        "metrics_config": {
            "instance_iou_threshold": metric_config.instance_iou_threshold,
            "ignore_gt_instances_below_points": metric_config.ignore_gt_instances_below_points,
            "merge_split_overlap_threshold": metric_config.merge_split_overlap_threshold,
        },
        "overall_semantic_metrics": overall_semantic.compute(),
        "mean_offset_metrics": aggregate_offset_metrics(samples),
        "mean_instance_metrics": mean_instance_metrics(samples),
        "mean_dbscan_eps_sweep": aggregate_sweep_metrics(samples),
        "samples": samples,
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[eval] wrote metrics: {output_path}")
    print(f"[eval] semantic metrics: {summary['overall_semantic_metrics']}")
    print(f"[eval] offset metrics: {summary['mean_offset_metrics']}")
    print(f"[eval] instance metrics: {summary['mean_instance_metrics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
