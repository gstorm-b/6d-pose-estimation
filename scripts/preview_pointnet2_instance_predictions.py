"""Export PointNet++ instance predictions, voted centers, and clustering diagnostics."""

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


COLOR_BACKGROUND = np.array([120, 120, 120], dtype=np.uint8)
COLOR_TRUE_POSITIVE = np.array([40, 210, 90], dtype=np.uint8)
COLOR_FALSE_POSITIVE = np.array([235, 70, 60], dtype=np.uint8)
COLOR_FALSE_NEGATIVE = np.array([245, 205, 65], dtype=np.uint8)
COLOR_MISMATCH = np.array([190, 80, 230], dtype=np.uint8)
COLOR_NOISE = np.array([80, 80, 80], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a PointNet++ instance checkpoint.")
    parser.add_argument("--data", help="Processed dataset root. Defaults to checkpoint runtime/config data.")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--out", help="Output preview folder. Defaults to <experiment>/instance_previews/<split>.")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--normalize", help="Override dataset normalization mode.")
    parser.add_argument("--ops-backend", default="pytorch3d", help="Override PointNet++ ops backend for preview.")
    parser.add_argument("--seed", type=int, help="Override preview RNG seed. Defaults to checkpoint config seed.")
    parser.add_argument("--object-probability-threshold", type=float, help="Override object probability threshold.")
    parser.add_argument("--dbscan-eps-m", type=float, help="Override DBSCAN eps in meters.")
    parser.add_argument("--dbscan-eps-sweep-m", help="Comma-separated DBSCAN eps sweep in meters.")
    parser.add_argument("--dbscan-min-samples", type=int, help="Override DBSCAN min_samples.")
    parser.add_argument("--min-cluster-points", type=int, help="Override minimum accepted cluster points.")
    return parser.parse_args()


def resolve_data_root(config: dict[str, Any], cli_value: str | None) -> Path:
    if cli_value:
        return Path(cli_value)
    runtime_data = config.get("runtime_args", {}).get("data")
    if runtime_data:
        return Path(runtime_data)
    root_effective = config.get("dataset", {}).get("root_effective")
    if root_effective:
        return Path(root_effective)
    return Path(config.get("dataset", {}).get("root", "processed-data/pointnet2_semseg_k41144"))


def default_output_dir(checkpoint_path: Path, split: str) -> Path:
    experiment = checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent
    return experiment / "instance_previews" / split


def parse_eps_sweep(value: str | None, config: dict[str, Any], selected_eps: float) -> list[float]:
    if value:
        return [float(part.strip()) for part in value.split(",") if part.strip()]
    sweep = config.get("inference", {}).get("dbscan_eps_sweep_m")
    if isinstance(sweep, list) and sweep:
        return [float(item) for item in sweep]
    return [float(selected_eps)]


def build_clustering_config(config: dict[str, Any], args: argparse.Namespace) -> VotedCenterClusteringConfig:
    values = dict(config.get("inference", {}))
    if args.object_probability_threshold is not None:
        values["object_probability_threshold"] = args.object_probability_threshold
    if args.dbscan_eps_m is not None:
        values["dbscan_eps_m"] = args.dbscan_eps_m
    if args.dbscan_min_samples is not None:
        values["dbscan_min_samples"] = args.dbscan_min_samples
    if args.min_cluster_points is not None:
        values["min_cluster_points"] = args.min_cluster_points
    return VotedCenterClusteringConfig.from_mapping(values)


def safe_sample_name(raw_sample: str) -> str:
    return raw_sample.replace("/", "_").replace("\\", "_").replace(":", "_")


def deterministic_color(value: int) -> np.ndarray:
    if value <= 0:
        return COLOR_BACKGROUND
    state = (int(value) * 747796405 + 2891336453) & 0xFFFFFFFF
    state = ((((state >> ((state >> 28) + 4)) ^ state) * 277803737) & 0xFFFFFFFF)
    state = ((state >> 22) ^ state) & 0xFFFFFFFF
    red = 60 + int(state & 0x7F)
    green = 60 + int((state >> 8) & 0x7F)
    blue = 60 + int((state >> 16) & 0x7F)
    return np.array([red, green, blue], dtype=np.uint8)


def instance_labels_to_colors(labels: np.ndarray, *, noise_color: np.ndarray | None = None) -> np.ndarray:
    colors = np.repeat(COLOR_BACKGROUND[None, :], labels.shape[0], axis=0)
    if noise_color is not None:
        colors[labels < 0] = noise_color
    for label in sorted(int(value) for value in np.unique(labels) if int(value) > 0):
        colors[labels == label] = deterministic_color(label)
    return colors


def error_colors(target_labels: np.ndarray, predicted_labels: np.ndarray, metrics: dict[str, Any]) -> np.ndarray:
    colors = np.repeat(COLOR_BACKGROUND[None, :], target_labels.shape[0], axis=0)
    matched_pairs = {(int(match["gt_id"]), int(match["pred_id"])) for match in metrics.get("matches", [])}
    matched_gt = {gt_id for gt_id, _pred_id in matched_pairs}
    matched_pred = {pred_id for _gt_id, pred_id in matched_pairs}
    for index, (gt_id, pred_id) in enumerate(zip(target_labels, predicted_labels)):
        gt_id = int(gt_id)
        pred_id = int(pred_id)
        if gt_id <= 0 and pred_id <= 0:
            continue
        if gt_id <= 0 and pred_id > 0:
            colors[index] = COLOR_FALSE_POSITIVE
        elif gt_id > 0 and pred_id <= 0:
            colors[index] = COLOR_FALSE_NEGATIVE
        elif (gt_id, pred_id) in matched_pairs:
            colors[index] = COLOR_TRUE_POSITIVE
        elif gt_id not in matched_gt:
            colors[index] = COLOR_FALSE_NEGATIVE
        elif pred_id not in matched_pred:
            colors[index] = COLOR_FALSE_POSITIVE
        else:
            colors[index] = COLOR_MISMATCH
    return colors


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.shape[0] != colors.shape[0]:
        raise ValueError("points and colors length mismatch")
    with path.open("w", encoding="ascii") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {points.shape[0]}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("end_header\n")
        for point, color in zip(points, colors):
            file.write(
                f"{float(point[0]):.7f} {float(point[1]):.7f} {float(point[2]):.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def preview_sample(
    model: Any,
    item: dict[str, Any],
    device: str,
    output_dir: Path,
    clustering_config: VotedCenterClusteringConfig,
    metric_config: InstanceMetricConfig,
    eps_sweep: list[float],
    num_classes: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    import torch

    raw_sample = str(item["raw_sample"])
    safe_name = safe_sample_name(raw_sample)
    points_camera = item["points_camera"].numpy()
    target_semantic = item["semantic_labels"].numpy().astype(np.int64)
    target_instances = item["instance_labels"].numpy().astype(np.int64)
    inputs = item["model_input"].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(inputs)
        probabilities = torch.softmax(outputs["semantic_logits"], dim=-1).squeeze(0).cpu().numpy()
        offsets = outputs["offsets"].squeeze(0).cpu().numpy().astype(np.float32)

    object_probabilities = probabilities[:, 1] if num_classes > 1 else np.zeros(points_camera.shape[0], dtype=np.float32)
    cluster_result = cluster_voted_centers(points_camera, object_probabilities, offsets, clustering_config)
    predicted_instances = np.asarray(cluster_result["instance_labels"], dtype=np.int64)
    voted_centers = np.asarray(cluster_result["voted_centers"], dtype=np.float32)
    predicted_semantic = (object_probabilities >= clustering_config.object_probability_threshold).astype(np.int64)
    instance_metrics = evaluate_instance_predictions(target_instances, predicted_instances, metric_config)

    gt_colors = instance_labels_to_colors(target_instances)
    pred_colors = instance_labels_to_colors(predicted_instances)
    error = error_colors(target_instances, predicted_instances, instance_metrics)
    voted_colors = instance_labels_to_colors(target_instances)
    voted_colors[target_instances <= 0] = instance_labels_to_colors(predicted_instances)[target_instances <= 0]

    gt_file = f"{safe_name}_gt_instances.ply"
    pred_file = f"{safe_name}_pred_instances.ply"
    error_file = f"{safe_name}_error.ply"
    voted_file = f"{safe_name}_voted_centers.ply"
    write_ply(output_dir / gt_file, points_camera, gt_colors)
    write_ply(output_dir / pred_file, points_camera, pred_colors)
    write_ply(output_dir / error_file, points_camera, error)
    write_ply(output_dir / voted_file, voted_centers, voted_colors)

    sweep_rows = []
    for eps in eps_sweep:
        sweep_config = VotedCenterClusteringConfig(
            object_probability_threshold=clustering_config.object_probability_threshold,
            dbscan_eps_m=float(eps),
            dbscan_min_samples=clustering_config.dbscan_min_samples,
            min_cluster_points=clustering_config.min_cluster_points,
        )
        sweep_cluster = cluster_voted_centers(points_camera, object_probabilities, offsets, sweep_config)
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

    semantic_meter = SegmentationMeter(num_classes)
    semantic_meter.update(predicted_semantic, target_semantic)
    sample_summary = {
        "raw_sample": raw_sample,
        "source_path": item["path"],
        "gt_instances": gt_file,
        "pred_instances": pred_file,
        "error": error_file,
        "voted_centers": voted_file,
        "point_count": int(points_camera.shape[0]),
        "object_probability_min": float(object_probabilities.min()),
        "object_probability_mean": float(object_probabilities.mean()),
        "object_probability_max": float(object_probabilities.max()),
        "predicted_object_points": int(np.count_nonzero(predicted_semantic == 1)),
        "gt_object_points": int(np.count_nonzero(target_semantic == 1)),
        "dbscan_eps_m": clustering_config.dbscan_eps_m,
        "semantic_metrics": semantic_meter.compute(),
        "instance_metrics": instance_metrics,
        "dbscan_eps_sweep": sweep_rows,
    }
    return sample_summary, predicted_semantic, target_semantic


def mean_sample_metrics(samples: list[dict[str, Any]]) -> dict[str, float]:
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
    result: dict[str, float] = {}
    for key in keys:
        result[key] = float(np.mean([float(sample["instance_metrics"].get(key, 0.0)) for sample in samples]))
    return result


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
        augment=PointCloudAugmentationConfig(enabled=False),
        seed=seed,
    )
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    dataset.files = dataset.files[: args.max_samples]

    clustering_config = build_clustering_config(config, args)
    metric_config = InstanceMetricConfig.from_mapping(config.get("metrics", {}))
    eps_sweep = parse_eps_sweep(args.dbscan_eps_sweep_m, config, clustering_config.dbscan_eps_m)
    output_dir = Path(args.out) if args.out else default_output_dir(checkpoint_path, args.split)
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_semantic = SegmentationMeter(num_classes)
    samples: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        item = dataset[index]
        sample_summary, predicted_semantic, target_semantic = preview_sample(
            model,
            item,
            device,
            output_dir,
            clustering_config,
            metric_config,
            eps_sweep,
            num_classes,
        )
        samples.append(sample_summary)
        overall_semantic.update(predicted_semantic, target_semantic)
        instance_metrics = sample_summary["instance_metrics"]
        print(
            f"[preview] {sample_summary['raw_sample']}: "
            f"pred_clusters={instance_metrics['cluster_count_pred']} "
            f"gt_clusters={instance_metrics['cluster_count_gt']} "
            f"instance_miou={instance_metrics['instance_mean_iou']:.4f} "
            f"instance_recall={instance_metrics['instance_recall']:.4f} "
            f"semantic_object_iou={sample_summary['semantic_metrics'].get('object_iou', 0.0):.4f}"
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
        "mean_instance_metrics": mean_sample_metrics(samples),
        "samples": samples,
        "legend": {
            "background": COLOR_BACKGROUND.tolist(),
            "true_positive": COLOR_TRUE_POSITIVE.tolist(),
            "false_positive": COLOR_FALSE_POSITIVE.tolist(),
            "false_negative": COLOR_FALSE_NEGATIVE.tolist(),
            "mismatched_instance": COLOR_MISMATCH.tolist(),
            "noise": COLOR_NOISE.tolist(),
        },
    }
    (output_dir / "preview_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[preview] wrote previews: {output_dir}")
    print(f"[preview] mean instance metrics: {summary['mean_instance_metrics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
