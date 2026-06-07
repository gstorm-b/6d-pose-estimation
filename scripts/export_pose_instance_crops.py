"""Export pose-stage instance crops from GT or predicted instance labels."""

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
from src.inference.instance_clustering import VotedCenterClusteringConfig, cluster_voted_centers  # noqa: E402
from src.inference.pose_bridge import (  # noqa: E402
    build_pose_instance_crops,
    load_raw_metadata,
    pose_crop_summary,
    save_pose_instance_crops_npz,
)
from src.models.pointnet2_instance_seg import build_pointnet2_instance_seg_from_config  # noqa: E402
from src.training.checkpoint import load_checkpoint  # noqa: E402
from src.training.config import load_config  # noqa: E402
from src.training.seed import seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Processed PointNet++ dataset root.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--out", required=True, help="Output folder for crop NPZ files and manifest.json.")
    parser.add_argument("--source", choices=("gt", "predicted"), default="predicted")
    parser.add_argument("--checkpoint", help="Required for --source predicted.")
    parser.add_argument("--config", help="Config path for --source gt when no checkpoint is provided.")
    parser.add_argument("--raw-root", help="Raw synthetic dataset root used to load metadata.json pose labels.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ops-backend", default="pytorch3d", help="Override PointNet++ ops backend for predicted crops.")
    parser.add_argument("--normalize", help="Override dataset normalization mode.")
    parser.add_argument("--seed", type=int, help="Override RNG seed.")
    parser.add_argument("--limit-samples", type=int, help="Export only the first N samples.")
    parser.add_argument("--min-crop-points", type=int, help="Minimum points for exported pose crops.")
    parser.add_argument("--object-probability-threshold", type=float, help="Override predicted object threshold.")
    parser.add_argument("--dbscan-eps-m", type=float, help="Override predicted DBSCAN eps in meters.")
    parser.add_argument("--dbscan-min-samples", type=int, help="Override predicted DBSCAN min_samples.")
    parser.add_argument("--min-cluster-points", type=int, help="Override predicted minimum accepted cluster points.")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.checkpoint:
        checkpoint = load_checkpoint(Path(args.checkpoint), map_location="cpu")
        config = checkpoint.get("config", {})
        config["_checkpoint"] = checkpoint
        return config
    if args.source == "predicted":
        raise ValueError("--checkpoint is required when --source predicted")
    if args.config:
        return load_config(args.config)
    return {
        "seed": 7,
        "dataset": {"normalize": "scene_center", "min_points_per_instance": 16},
        "model": {"input_features": "xyz_normal", "num_classes": 2},
        "inference": {},
    }


def build_dataset(config: dict[str, Any], args: argparse.Namespace, seed: int) -> PointNet2InstanceSegDataset:
    dataset_config = config.get("dataset", {})
    use_normals = config.get("model", {}).get("input_features", "xyz_normal") == "xyz_normal"
    normalize = args.normalize or dataset_config.get("normalize", "scene_center")
    min_points_per_instance = int(config.get("loss", {}).get("min_points_per_instance", dataset_config.get("min_points_per_instance", 16)))
    dataset = PointNet2InstanceSegDataset(
        args.data,
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
        raise ValueError(f"No samples found in {Path(args.data) / args.split}")
    return dataset


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


def build_model(config: dict[str, Any], args: argparse.Namespace) -> Any:
    if args.source != "predicted":
        return None
    checkpoint = config.get("_checkpoint")
    if checkpoint is None:
        raise ValueError("checkpoint payload is missing")
    if args.ops_backend:
        config.setdefault("model", {})["ops_backend"] = args.ops_backend

    import torch

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = build_pointnet2_instance_seg_from_config(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def predicted_instance_labels(
    model: Any,
    item: dict[str, Any],
    device: str,
    clustering_config: VotedCenterClusteringConfig,
    num_classes: int,
) -> np.ndarray:
    import torch

    device_effective = device
    if device_effective == "cuda" and not torch.cuda.is_available():
        device_effective = "cpu"
    inputs = item["model_input"].unsqueeze(0).to(device_effective)
    with torch.no_grad():
        outputs = model(inputs)
        probabilities = torch.softmax(outputs["semantic_logits"], dim=-1).squeeze(0).cpu().numpy()
        offsets = outputs["offsets"].squeeze(0).cpu().numpy().astype(np.float32)
    object_probabilities = probabilities[:, 1] if num_classes > 1 else np.zeros(offsets.shape[0], dtype=np.float32)
    cluster_result = cluster_voted_centers(
        item["points_camera"].numpy(),
        object_probabilities,
        offsets,
        clustering_config,
    )
    return np.asarray(cluster_result["instance_labels"], dtype=np.int64)


def matched_iou_stats(sample_summaries: list[dict[str, Any]]) -> dict[str, float]:
    values = [
        float(crop["matched_gt_iou"])
        for sample in sample_summaries
        for crop in sample["crop_summaries"]
        if crop.get("matched_gt_iou") is not None
    ]
    if not values:
        return {
            "count": 0.0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "at_least_0_5": 0.0,
            "at_least_0_75": 0.0,
        }
    array = np.asarray(values, dtype=np.float32)
    return {
        "count": float(array.shape[0]),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
        "at_least_0_5": float(np.count_nonzero(array >= 0.5)),
        "at_least_0_75": float(np.count_nonzero(array >= 0.75)),
    }


def main() -> int:
    args = parse_args()
    config = config_from_args(args)
    seed = int(args.seed if args.seed is not None else config.get("seed", config.get("train", {}).get("seed", 7)))
    seed_everything(seed)
    dataset = build_dataset(config, args, seed)
    model = build_model(config, args)
    num_classes = int(config.get("model", {}).get("num_classes", 2))
    clustering_config = build_clustering_config(config, args)
    min_crop_points = int(
        args.min_crop_points
        if args.min_crop_points is not None
        else clustering_config.min_cluster_points
        if args.source == "predicted"
        else config.get("metrics", {}).get("ignore_gt_instances_below_points", 32)
    )

    output_dir = Path(args.out)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    sample_summaries: list[dict[str, Any]] = []
    total_crops = 0
    crops_with_pose = 0
    for index in range(len(dataset)):
        item = dataset[index]
        sample_name = str(item["raw_sample"])
        points_camera = item["points_camera"].numpy()
        point_features = item["features"].numpy() if item.get("features") is not None else None
        target_labels = item["instance_labels"].numpy().astype(np.int64)
        if args.source == "gt":
            crop_labels = target_labels
        else:
            crop_labels = predicted_instance_labels(model, item, args.device, clustering_config, num_classes)

        metadata = load_raw_metadata(args.raw_root, sample_name)
        crops = build_pose_instance_crops(
            sample_name=sample_name,
            source=args.source,
            points_camera=points_camera,
            point_features=point_features,
            instance_labels=crop_labels,
            target_instance_labels=target_labels,
            metadata=metadata,
            min_points=min_crop_points,
        )
        crop_path = crops_dir / f"{sample_name}.npz"
        save_pose_instance_crops_npz(crop_path, crops)
        total_crops += len(crops)
        crops_with_pose += sum(1 for crop in crops if crop.object_to_camera is not None)
        sample_summaries.append(
            {
                "sample_name": sample_name,
                "source_path": item["path"],
                "crop_file": str(crop_path.relative_to(output_dir)),
                "crop_count": len(crops),
                "crops_with_object_to_camera": sum(1 for crop in crops if crop.object_to_camera is not None),
                "gt_instance_count": int(len([value for value in np.unique(target_labels) if int(value) > 0])),
                "crop_summaries": [pose_crop_summary(crop) for crop in crops],
            }
        )
        print(
            f"[pose-bridge] {sample_name}: crops={len(crops)} "
            f"gt_instances={sample_summaries[-1]['gt_instance_count']} "
            f"with_pose={sample_summaries[-1]['crops_with_object_to_camera']}"
        )

    manifest = {
        "data_root": str(args.data),
        "raw_root": str(args.raw_root) if args.raw_root else None,
        "split": args.split,
        "source": args.source,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "config": str(args.config) if args.config else None,
        "seed": seed,
        "sample_count": len(sample_summaries),
        "total_crops": total_crops,
        "crops_with_object_to_camera": crops_with_pose,
        "matched_gt_iou_stats": matched_iou_stats(sample_summaries),
        "min_crop_points": min_crop_points,
        "clustering": {
            "object_probability_threshold": clustering_config.object_probability_threshold,
            "dbscan_eps_m": clustering_config.dbscan_eps_m,
            "dbscan_min_samples": clustering_config.dbscan_min_samples,
            "min_cluster_points": clustering_config.min_cluster_points,
        }
        if args.source == "predicted"
        else None,
        "samples": sample_summaries,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[pose-bridge] wrote manifest: {output_dir / 'manifest.json'}")
    print(f"[pose-bridge] total_crops={total_crops} crops_with_object_to_camera={crops_with_pose}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
