"""Train a learned translation refiner on top of a frozen PointNet++ pose checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
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
from src.training.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from src.training.config import load_config  # noqa: E402
from src.training.pose_metrics import (  # noqa: E402
    CenterVoteAggregationConfig,
    compute_pose_metrics_per_sample,
    compute_pose_metrics_per_sample_from_pose,
    pose_predictions_to_object_to_camera,
)
from src.training.pose_refiner_losses import build_pose_translation_refiner_loss_from_config  # noqa: E402
from src.training.seed import seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Refiner config.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--val-data")
    parser.add_argument("--pose-config", help="Override base pose config.")
    parser.add_argument("--pose-checkpoint", help="Override frozen base pose checkpoint.")
    parser.add_argument("--experiment-root", default="experiments")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--limit-train-samples", type=int)
    parser.add_argument("--limit-val-samples", type=int)
    parser.add_argument("--min-matched-gt-iou", type=float)
    parser.add_argument("--center-vote-aggregation", default="mean", choices=("mean", "median", "trimmed_mean", "weighted_trimmed_mean", "geometric_median"))
    parser.add_argument("--center-vote-trim-fraction", type=float, default=0.1)
    return parser.parse_args()


def make_experiment_dir(root: Path, name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = root / f"{name}_{stamp}"
    suffix = 1
    while path.exists():
        suffix += 1
        path = root / f"{name}_{stamp}_{suffix}"
    path.mkdir(parents=True)
    (path / "checkpoints").mkdir()
    return path


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    train_config = config.setdefault("train", {})
    if args.epochs is not None:
        train_config["epochs"] = args.epochs
    if args.batch_size is not None:
        train_config["batch_size"] = args.batch_size
    if args.device is not None:
        train_config["device"] = args.device
    if args.num_workers is not None:
        train_config["num_workers"] = args.num_workers
    if args.learning_rate is not None:
        train_config["learning_rate"] = args.learning_rate
    if args.pose_config is not None:
        config.setdefault("pose", {})["config"] = args.pose_config
    if args.pose_checkpoint is not None:
        config.setdefault("pose", {})["checkpoint"] = args.pose_checkpoint
    return config


def limit_records(dataset: Any, limit: int | None) -> None:
    if limit is None:
        return
    if limit <= 0:
        raise ValueError("sample limits must be positive")
    dataset.records = dataset.records[:limit]
    if not dataset.records:
        raise ValueError(f"No records left after applying limit {limit}")


def move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    import torch

    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def build_loaders(pose_config: dict[str, Any], refiner_config: dict[str, Any], args: argparse.Namespace):
    from torch.utils.data import DataLoader

    train_config = refiner_config.get("train", {})
    train_dataset = build_pose_crop_dataset_from_config(
        pose_config,
        args.data,
        min_matched_gt_iou=args.min_matched_gt_iou,
        enable_augmentation=True,
    )
    val_dataset = build_pose_crop_dataset_from_config(
        pose_config,
        args.val_data or args.data,
        min_matched_gt_iou=args.min_matched_gt_iou,
    )
    limit_records(train_dataset, args.limit_train_samples)
    limit_records(val_dataset, args.limit_val_samples)
    batch_size = int(train_config.get("batch_size", 16))
    workers = int(train_config.get("num_workers", 0))
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=False),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, drop_last=False),
        train_dataset,
        val_dataset,
    )


def update_tensor_totals(totals: dict[str, float], metrics: dict[str, Any]) -> int:
    first = next(iter(metrics.values()))
    batch_size = int(first.shape[0])
    for key, value in metrics.items():
        totals[key] = totals.get(key, 0.0) + float(value.detach().cpu().sum())
    return batch_size


def update_loss_totals(totals: dict[str, float], losses: dict[str, Any], weight: int) -> None:
    for key, value in losses.items():
        if hasattr(value, "detach"):
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * weight


def averaged(totals: dict[str, float], weight: int) -> dict[str, float]:
    divisor = max(weight, 1)
    return {key: value / divisor for key, value in sorted(totals.items())}


def run_epoch(
    *,
    pose_model: Any,
    refiner: Any,
    loader: Any,
    criterion: Any,
    optimizer: Any,
    device: str,
    feature_config: PoseRefinerFeatureConfig,
    train: bool,
) -> dict[str, float]:
    import torch

    pose_model.eval()
    refiner.train(train)
    loss_totals: dict[str, float] = {}
    raw_totals: dict[str, float] = {}
    refined_totals: dict[str, float] = {}
    sample_count = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            pose_outputs = pose_model(batch["points_pose_input"], batch.get("crop_aux_features"))
            raw_pose = pose_predictions_to_object_to_camera(
                pose_outputs,
                batch,
                center_vote_config=feature_config.center_vote_config,
            )
            features = build_pose_refiner_features(
                pose_outputs=pose_outputs,
                batch=batch,
                raw_pose=raw_pose,
                feature_config=feature_config,
            ).detach()
            raw_metrics = compute_pose_metrics_per_sample(
                pose_outputs,
                batch,
                center_vote_config=feature_config.center_vote_config,
            )
        with torch.set_grad_enabled(train):
            refiner_outputs = refiner(features)
            losses = criterion(
                refiner_outputs,
                batch,
                raw_rotation_object_to_camera=raw_pose["rotation_object_to_camera"].detach(),
                raw_origin_camera=raw_pose["object_origin_camera"].detach(),
            )
            if train:
                losses["total_loss"].backward()
                optimizer.step()
        diameter = batch["model_diameter_m"].to(device=device, dtype=features.dtype).view(-1, 1)
        refined_origin = raw_pose["object_origin_camera"].detach() + refiner_outputs["origin_delta_camera_normalized"].detach() * diameter
        refined_metrics = compute_pose_metrics_per_sample_from_pose(
            rotation_pred_object_to_camera=raw_pose["rotation_object_to_camera"].detach(),
            origin_pred_camera=refined_origin,
            batch=batch,
        )
        batch_size = int(features.shape[0])
        sample_count += batch_size
        update_loss_totals(loss_totals, losses, batch_size)
        update_tensor_totals(raw_totals, raw_metrics)
        update_tensor_totals(refined_totals, refined_metrics)

    metrics = {"raw_" + key: value for key, value in averaged(raw_totals, sample_count).items()}
    metrics.update({"refined_" + key: value for key, value in averaged(refined_totals, sample_count).items()})
    metrics.update(averaged(loss_totals, sample_count))
    metrics["loss"] = metrics.get("total_loss", 0.0)
    metrics["delta_add_mm"] = metrics.get("refined_add_mm", 0.0) - metrics.get("raw_add_mm", 0.0)
    metrics["delta_translation_mm"] = metrics.get("refined_translation_error_mm", 0.0) - metrics.get("raw_translation_error_mm", 0.0)
    return metrics


def main() -> int:
    args = parse_args()
    refiner_config = apply_overrides(load_config(args.config), args)
    pose_config_path = refiner_config.get("pose", {}).get("config")
    pose_checkpoint_path = refiner_config.get("pose", {}).get("checkpoint")
    if not pose_config_path or not pose_checkpoint_path:
        raise ValueError("refiner config must provide pose.config and pose.checkpoint, or pass overrides")
    pose_config = load_config(pose_config_path)
    train_config = refiner_config.get("train", {})
    seed = int(refiner_config.get("seed", train_config.get("seed", 7)))
    seed_everything(seed)

    import torch

    device = str(train_config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    pose_checkpoint = load_checkpoint(pose_checkpoint_path, map_location="cpu")
    pose_model_config = pose_checkpoint.get("config", pose_config)
    pose_model = build_pointnet2_pose_from_config(pose_model_config)
    pose_model.load_state_dict(pose_checkpoint["model_state"])
    pose_model.to(device)
    pose_model.eval()
    for parameter in pose_model.parameters():
        parameter.requires_grad_(False)

    train_loader, val_loader, train_dataset, val_dataset = build_loaders(pose_config, refiner_config, args)
    refiner = build_pointnet2_pose_refiner_from_config(refiner_config).to(device)
    criterion = build_pose_translation_refiner_loss_from_config(refiner_config).to(device)
    optimizer = torch.optim.AdamW(
        refiner.parameters(),
        lr=float(train_config.get("learning_rate", 0.001)),
        weight_decay=float(train_config.get("weight_decay", 0.0001)),
    )
    feature_config = PoseRefinerFeatureConfig(
        center_vote_config=CenterVoteAggregationConfig(
            method=args.center_vote_aggregation,
            trim_fraction=args.center_vote_trim_fraction,
        )
    )
    class_name = pose_config.get("object", {}).get("class_name", "pose")
    experiment = make_experiment_dir(Path(args.experiment_root), f"pointnet2_pose_refiner_{str(class_name).lower()}")
    refiner_config["runtime_args"] = vars(args)
    refiner_config["dataset_runtime"] = {
        "train_data": args.data,
        "val_data": args.val_data or args.data,
        "train_records": len(train_dataset),
        "val_records": len(val_dataset),
        "model_diameter_m": train_dataset.geometry.model_diameter_m,
        "symmetry_name": train_dataset.geometry.symmetry_name,
        "pose_checkpoint": str(pose_checkpoint_path),
    }
    refiner_config.setdefault("refiner", {})["input_dim"] = int(refiner.input_dim)
    (experiment / "config.json").write_text(json.dumps(refiner_config, indent=2), encoding="utf-8")

    best_val_add = float("inf")
    history: list[dict[str, Any]] = []
    epochs = int(train_config.get("epochs", 40))
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            pose_model=pose_model,
            refiner=refiner,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            feature_config=feature_config,
            train=True,
        )
        val_metrics = run_epoch(
            pose_model=pose_model,
            refiner=refiner,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            feature_config=feature_config,
            train=False,
        )
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(
            f"[pose-refiner-train] epoch={epoch} "
            f"train_total={train_metrics['total_loss']:.4f} val_total={val_metrics['total_loss']:.4f} "
            f"raw_val_add={val_metrics['raw_add_mm']:.3f} refined_val_add={val_metrics['refined_add_mm']:.3f} "
            f"delta_trans={val_metrics['delta_translation_mm']:.3f}"
        )
        state = {
            "epoch": epoch,
            "model_state": refiner.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": refiner_config,
            "pose_config": pose_model_config,
            "pose_checkpoint": str(pose_checkpoint_path),
            "metrics": record,
        }
        save_checkpoint(experiment / "checkpoints" / "last.pt", state)
        if float(val_metrics.get("refined_add_mm", float("inf"))) < best_val_add:
            best_val_add = float(val_metrics["refined_add_mm"])
            save_checkpoint(experiment / "checkpoints" / "best_add.pt", state)
        (experiment / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"[pose-refiner-train] wrote experiment: {experiment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
