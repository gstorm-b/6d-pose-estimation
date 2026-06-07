"""Train PointNet++ pose estimation on exported instance crops."""

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
from src.training.checkpoint import save_checkpoint  # noqa: E402
from src.training.config import load_config  # noqa: E402
from src.training.pose_losses import build_pointnet2_pose_loss_from_config  # noqa: E402
from src.training.pose_metrics import PoseMetricMeter  # noqa: E402
from src.training.seed import seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True, help="Train pose crop export root.")
    parser.add_argument("--val-data", help="Validation pose crop export root. Defaults to --data.")
    parser.add_argument("--experiment-root", default="experiments")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--translation-weight", type=float)
    parser.add_argument("--rotation-weight", type=float)
    parser.add_argument("--model-points-weight", type=float)
    parser.add_argument("--limit-train-samples", type=int)
    parser.add_argument("--limit-val-samples", type=int)
    parser.add_argument("--min-matched-gt-iou", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--freeze-batchnorm", action="store_true", help="Keep BatchNorm layers in eval mode during training.")
    parser.add_argument("--resume", help="Resume model and optimizer state from a pose checkpoint.")
    parser.add_argument("--resume-new-experiment", action="store_true", help="Load --resume but write a new experiment directory.")
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


def experiment_name(config: dict[str, Any], data_root: str) -> str:
    class_name = config.get("object", {}).get("class_name")
    if class_name:
        return f"pointnet2_pose_{str(class_name).lower()}"
    return f"pointnet2_pose_{Path(data_root).name}"


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    train_config = config.get("train", {})
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
    if args.dropout is not None:
        config.setdefault("model", {})["dropout"] = args.dropout
    if args.translation_weight is not None:
        config.setdefault("loss", {})["translation_weight"] = args.translation_weight
    if args.rotation_weight is not None:
        config.setdefault("loss", {})["rotation_weight"] = args.rotation_weight
    if args.model_points_weight is not None:
        config.setdefault("loss", {})["model_points_weight"] = args.model_points_weight
    if args.freeze_batchnorm:
        train_config["freeze_batchnorm"] = True
    config["train"] = train_config
    return config


def limit_records(dataset: Any, limit: int | None) -> None:
    if limit is None:
        return
    if limit <= 0:
        raise ValueError("sample limits must be positive")
    dataset.records = dataset.records[:limit]
    if not dataset.records:
        raise ValueError(f"No records left after applying limit {limit}")


def build_loaders(config: dict[str, Any], args: argparse.Namespace):
    from torch.utils.data import DataLoader

    train_config = config.get("train", {})
    train_dataset = build_pose_crop_dataset_from_config(
        config,
        args.data,
        min_matched_gt_iou=args.min_matched_gt_iou,
        enable_augmentation=True,
    )
    val_dataset = build_pose_crop_dataset_from_config(config, args.val_data or args.data, min_matched_gt_iou=args.min_matched_gt_iou)
    limit_records(train_dataset, args.limit_train_samples)
    limit_records(val_dataset, args.limit_val_samples)
    batch_size = int(train_config.get("batch_size", 16))
    workers = int(train_config.get("num_workers", 0))
    drop_last = len(train_dataset) > 1 and len(train_dataset) % batch_size == 1
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=drop_last),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, drop_last=False),
        train_dataset,
        val_dataset,
    )


def move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    import torch

    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def update_loss_totals(totals: dict[str, float], losses: dict[str, Any], weight: int) -> None:
    for key, value in losses.items():
        if hasattr(value, "detach"):
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * weight


def averaged(totals: dict[str, float], weight: int) -> dict[str, float]:
    divisor = max(weight, 1)
    return {key: value / divisor for key, value in sorted(totals.items())}


def run_epoch(
    model: Any,
    loader: Any,
    criterion: Any,
    optimizer: Any,
    device: str,
    *,
    train: bool,
    freeze_batchnorm: bool = False,
) -> dict[str, float]:
    import torch

    model.train(train)
    if train and freeze_batchnorm:
        set_batchnorm_eval(model)
    meter = PoseMetricMeter()
    loss_totals: dict[str, float] = {}
    sample_count = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            outputs = model(batch["points_pose_input"], batch.get("crop_aux_features"))
            losses = criterion(outputs, batch)
            loss = losses["total_loss"]
            if train:
                loss.backward()
                optimizer.step()
        batch_size = int(batch["points_pose_input"].shape[0])
        sample_count += batch_size
        update_loss_totals(loss_totals, losses, batch_size)
        meter.update(outputs, batch)

    metrics = meter.compute()
    metrics.update(averaged(loss_totals, sample_count))
    metrics["loss"] = metrics.get("total_loss", 0.0)
    return metrics


def set_batchnorm_eval(module: Any) -> None:
    import torch

    for child in module.modules():
        if isinstance(child, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            child.eval()


def main() -> int:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    train_config = config.get("train", {})
    seed = int(config.get("seed", train_config.get("seed", 7)))
    seed_everything(seed)

    import torch

    device = str(train_config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    config.setdefault("runtime", {})["device_effective"] = device
    train_loader, val_loader, train_dataset, val_dataset = build_loaders(config, args)
    model = build_pointnet2_pose_from_config(config).to(device)
    config.setdefault("model", {})["ops_backend_effective"] = getattr(model, "ops_backend_name", "unknown")
    criterion = build_pointnet2_pose_loss_from_config(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.get("learning_rate", 0.001)),
        weight_decay=float(train_config.get("weight_decay", 0.0001)),
    )

    start_epoch = 1
    history: list[dict[str, Any]] = []
    if args.resume:
        resume_path = Path(args.resume)
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        if args.resume_new_experiment:
            experiment = make_experiment_dir(Path(args.experiment_root), experiment_name(config, args.data))
        else:
            experiment = resume_path.resolve().parents[1] if resume_path.parent.name == "checkpoints" else make_experiment_dir(Path(args.experiment_root), experiment_name(config, args.data))
        history_path = experiment / "metrics.json"
        if history_path.exists() and not args.resume_new_experiment:
            history = json.loads(history_path.read_text(encoding="utf-8"))
            history = [record for record in history if int(record.get("epoch", 0)) < start_epoch]
    else:
        experiment = make_experiment_dir(Path(args.experiment_root), experiment_name(config, args.data))
    config["runtime_args"] = vars(args)
    config["dataset_runtime"] = {
        "train_data": args.data,
        "val_data": args.val_data or args.data,
        "train_records": len(train_dataset),
        "val_records": len(val_dataset),
        "model_diameter_m": train_dataset.geometry.model_diameter_m,
        "symmetry_name": train_dataset.geometry.symmetry_name,
    }
    (experiment / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    epochs = int(train_config.get("epochs", 100))
    best_val_loss = float("inf")
    best_val_add = float("inf")
    for record in history:
        best_val_loss = min(best_val_loss, float(record.get("val", {}).get("total_loss", float("inf"))))
        best_val_add = min(best_val_add, float(record.get("val", {}).get("add_mm", float("inf"))))
    for epoch in range(start_epoch, epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True,
            freeze_batchnorm=bool(train_config.get("freeze_batchnorm", False)),
        )
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(
            f"[pose-train] epoch={epoch} "
            f"train_total={train_metrics['total_loss']:.4f} val_total={val_metrics['total_loss']:.4f} "
            f"val_trans_mm={val_metrics['translation_error_mm']:.3f} "
            f"val_add_mm={val_metrics['add_mm']:.3f} "
            f"val_add_01d={val_metrics['pose_success_add_0.1d']:.3f}"
        )

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
            "metrics": record,
        }
        save_checkpoint(experiment / "checkpoints" / "last.pt", state)
        if float(val_metrics.get("total_loss", float("inf"))) < best_val_loss:
            best_val_loss = float(val_metrics["total_loss"])
            save_checkpoint(experiment / "checkpoints" / "best.pt", state)
        if float(val_metrics.get("add_mm", float("inf"))) < best_val_add:
            best_val_add = float(val_metrics["add_mm"])
            save_checkpoint(experiment / "checkpoints" / "best_add.pt", state)
        (experiment / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"[pose-train] wrote experiment: {experiment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
