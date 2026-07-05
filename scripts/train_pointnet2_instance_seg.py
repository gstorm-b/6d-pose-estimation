"""Train the PointNet++ instance segmentation branch."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.config import load_config  # noqa: E402
from src.training.metrics import SegmentationMeter  # noqa: E402
from src.training.seed import seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train/pointnet2_instance_k41144.yaml")
    parser.add_argument("--data", help="Override processed dataset root.")
    parser.add_argument("--experiment-root", default="experiments")
    parser.add_argument("--epochs", type=int, help="Override epoch count.")
    parser.add_argument("--batch-size", type=int, help="Override batch size.")
    parser.add_argument("--device", help="Override device, e.g. cuda or cpu.")
    parser.add_argument("--num-workers", type=int, help="Override DataLoader workers.")
    parser.add_argument("--limit-train-samples", type=int, help="Use only the first N train samples.")
    parser.add_argument("--limit-val-samples", type=int, help="Use only the first N validation samples.")
    parser.add_argument("--disable-augment", action="store_true", help="Disable training augmentation.")
    parser.add_argument("--overfit", action="store_true", help="Use train samples for validation and disable augmentation.")
    parser.add_argument("--dropout", type=float, help="Override model dropout.")
    parser.add_argument("--learning-rate", type=float, help="Override optimizer learning rate.")
    parser.add_argument("--semantic-weight", type=float, help="Override semantic loss weight.")
    parser.add_argument("--offset-weight", type=float, help="Override centroid-offset loss weight.")
    parser.add_argument("--embedding-weight", type=float, help="Override discriminative embedding loss weight.")
    parser.add_argument("--centroid-distance-weighting", action="store_true", help="Enable center-distance-sensitive offset loss weighting.")
    parser.add_argument("--init-checkpoint", help="Initialize model weights from this checkpoint (fine-tuning).")
    return parser.parse_args()


def make_experiment_dir(root: Path, name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = root / f"{name}_{stamp}"
    path = base
    suffix = 1
    while path.exists():
        suffix += 1
        path = root / f"{base.name}_{suffix}"
    path.mkdir(parents=True)
    (path / "checkpoints").mkdir()
    return path


def experiment_name_from_dataset(root: Path) -> str:
    dataset_name = root.name
    if dataset_name.startswith("pointnet2_semseg_"):
        dataset_name = dataset_name.removeprefix("pointnet2_semseg_")
    return f"pointnet2_instance_{dataset_name}"


def class_weights_from_dataset(dataset: Any, num_classes: int) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.float64)
    for path in dataset.files:
        with np.load(path) as data:
            labels = np.asarray(data["semantic_labels"], dtype=np.int64)
            counts += np.bincount(labels, minlength=num_classes)[:num_classes]
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def limit_dataset_files(dataset: Any, limit: int | None) -> None:
    if limit is None:
        return
    if limit <= 0:
        raise ValueError("sample limits must be positive")
    dataset.files = dataset.files[:limit]
    if not dataset.files:
        raise ValueError(f"No samples left after applying limit {limit}")


def build_loaders(config: dict[str, Any], args: argparse.Namespace):
    from torch.utils.data import DataLoader

    from src.data.augmentation import PointCloudAugmentationConfig
    from src.data.pointnet2_instance_dataset import PointNet2InstanceSegDataset

    dataset_config = config.get("dataset", {})
    train_config = config.get("train", {})
    loss_config = config.get("loss", {})
    root = Path(args.data or dataset_config.get("root", "processed-data/pointnet2_semseg_k41144"))
    use_normals = config.get("model", {}).get("input_features", "xyz_normal") == "xyz_normal"
    normalize = dataset_config.get("normalize", "scene_center")
    seed = int(config.get("seed", 7))
    min_points_per_instance = int(loss_config.get("min_points_per_instance", dataset_config.get("min_points_per_instance", 16)))
    train_augment = (
        PointCloudAugmentationConfig(enabled=False)
        if args.disable_augment or args.overfit
        else PointCloudAugmentationConfig.from_mapping(dataset_config.get("augment", {}))
    )

    train_dataset = PointNet2InstanceSegDataset(
        root,
        "train",
        use_normals=use_normals,
        normalize=normalize,
        min_points_per_instance=min_points_per_instance,
        max_instances=int(dataset_config.get("max_instances", 128)),
        centroid_distance_weight_min=float(loss_config.get("centroid_distance_weight_min", 0.5)),
        centroid_distance_weight_max=float(loss_config.get("centroid_distance_weight_max", 1.5)),
        augment=train_augment,
        seed=seed,
    )
    val_split = "train" if args.overfit else "val"
    val_dataset = PointNet2InstanceSegDataset(
        root,
        val_split,
        use_normals=use_normals,
        normalize=normalize,
        min_points_per_instance=min_points_per_instance,
        max_instances=int(dataset_config.get("max_instances", 128)),
        centroid_distance_weight_min=float(loss_config.get("centroid_distance_weight_min", 0.5)),
        centroid_distance_weight_max=float(loss_config.get("centroid_distance_weight_max", 1.5)),
        augment=PointCloudAugmentationConfig(enabled=False),
        seed=seed,
    )
    limit_dataset_files(train_dataset, args.limit_train_samples)
    val_limit = args.limit_val_samples
    if args.overfit and val_limit is None:
        val_limit = args.limit_train_samples
    limit_dataset_files(val_dataset, val_limit)
    if args.overfit:
        val_dataset.files = train_dataset.files[:]

    batch_size = int(args.batch_size or train_config.get("batch_size", 1))
    workers = int(args.num_workers if args.num_workers is not None else train_config.get("num_workers", 0))
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=False),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, drop_last=False),
        train_dataset,
        root,
    )


def move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    import torch

    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def update_loss_totals(totals: dict[str, float], losses: dict[str, Any], weight: int) -> None:
    for key, value in losses.items():
        if hasattr(value, "detach"):
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * weight


def averaged_loss_totals(totals: dict[str, float], weight_total: int) -> dict[str, float]:
    divisor = max(weight_total, 1)
    return {key: value / divisor for key, value in sorted(totals.items())}


def run_epoch(
    model: Any,
    loader: Any,
    criterion: Any,
    optimizer: Any,
    device: str,
    num_classes: int,
    train: bool,
) -> dict[str, Any]:
    import torch

    model.train(train)
    meter = SegmentationMeter(num_classes)
    loss_totals: dict[str, float] = {}
    sample_count = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        inputs = batch["model_input"]
        labels = batch["semantic_labels"]
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            outputs = model(inputs)
            losses = criterion(outputs, batch)
            loss = losses["total_loss"]
            if train:
                loss.backward()
                optimizer.step()

        batch_size = int(inputs.shape[0])
        sample_count += batch_size
        update_loss_totals(loss_totals, losses, batch_size)
        predictions = outputs["semantic_logits"].argmax(dim=-1).detach().cpu().numpy()
        meter.update(predictions, labels.detach().cpu().numpy())

    metrics = meter.compute()
    metrics.update(averaged_loss_totals(loss_totals, sample_count))
    metrics["loss"] = metrics.get("total_loss", 0.0)
    return metrics


def apply_runtime_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    train_config = config.get("train", {})
    if args.dropout is not None:
        config.setdefault("model", {})["dropout"] = args.dropout
    if args.learning_rate is not None:
        train_config["learning_rate"] = args.learning_rate
    if args.semantic_weight is not None:
        config.setdefault("loss", {})["semantic_weight"] = args.semantic_weight
    if args.offset_weight is not None:
        config.setdefault("loss", {})["offset_weight"] = args.offset_weight
    if args.embedding_weight is not None:
        config.setdefault("loss", {})["embedding_weight"] = args.embedding_weight
    if args.centroid_distance_weighting:
        config.setdefault("loss", {})["centroid_distance_weighting"] = True
    if args.batch_size is not None:
        train_config["batch_size"] = args.batch_size
    if args.epochs is not None:
        train_config["epochs"] = args.epochs
    if args.device is not None:
        train_config["device"] = args.device
    if args.num_workers is not None:
        train_config["num_workers"] = args.num_workers
    config["train"] = train_config
    return config


def main() -> int:
    args = parse_args()
    config = apply_runtime_overrides(load_config(args.config), args)
    train_config = config.get("train", {})
    seed = int(config.get("seed", train_config.get("seed", 7)))
    seed_everything(seed)

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for training. Install requirements-train.txt first.") from exc

    from src.models.pointnet2_instance_seg import build_pointnet2_instance_seg_from_config
    from src.training.checkpoint import save_checkpoint
    from src.training.instance_losses import build_pointnet2_instance_loss_from_config

    device = str(train_config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    config.setdefault("runtime", {})["device_effective"] = device

    train_loader, val_loader, train_dataset, dataset_root = build_loaders(config, args)
    model = build_pointnet2_instance_seg_from_config(config).to(device)
    if args.init_checkpoint:
        from src.training.checkpoint import load_checkpoint

        init_state = load_checkpoint(Path(args.init_checkpoint), map_location="cpu")
        model.load_state_dict(init_state["model_state"])
        config.setdefault("runtime", {})["init_checkpoint"] = str(args.init_checkpoint)
        print(f"[train] initialized weights from {args.init_checkpoint}")
    config.setdefault("model", {})["ops_backend_effective"] = getattr(model, "ops_backend_name", "unknown")
    num_classes = int(config.get("model", {}).get("num_classes", 2))
    if train_config.get("class_weight_mode", "auto") == "auto":
        weights = class_weights_from_dataset(train_dataset, num_classes)
        config.setdefault("loss", {})["semantic_class_weights"] = [float(value) for value in weights]
    criterion = build_pointnet2_instance_loss_from_config(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.get("learning_rate", 0.001)),
        weight_decay=float(train_config.get("weight_decay", 0.0001)),
    )

    experiment = make_experiment_dir(Path(args.experiment_root), experiment_name_from_dataset(dataset_root))
    config["runtime_args"] = {
        "config": args.config,
        "data": args.data,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": args.device,
        "num_workers": args.num_workers,
        "limit_train_samples": args.limit_train_samples,
        "limit_val_samples": args.limit_val_samples,
        "disable_augment": args.disable_augment,
        "overfit": args.overfit,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "semantic_weight": args.semantic_weight,
        "offset_weight": args.offset_weight,
        "embedding_weight": args.embedding_weight,
        "centroid_distance_weighting": args.centroid_distance_weighting,
    }
    config.setdefault("dataset", {})["root_effective"] = str(dataset_root)
    (experiment / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    epochs = int(train_config.get("epochs", 50))
    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, num_classes, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, num_classes, train=False)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(
            f"[train] epoch={epoch} "
            f"train_total={train_metrics['total_loss']:.4f} val_total={val_metrics['total_loss']:.4f} "
            f"val_object_iou={val_metrics.get('object_iou', 0.0):.4f} "
            f"val_offset_l1={val_metrics.get('offset_l1_object', 0.0):.4f}"
        )

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
            "metrics": record,
        }
        save_checkpoint(experiment / "checkpoints" / "last.pt", state)
        val_loss = float(val_metrics.get("total_loss", float("inf")))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(experiment / "checkpoints" / "best.pt", state)
        (experiment / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"[train] wrote experiment: {experiment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
