"""Train the PointNet++ semantic segmentation MVP."""

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
    parser.add_argument("--config", default="configs/train/pointnet2_semseg_k41144.yaml")
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
    import torch
    from torch.utils.data import DataLoader

    from src.data.augmentation import PointCloudAugmentationConfig
    from src.data.pointnet2_dataset import PointNet2SemSegDataset

    dataset_config = config.get("dataset", {})
    train_config = config.get("train", {})
    root = Path(args.data or dataset_config.get("root", "processed-data/pointnet2_semseg_k41144"))
    use_normals = config.get("model", {}).get("input_features", "xyz_normal") == "xyz_normal"
    normalize = dataset_config.get("normalize", "scene_center")
    seed = int(config.get("seed", 7))
    train_augment = PointCloudAugmentationConfig(enabled=False) if args.disable_augment or args.overfit else PointCloudAugmentationConfig.from_mapping(dataset_config.get("augment", {}))

    train_dataset = PointNet2SemSegDataset(
        root,
        "train",
        use_normals=use_normals,
        normalize=normalize,
        augment=train_augment,
        seed=seed,
    )
    val_split = "train" if args.overfit else "val"
    val_dataset = PointNet2SemSegDataset(
        root,
        val_split,
        use_normals=use_normals,
        normalize=normalize,
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

    batch_size = int(args.batch_size or train_config.get("batch_size", 4))
    workers = int(args.num_workers if args.num_workers is not None else train_config.get("num_workers", 0))
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=False),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, drop_last=False),
        train_dataset,
        torch,
    )


def run_epoch(model, loader, criterion, optimizer, device: str, num_classes: int, train: bool) -> dict[str, Any]:
    import torch

    model.train(train)
    meter = SegmentationMeter(num_classes)
    total_loss = 0.0
    total_points = 0
    for batch in loader:
        inputs = batch["model_input"].to(device)
        labels = batch["semantic_labels"].to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(inputs)
            loss = criterion(logits.transpose(1, 2), labels)
            if train:
                loss.backward()
                optimizer.step()
        points = int(labels.numel())
        total_loss += float(loss.detach().cpu()) * points
        total_points += points
        predictions = logits.argmax(dim=-1).detach().cpu().numpy()
        meter.update(predictions, labels.detach().cpu().numpy())

    metrics = meter.compute()
    metrics["loss"] = total_loss / max(total_points, 1)
    return metrics


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    train_config = config.get("train", {})
    if args.dropout is not None:
        config.setdefault("model", {})["dropout"] = args.dropout
    if args.learning_rate is not None:
        train_config["learning_rate"] = args.learning_rate
        config["train"] = train_config
    seed = int(config.get("seed", train_config.get("seed", 7)))
    seed_everything(seed)

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for training. Install requirements-train.txt first.") from exc

    from src.models.pointnet2_semseg import build_pointnet2_semseg_from_config
    from src.training.checkpoint import save_checkpoint

    device = args.device or train_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    train_loader, val_loader, train_dataset, _torch = build_loaders(config, args)
    model = build_pointnet2_semseg_from_config(config).to(device)
    config.setdefault("model", {})["ops_backend_effective"] = getattr(model, "ops_backend_name", "unknown")
    num_classes = int(config.get("model", {}).get("num_classes", 2))
    if train_config.get("class_weight_mode", "auto") == "auto":
        weights = torch.from_numpy(class_weights_from_dataset(train_dataset, num_classes)).to(device)
    else:
        weights = None
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.get("learning_rate", 0.001)),
        weight_decay=float(train_config.get("weight_decay", 0.0001)),
    )

    experiment = make_experiment_dir(Path(args.experiment_root), "pointnet2_semseg_k41144")
    config["runtime_args"] = {
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
    }
    (experiment / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    epochs = int(args.epochs or train_config.get("epochs", 100))
    history: list[dict[str, Any]] = []
    best_iou = -1.0
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, num_classes, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, num_classes, train=False)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(
            f"[train] epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_miou={val_metrics['mean_iou']:.4f} val_object_recall={val_metrics.get('object_recall', 0.0):.4f}"
        )

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
            "metrics": record,
        }
        save_checkpoint(experiment / "checkpoints" / "last.pt", state)
        if float(val_metrics["mean_iou"]) > best_iou:
            best_iou = float(val_metrics["mean_iou"])
            save_checkpoint(experiment / "checkpoints" / "best.pt", state)
        (experiment / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"[train] wrote experiment: {experiment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
