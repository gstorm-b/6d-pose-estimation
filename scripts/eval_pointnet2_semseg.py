"""Evaluate a PointNet++ semantic segmentation checkpoint on a processed split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augmentation import PointCloudAugmentationConfig  # noqa: E402
from src.data.pointnet2_dataset import PointNet2SemSegDataset  # noqa: E402
from src.models.pointnet2_semseg import build_pointnet2_semseg_from_config  # noqa: E402
from src.training.checkpoint import load_checkpoint  # noqa: E402
from src.training.metrics import SegmentationMeter  # noqa: E402
from src.training.seed import seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a PointNet++ checkpoint.")
    parser.add_argument("--data", help="Processed dataset root. Defaults to checkpoint runtime/config data.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--out", help="Output JSON path. Defaults to <experiment>/<split>_metrics.json.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--normalize", help="Override dataset normalization mode.")
    parser.add_argument("--ops-backend", default="pytorch3d", help="Override PointNet++ ops backend for eval.")
    parser.add_argument("--seed", type=int, help="Override evaluation RNG seed. Defaults to checkpoint config seed.")
    return parser.parse_args()


def resolve_data_root(config: dict[str, Any], cli_value: str | None) -> Path:
    if cli_value:
        return Path(cli_value)
    runtime_data = config.get("runtime_args", {}).get("data")
    if runtime_data:
        return Path(runtime_data)
    return Path(config.get("dataset", {}).get("root", "processed-data/pointnet2_semseg_k41144"))


def default_output_path(checkpoint_path: Path, split: str) -> Path:
    experiment = checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent
    return experiment / f"{split}_metrics.json"


def config_with_eval_overrides(config: dict[str, Any], ops_backend: str | None) -> dict[str, Any]:
    config = json.loads(json.dumps(config))
    if ops_backend:
        config.setdefault("model", {})["ops_backend"] = ops_backend
    return config


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    checkpoint_config = checkpoint.get("config", {})
    config = config_with_eval_overrides(checkpoint_config, args.ops_backend)
    model_config = config.get("model", {})
    dataset_config = config.get("dataset", {})
    num_classes = int(model_config.get("num_classes", 2))
    seed = int(args.seed if args.seed is not None else config.get("seed", config.get("train", {}).get("seed", 7)))
    seed_everything(seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = build_pointnet2_semseg_from_config(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    data_root = resolve_data_root(config, args.data)
    use_normals = model_config.get("input_features", "xyz_normal") == "xyz_normal"
    normalize = args.normalize or dataset_config.get("normalize", "scene_center")
    dataset = PointNet2SemSegDataset(
        data_root,
        args.split,
        use_normals=use_normals,
        normalize=normalize,
        augment=PointCloudAugmentationConfig(enabled=False),
        seed=int(config.get("seed", 7)),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    meter = SegmentationMeter(num_classes)
    sample_results: list[dict[str, Any]] = []
    total_loss = 0.0
    total_points = 0
    criterion = torch.nn.CrossEntropyLoss()
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            inputs = batch["model_input"].to(device)
            labels = batch["semantic_labels"].to(device)
            logits = model(inputs)
            loss = criterion(logits.transpose(1, 2), labels)
            predictions = logits.argmax(dim=-1)

            point_count = int(labels.numel())
            total_loss += float(loss.detach().cpu()) * point_count
            total_points += point_count
            meter.update(predictions.detach().cpu().numpy(), labels.detach().cpu().numpy())

            batch_predictions = predictions.detach().cpu().numpy()
            batch_labels = labels.detach().cpu().numpy()
            for index, raw_sample in enumerate(batch["raw_sample"]):
                sample_meter = SegmentationMeter(num_classes)
                sample_meter.update(batch_predictions[index], batch_labels[index])
                sample_results.append(
                    {
                        "raw_sample": str(raw_sample),
                        "path": str(batch["path"][index]),
                        "metrics": sample_meter.compute(),
                    }
                )

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    overall_metrics = meter.compute()
    overall_metrics["loss"] = total_loss / max(total_points, 1)
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "data_root": str(data_root),
        "split": args.split,
        "sample_count": len(dataset),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "ops_backend_requested": args.ops_backend,
        "ops_backend_effective": getattr(model, "ops_backend_name", "unknown"),
        "seed": seed,
        "input_features": model_config.get("input_features", "xyz_normal"),
        "normalize": normalize,
        "elapsed_seconds": elapsed,
        "samples_per_second": len(dataset) / elapsed if elapsed > 0 else None,
        "points_per_second": total_points / elapsed if elapsed > 0 else None,
        "overall_metrics": overall_metrics,
        "samples": sample_results,
    }
    return result


def main() -> int:
    args = parse_args()
    result = evaluate(args)
    output_path = Path(args.out) if args.out else default_output_path(Path(args.checkpoint), args.split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    metrics = result["overall_metrics"]
    print(
        f"[eval] split={result['split']} samples={result['sample_count']} "
        f"loss={metrics['loss']:.4f} miou={metrics['mean_iou']:.4f} "
        f"object_iou={metrics.get('object_iou', 0.0):.4f} "
        f"object_recall={metrics.get('object_recall', 0.0):.4f} "
        f"ops_backend={result['ops_backend_effective']}"
    )
    print(f"[eval] wrote metrics: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
