"""Profile PointNet++ memory and speed on a processed dataset."""

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
from src.training.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train/pointnet2_semseg_k41144.yaml")
    parser.add_argument("--data", required=True, help="Processed dataset root.")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--sa-npoints", help="Override Set Abstraction npoints, e.g. 1024,256,64.")
    parser.add_argument("--sa-nsamples", help="Override Set Abstraction nsamples, e.g. 32,32,32.")
    parser.add_argument("--ops-backend", help="Override PointNet++ ops backend, e.g. pure_torch or pytorch3d.")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--output-json", help="Optional path to write profile result JSON.")
    return parser.parse_args()


def parse_int_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def apply_model_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(json.dumps(config))
    model_config = config.setdefault("model", {})
    sa_npoints = parse_int_list(args.sa_npoints)
    sa_nsamples = parse_int_list(args.sa_nsamples)
    if sa_npoints is not None:
        model_config["sa_npoints"] = sa_npoints
    if sa_nsamples is not None:
        model_config["sa_nsamples"] = sa_nsamples
    if args.ops_backend:
        model_config["ops_backend"] = args.ops_backend
    model_config["dropout"] = args.dropout
    return config


def dataset_point_count(data_root: Path, split: str) -> int:
    sample = next((data_root / split).glob("*.npz"))
    with np.load(sample) as arrays:
        return int(arrays["points"].shape[0])


def bytes_to_mib(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0)


def profile(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    config = apply_model_overrides(load_config(args.config), args)
    data_root = Path(args.data)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    dataset = PointNet2SemSegDataset(
        data_root,
        args.split,
        use_normals=config.get("model", {}).get("input_features", "xyz_normal") == "xyz_normal",
        normalize=config.get("dataset", {}).get("normalize", "scene_center"),
        augment=PointCloudAugmentationConfig(enabled=False),
        seed=int(config.get("seed", 7)),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    model = build_pointnet2_semseg_from_config(config).to(device)
    effective_ops_backend = getattr(model, "ops_backend_name", "unknown")
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        total_memory = torch.cuda.get_device_properties(0).total_memory
    else:
        total_memory = 0

    batch_times: list[float] = []
    losses: list[float] = []
    processed_batches = 0
    status = "ok"
    error = ""
    try:
        target_batches = args.warmup_batches + args.max_batches
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= target_batches:
                break
            start = time.perf_counter()
            inputs = batch["model_input"].to(device)
            labels = batch["semantic_labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits.transpose(1, 2), labels)
            loss.backward()
            optimizer.step()
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if batch_idx >= args.warmup_batches:
                batch_times.append(elapsed)
                losses.append(float(loss.detach().cpu()))
                processed_batches += 1
    except RuntimeError as exc:
        error = str(exc)
        status = "oom" if "out of memory" in error.lower() else "runtime_error"
        if device == "cuda":
            torch.cuda.empty_cache()

    max_allocated = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    max_reserved = torch.cuda.max_memory_reserved() if device == "cuda" else 0
    result = {
        "status": status,
        "error": error,
        "data": str(data_root),
        "split": args.split,
        "point_count": dataset_point_count(data_root, args.split),
        "batch_size": args.batch_size,
        "warmup_batches": args.warmup_batches,
        "processed_batches": processed_batches,
        "device": device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "gpu_total_mib": bytes_to_mib(total_memory),
        "max_allocated_mib": bytes_to_mib(max_allocated),
        "max_reserved_mib": bytes_to_mib(max_reserved),
        "seconds_per_batch": batch_times,
        "mean_seconds_per_batch": float(np.mean(batch_times)) if batch_times else None,
        "losses": losses,
        "model": {
            "sa_npoints": config.get("model", {}).get("sa_npoints"),
            "sa_nsamples": config.get("model", {}).get("sa_nsamples"),
            "input_features": config.get("model", {}).get("input_features"),
            "ops_backend": config.get("model", {}).get("ops_backend", "pure_torch"),
            "ops_backend_effective": effective_ops_backend,
        },
    }
    return result


def main() -> int:
    args = parse_args()
    result = profile(args)
    print(json.dumps(result, indent=2))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
