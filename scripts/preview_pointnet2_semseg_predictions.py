"""Export PointNet++ semantic predictions as colored PLY previews."""

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
from src.data.pointnet2_dataset import PointNet2SemSegDataset  # noqa: E402
from src.models.pointnet2_semseg import build_pointnet2_semseg_from_config  # noqa: E402
from src.training.checkpoint import load_checkpoint  # noqa: E402
from src.training.metrics import SegmentationMeter  # noqa: E402
from src.training.seed import seed_everything  # noqa: E402


COLOR_BACKGROUND = np.array([120, 120, 120], dtype=np.uint8)
COLOR_GT_OBJECT = np.array([40, 210, 90], dtype=np.uint8)
COLOR_PRED_OBJECT = np.array([60, 130, 255], dtype=np.uint8)
COLOR_TRUE_POSITIVE = np.array([40, 210, 90], dtype=np.uint8)
COLOR_FALSE_POSITIVE = np.array([235, 70, 60], dtype=np.uint8)
COLOR_FALSE_NEGATIVE = np.array([245, 205, 65], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a PointNet++ checkpoint.")
    parser.add_argument("--data", help="Processed dataset root. Defaults to checkpoint runtime/config data.")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--out", help="Output preview folder. Defaults to <experiment>/previews/<split>.")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--normalize", help="Override dataset normalization mode.")
    parser.add_argument("--ops-backend", default="pytorch3d", help="Override PointNet++ ops backend for preview.")
    parser.add_argument("--seed", type=int, help="Override preview RNG seed. Defaults to checkpoint config seed.")
    return parser.parse_args()


def resolve_data_root(config: dict[str, Any], cli_value: str | None) -> Path:
    if cli_value:
        return Path(cli_value)
    runtime_data = config.get("runtime_args", {}).get("data")
    if runtime_data:
        return Path(runtime_data)
    return Path(config.get("dataset", {}).get("root", "processed-data/pointnet2_semseg_k41144"))


def default_output_dir(checkpoint_path: Path, split: str) -> Path:
    experiment = checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent
    return experiment / "previews" / split


def labels_to_gt_colors(labels: np.ndarray) -> np.ndarray:
    colors = np.repeat(COLOR_BACKGROUND[None, :], labels.shape[0], axis=0)
    colors[labels == 1] = COLOR_GT_OBJECT
    return colors


def labels_to_prediction_colors(labels: np.ndarray) -> np.ndarray:
    colors = np.repeat(COLOR_BACKGROUND[None, :], labels.shape[0], axis=0)
    colors[labels == 1] = COLOR_PRED_OBJECT
    return colors


def labels_to_error_colors(targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    colors = np.repeat(COLOR_BACKGROUND[None, :], targets.shape[0], axis=0)
    true_positive = (targets == 1) & (predictions == 1)
    false_positive = (targets == 0) & (predictions == 1)
    false_negative = (targets == 1) & (predictions == 0)
    colors[true_positive] = COLOR_TRUE_POSITIVE
    colors[false_positive] = COLOR_FALSE_POSITIVE
    colors[false_negative] = COLOR_FALSE_NEGATIVE
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


def preview_sample(model: Any, item: dict[str, Any], device: str, output_dir: Path, num_classes: int) -> dict[str, Any]:
    import torch

    raw_sample = str(item["raw_sample"])
    points = item["points"].numpy()
    targets = item["semantic_labels"].numpy()
    inputs = item["model_input"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=-1)
        predictions = probabilities.argmax(dim=-1).squeeze(0).cpu().numpy().astype(np.int64)
        object_probabilities = probabilities.squeeze(0)[:, 1].cpu().numpy() if num_classes > 1 else np.zeros_like(targets, dtype=np.float32)

    safe_name = raw_sample.replace("/", "_").replace("\\", "_")
    write_ply(output_dir / f"{safe_name}_gt.ply", points, labels_to_gt_colors(targets))
    write_ply(output_dir / f"{safe_name}_pred.ply", points, labels_to_prediction_colors(predictions))
    write_ply(output_dir / f"{safe_name}_error.ply", points, labels_to_error_colors(targets, predictions))

    meter = SegmentationMeter(num_classes)
    meter.update(predictions, targets)
    metrics = meter.compute()
    return {
        "raw_sample": raw_sample,
        "source_path": item["path"],
        "gt": f"{safe_name}_gt.ply",
        "prediction": f"{safe_name}_pred.ply",
        "error": f"{safe_name}_error.ply",
        "object_probability_min": float(object_probabilities.min()),
        "object_probability_mean": float(object_probabilities.mean()),
        "object_probability_max": float(object_probabilities.max()),
        "metrics": metrics,
    }


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

    model = build_pointnet2_semseg_from_config(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    dataset_config = config.get("dataset", {})
    data_root = resolve_data_root(config, args.data)
    use_normals = config.get("model", {}).get("input_features", "xyz_normal") == "xyz_normal"
    normalize = args.normalize or dataset_config.get("normalize", "scene_center")
    dataset = PointNet2SemSegDataset(
        data_root,
        args.split,
        use_normals=use_normals,
        normalize=normalize,
        augment=PointCloudAugmentationConfig(enabled=False),
        seed=int(config.get("seed", 7)),
    )
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    dataset.files = dataset.files[: args.max_samples]

    output_dir = Path(args.out) if args.out else default_output_dir(checkpoint_path, args.split)
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_meter = SegmentationMeter(num_classes)
    samples = []
    for index in range(len(dataset)):
        item = dataset[index]
        result = preview_sample(model, item, device, output_dir, num_classes)
        samples.append(result)
        predictions = np.asarray(result["metrics"]["confusion_matrix"])
        overall_meter.confusion += predictions
        print(
            f"[preview] {result['raw_sample']}: "
            f"miou={result['metrics']['mean_iou']:.4f} "
            f"object_iou={result['metrics'].get('object_iou', 0.0):.4f} "
            f"object_recall={result['metrics'].get('object_recall', 0.0):.4f}"
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
        "overall_metrics": overall_meter.compute(),
        "samples": samples,
        "legend": {
            "gt": {"background": COLOR_BACKGROUND.tolist(), "object": COLOR_GT_OBJECT.tolist()},
            "prediction": {"background": COLOR_BACKGROUND.tolist(), "object": COLOR_PRED_OBJECT.tolist()},
            "error": {
                "true_negative": COLOR_BACKGROUND.tolist(),
                "true_positive": COLOR_TRUE_POSITIVE.tolist(),
                "false_positive": COLOR_FALSE_POSITIVE.tolist(),
                "false_negative": COLOR_FALSE_NEGATIVE.tolist(),
            },
        },
    }
    (output_dir / "preview_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[preview] wrote previews: {output_dir}")
    print(f"[preview] overall: {summary['overall_metrics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
