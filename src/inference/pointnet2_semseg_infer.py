"""Reusable PointNet++ semantic-segmentation inference helpers."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.data.depth_unprojection import unproject_depth_to_camera
from src.data.raw_dataset import RawSample, load_metadata
from src.models.pointnet2_semseg import build_pointnet2_semseg_from_config
from src.training.checkpoint import load_checkpoint
from src.training.seed import seed_everything


COLOR_BACKGROUND = np.array([120, 120, 120], dtype=np.uint8)
COLOR_PRED_OBJECT = np.array([60, 130, 255], dtype=np.uint8)


@dataclass(frozen=True)
class InferenceInput:
    """Model-ready inference input and metadata."""

    points_camera: np.ndarray
    model_input: np.ndarray
    point_pixels: np.ndarray
    raw_sample: str
    source_path: str
    source_type: str
    center: np.ndarray
    features: np.ndarray | None = None
    semantic_labels: np.ndarray | None = None
    instance_labels: np.ndarray | None = None


@dataclass(frozen=True)
class InferenceResult:
    """PointNet++ semantic inference outputs aligned with input points."""

    input: InferenceInput
    logits: np.ndarray
    probabilities: np.ndarray
    predicted_labels: np.ndarray
    elapsed_seconds: float
    device: str
    ops_backend_effective: str


def load_pointnet2_semseg_model(
    checkpoint_path: str | Path,
    *,
    device: str = "cuda",
    ops_backend: str = "pytorch3d",
    seed: int | None = None,
) -> tuple[Any, dict[str, Any], str]:
    """Load a PointNet++ semantic checkpoint for inference."""

    import torch

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config = json.loads(json.dumps(checkpoint.get("config", {})))
    if ops_backend:
        config.setdefault("model", {})["ops_backend"] = ops_backend
    resolved_seed = int(seed if seed is not None else config.get("seed", config.get("train", {}).get("seed", 7)))
    seed_everything(resolved_seed)

    resolved_device = device
    if resolved_device == "cuda" and not torch.cuda.is_available():
        resolved_device = "cpu"

    model = build_pointnet2_semseg_from_config(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(resolved_device)
    model.eval()
    config.setdefault("runtime_inference", {})["checkpoint_epoch"] = checkpoint.get("epoch")
    config["runtime_inference"]["seed"] = resolved_seed
    config["runtime_inference"]["ops_backend_effective"] = getattr(model, "ops_backend_name", "unknown")
    return model, config, resolved_device


def prepare_processed_sample(
    sample_path: str | Path,
    config: dict[str, Any],
    *,
    normalize: str | None = None,
) -> InferenceInput:
    """Load a processed `.npz` sample and prepare model input."""

    path = Path(sample_path)
    with np.load(path) as data:
        points = np.asarray(data["points"], dtype=np.float32)
        features = np.asarray(data["features"], dtype=np.float32) if "features" in data.files else None
        point_pixels = (
            np.asarray(data["point_pixels"], dtype=np.uint16)
            if "point_pixels" in data.files
            else np.zeros((points.shape[0], 2), dtype=np.uint16)
        )
        raw_sample = str(np.asarray(data["raw_sample"]).item()) if "raw_sample" in data.files else path.stem
        semantic_labels = np.asarray(data["semantic_labels"], dtype=np.int64) if "semantic_labels" in data.files else None
        instance_labels = np.asarray(data["instance_labels"], dtype=np.int64) if "instance_labels" in data.files else None

    use_normals = config.get("model", {}).get("input_features", "xyz_normal") == "xyz_normal"
    if use_normals and features is None:
        raise ValueError(f"{path} does not contain normal features required by the checkpoint")
    if not use_normals:
        features = None

    model_points, center = normalize_points(points, normalize or config.get("dataset", {}).get("normalize", "scene_center"))
    model_input = model_points if features is None else np.concatenate([model_points, features], axis=1).astype(np.float32)
    return InferenceInput(
        points_camera=points.astype(np.float32),
        model_input=model_input.astype(np.float32),
        point_pixels=point_pixels.astype(np.uint16),
        raw_sample=raw_sample,
        source_path=str(path),
        source_type="processed",
        center=center.astype(np.float32),
        features=None if features is None else features.astype(np.float32),
        semantic_labels=semantic_labels,
        instance_labels=instance_labels,
    )


def prepare_raw_sample(
    sample_path: str | Path,
    config: dict[str, Any],
    *,
    num_points: int | None = None,
    normalize: str | None = None,
    seed: int | None = None,
) -> InferenceInput:
    """Convert a raw synthetic sample folder into model-ready inference input.

    Sampling is intentionally label-free. If `instance_mask` exists, labels are
    copied into the result for debugging only and are not used to choose points.
    """

    path = Path(sample_path)
    raw_sample = RawSample(path.name, path)
    metadata = load_metadata(raw_sample)
    with np.load(raw_sample.sensor_path) as data:
        depth_m = np.asarray(data["depth_m"], dtype=np.float32)
        normal_camera = np.asarray(data["normal_camera"], dtype=np.float32) if "normal_camera" in data.files else None
        instance_mask = np.asarray(data["instance_mask"], dtype=np.uint16) if "instance_mask" in data.files else None

    points, pixels = unproject_depth_to_camera(depth_m, metadata["camera_intrinsics"])
    if points.shape[0] == 0:
        raise ValueError(f"{path} has no valid depth points")

    target_points = int(num_points or config.get("conversion", {}).get("num_points", points.shape[0]))
    rng = np.random.default_rng(seed if seed is not None else config.get("seed", config.get("train", {}).get("seed", 7)))
    indices = sample_indices(points.shape[0], target_points, rng)
    points = points[indices].astype(np.float32)
    pixels = pixels[indices].astype(np.uint16)

    features = None
    use_normals = config.get("model", {}).get("input_features", "xyz_normal") == "xyz_normal"
    if use_normals:
        if normal_camera is None:
            raise ValueError(f"{path} does not contain normal_camera required by the checkpoint")
        u = pixels[:, 0].astype(np.int64)
        v = pixels[:, 1].astype(np.int64)
        features = normal_camera[v, u].astype(np.float32)

    semantic_labels = None
    instance_labels = None
    if instance_mask is not None:
        u = pixels[:, 0].astype(np.int64)
        v = pixels[:, 1].astype(np.int64)
        instance_labels = instance_mask[v, u].astype(np.int64)
        semantic_labels = (instance_labels > 0).astype(np.int64)

    model_points, center = normalize_points(points, normalize or config.get("dataset", {}).get("normalize", "scene_center"))
    model_input = model_points if features is None else np.concatenate([model_points, features], axis=1).astype(np.float32)
    return InferenceInput(
        points_camera=points.astype(np.float32),
        model_input=model_input.astype(np.float32),
        point_pixels=pixels.astype(np.uint16),
        raw_sample=raw_sample.name,
        source_path=str(path),
        source_type="raw",
        center=center.astype(np.float32),
        features=features,
        semantic_labels=semantic_labels,
        instance_labels=instance_labels,
    )


def normalize_points(points: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Normalize points with the same modes as the training dataset."""

    points = np.asarray(points, dtype=np.float32)
    if mode == "none":
        center = np.zeros((1, 3), dtype=np.float32)
        return points.astype(np.float32), center
    if mode == "scene_center":
        center = points.mean(axis=0, keepdims=True).astype(np.float32)
        return (points - center).astype(np.float32), center
    raise ValueError(f"Unknown normalization mode: {mode}")


def sample_indices(point_count: int, num_points: int, rng: np.random.Generator) -> np.ndarray:
    """Return deterministic label-free sample indices, padding if needed."""

    if point_count <= 0:
        raise ValueError("cannot sample from an empty point cloud")
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    replace = point_count < num_points
    return rng.choice(np.arange(point_count), size=num_points, replace=replace).astype(np.int64)


def run_inference(model: Any, inference_input: InferenceInput, *, device: str) -> InferenceResult:
    """Run model inference for one prepared input."""

    import torch

    inputs = torch.from_numpy(inference_input.model_input.astype(np.float32)).unsqueeze(0).to(device)
    start = time.perf_counter()
    with torch.no_grad():
        logits_tensor = model(inputs)
        probabilities_tensor = torch.softmax(logits_tensor, dim=-1)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    logits = logits_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
    probabilities = probabilities_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
    predicted_labels = probabilities.argmax(axis=1).astype(np.int64)
    return InferenceResult(
        input=inference_input,
        logits=logits,
        probabilities=probabilities,
        predicted_labels=predicted_labels,
        elapsed_seconds=elapsed,
        device=device,
        ops_backend_effective=getattr(model, "ops_backend_name", "unknown"),
    )


def save_inference_npz(path: str | Path, result: InferenceResult) -> None:
    """Save inference arrays to `.npz`."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "points_camera": result.input.points_camera.astype(np.float32),
        "model_input": result.input.model_input.astype(np.float32),
        "point_pixels": result.input.point_pixels.astype(np.uint16),
        "logits": result.logits.astype(np.float32),
        "probabilities": result.probabilities.astype(np.float32),
        "predicted_labels": result.predicted_labels.astype(np.int64),
        "center": result.input.center.astype(np.float32),
        "raw_sample": np.asarray(result.input.raw_sample),
        "source_path": np.asarray(result.input.source_path),
        "source_type": np.asarray(result.input.source_type),
    }
    if result.input.features is not None:
        arrays["features"] = result.input.features.astype(np.float32)
    if result.input.semantic_labels is not None:
        arrays["semantic_labels"] = result.input.semantic_labels.astype(np.int64)
    if result.input.instance_labels is not None:
        arrays["instance_labels"] = result.input.instance_labels.astype(np.int64)
    np.savez_compressed(output_path, **arrays)


def save_inference_summary(path: str | Path, result: InferenceResult, config: dict[str, Any]) -> None:
    """Save lightweight inference metadata to JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    object_probabilities = result.probabilities[:, 1] if result.probabilities.shape[1] > 1 else np.zeros_like(result.predicted_labels, dtype=np.float32)
    summary: dict[str, Any] = {
        "raw_sample": result.input.raw_sample,
        "source_path": result.input.source_path,
        "source_type": result.input.source_type,
        "point_count": int(result.input.points_camera.shape[0]),
        "device": result.device,
        "ops_backend_effective": result.ops_backend_effective,
        "checkpoint_epoch": config.get("runtime_inference", {}).get("checkpoint_epoch"),
        "seed": config.get("runtime_inference", {}).get("seed"),
        "input_features": config.get("model", {}).get("input_features", "xyz_normal"),
        "elapsed_seconds": result.elapsed_seconds,
        "predicted_object_points": int(np.count_nonzero(result.predicted_labels == 1)),
        "object_probability_min": float(object_probabilities.min()) if object_probabilities.size else 0.0,
        "object_probability_mean": float(object_probabilities.mean()) if object_probabilities.size else 0.0,
        "object_probability_max": float(object_probabilities.max()) if object_probabilities.size else 0.0,
    }
    if result.input.semantic_labels is not None:
        summary["ground_truth_object_points"] = int(np.count_nonzero(result.input.semantic_labels == 1))
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_prediction_ply(path: str | Path, points: np.ndarray, predicted_labels: np.ndarray) -> None:
    """Write predicted semantic labels as a colored ASCII PLY."""

    colors = np.repeat(COLOR_BACKGROUND[None, :], predicted_labels.shape[0], axis=0)
    colors[predicted_labels == 1] = COLOR_PRED_OBJECT
    write_colored_ply(path, points, colors)


def write_colored_ply(path: str | Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write colored points to an ASCII PLY file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.shape[0] != colors.shape[0]:
        raise ValueError("points and colors length mismatch")
    with output_path.open("w", encoding="ascii") as file:
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
