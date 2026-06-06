"""Conversion helpers for PointNet++ semantic-segmentation datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .depth_unprojection import unproject_depth_to_camera
from .raw_dataset import RawSample, load_metadata


@dataclass(frozen=True)
class PointNet2ConversionConfig:
    num_points: int = 16384
    use_normals: bool = True
    object_fraction_target: float = 0.65
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 7

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "PointNet2ConversionConfig":
        if not value:
            return cls()
        known = cls.__dataclass_fields__.keys()  # type: ignore[attr-defined]
        kwargs = {key: value[key] for key in known if key in value}
        return cls(**kwargs)


@dataclass(frozen=True)
class ConvertedSample:
    arrays: dict[str, np.ndarray]
    stats: dict[str, Any]


def split_samples(samples: list[RawSample], config: PointNet2ConversionConfig) -> dict[str, list[RawSample]]:
    """Return deterministic train/val/test splits."""

    if not samples:
        return {"train": [], "val": [], "test": []}

    rng = np.random.default_rng(config.seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)

    train_count = int(round(len(samples) * config.train_ratio))
    val_count = int(round(len(samples) * config.val_ratio))
    test_count = len(samples) - train_count - val_count
    if len(samples) >= 3:
        val_count = max(1, val_count)
        test_count = max(1, test_count)
        train_count = max(1, len(samples) - val_count - test_count)
        while train_count + val_count + test_count > len(samples):
            if train_count >= val_count and train_count > 1:
                train_count -= 1
            elif val_count > 1:
                val_count -= 1
            else:
                test_count -= 1
    elif train_count + val_count > len(samples):
        val_count = max(0, len(samples) - train_count)

    train_idx = set(int(i) for i in indices[:train_count])
    val_idx = set(int(i) for i in indices[train_count : train_count + val_count])
    return {
        "train": [sample for idx, sample in enumerate(samples) if idx in train_idx],
        "val": [sample for idx, sample in enumerate(samples) if idx in val_idx],
        "test": [sample for idx, sample in enumerate(samples) if idx not in train_idx and idx not in val_idx],
    }


def convert_raw_sample(
    sample: RawSample,
    config: PointNet2ConversionConfig,
    rng: np.random.Generator,
) -> ConvertedSample:
    """Convert one raw Blender sample into a fixed-size PointNet++ sample."""

    metadata = load_metadata(sample)
    with np.load(sample.sensor_path) as data:
        depth_m = np.asarray(data["depth_m"], dtype=np.float32)
        instance_mask = np.asarray(data["instance_mask"], dtype=np.uint16)
        normal_camera = np.asarray(data["normal_camera"], dtype=np.float32) if config.use_normals else None

    points, pixels = unproject_depth_to_camera(depth_m, metadata["camera_intrinsics"])
    if points.shape[0] == 0:
        raise ValueError(f"{sample.name} has no valid depth points")

    point_u = pixels[:, 0].astype(np.int64)
    point_v = pixels[:, 1].astype(np.int64)
    instance_labels_full = instance_mask[point_v, point_u].astype(np.int64)
    semantic_labels_full = (instance_labels_full > 0).astype(np.int64)
    features_full = None
    if normal_camera is not None:
        features_full = normal_camera[point_v, point_u].astype(np.float32)

    indices = balanced_sample_indices(
        semantic_labels_full,
        num_points=config.num_points,
        object_fraction_target=config.object_fraction_target,
        rng=rng,
    )

    sampled_points = points[indices].astype(np.float32)
    sampled_pixels = pixels[indices].astype(np.uint16)
    sampled_semantic = semantic_labels_full[indices].astype(np.int64)
    sampled_instance = instance_labels_full[indices].astype(np.int64)

    arrays = {
        "points": sampled_points,
        "semantic_labels": sampled_semantic,
        "instance_labels": sampled_instance,
        "point_pixels": sampled_pixels,
        "raw_sample": np.asarray(sample.name),
    }
    if features_full is not None:
        arrays["features"] = features_full[indices].astype(np.float32)

    stats = {
        "raw_sample": sample.name,
        "valid_depth_points": int(points.shape[0]),
        "full_object_points": int(np.count_nonzero(semantic_labels_full == 1)),
        "full_background_points": int(np.count_nonzero(semantic_labels_full == 0)),
        "sampled_points": int(sampled_points.shape[0]),
        "sampled_object_points": int(np.count_nonzero(sampled_semantic == 1)),
        "sampled_background_points": int(np.count_nonzero(sampled_semantic == 0)),
        "visible_instances": int(len(set(int(v) for v in np.unique(sampled_instance) if int(v) != 0))),
        "used_normals": bool(config.use_normals),
    }
    return ConvertedSample(arrays=arrays, stats=stats)


def balanced_sample_indices(
    semantic_labels: np.ndarray,
    num_points: int,
    object_fraction_target: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample fixed-size point indices with a target object/background mix."""

    labels = np.asarray(semantic_labels)
    object_indices = np.flatnonzero(labels == 1)
    background_indices = np.flatnonzero(labels == 0)
    if labels.size == 0:
        raise ValueError("cannot sample from an empty point set")

    target_object = int(round(num_points * object_fraction_target))
    target_background = num_points - target_object

    chosen_parts: list[np.ndarray] = []
    chosen_object = _sample_group(object_indices, target_object, rng)
    chosen_background = _sample_group(background_indices, target_background, rng)
    if chosen_object.size:
        chosen_parts.append(chosen_object)
    if chosen_background.size:
        chosen_parts.append(chosen_background)

    chosen = np.concatenate(chosen_parts) if chosen_parts else np.empty((0,), dtype=np.int64)
    if chosen.shape[0] < num_points:
        available = np.arange(labels.size)
        fill = rng.choice(available, size=num_points - chosen.shape[0], replace=available.size < num_points)
        chosen = np.concatenate([chosen, fill.astype(np.int64)])
    elif chosen.shape[0] > num_points:
        chosen = rng.choice(chosen, size=num_points, replace=False).astype(np.int64)

    rng.shuffle(chosen)
    return chosen.astype(np.int64)


def _sample_group(indices: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if count <= 0 or indices.size == 0:
        return np.empty((0,), dtype=np.int64)
    replace = indices.size < count
    return rng.choice(indices, size=count, replace=replace).astype(np.int64)


def summarize_converted_samples(sample_stats: list[dict[str, Any]], splits: dict[str, list[str]]) -> dict[str, Any]:
    """Build dataset-level summary stats for converted samples."""

    return {
        "sample_count": len(sample_stats),
        "splits": {name: len(values) for name, values in splits.items()},
        "split_samples": splits,
        "valid_depth_points": _min_median_max([int(s["valid_depth_points"]) for s in sample_stats]),
        "full_object_points": _min_median_max([int(s["full_object_points"]) for s in sample_stats]),
        "full_background_points": _min_median_max([int(s["full_background_points"]) for s in sample_stats]),
        "sampled_object_points": _min_median_max([int(s["sampled_object_points"]) for s in sample_stats]),
        "sampled_background_points": _min_median_max([int(s["sampled_background_points"]) for s in sample_stats]),
        "visible_instances": _min_median_max([int(s["visible_instances"]) for s in sample_stats]),
        "samples": sample_stats,
    }


def write_schema(path: Path, config: PointNet2ConversionConfig) -> None:
    text = f"""# PointNet++ Semantic Segmentation Processed Dataset

Each split folder contains `.npz` files converted from raw Blender samples.

## Arrays

- `points`: `(num_points, 3) float32`, positive-forward camera coordinates in meters.
- `features`: `(num_points, 3) float32`, optional `normal_camera` features when `use_normals=true`.
- `semantic_labels`: `(num_points,) int64`, `0=background/bin`, `1=K41144`.
- `instance_labels`: `(num_points,) int64`, `0=background`, positive values are raw instance ids.
- `point_pixels`: `(num_points, 2) uint16`, `(u, v)` source pixel coordinates.
- `raw_sample`: string scalar with the source raw sample folder name.

## Conversion Settings

- `num_points`: {config.num_points}
- `use_normals`: {str(config.use_normals).lower()}
- `object_fraction_target`: {config.object_fraction_target}

Raw samples are reconstructed from `depth_m` and labeled with `instance_mask`; they do not use the raw object-only `points_camera` arrays as the full training scene.
"""
    path.write_text(text, encoding="utf-8")


def _min_median_max(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {"min": min(values), "median": float(np.median(values)), "max": max(values)}
