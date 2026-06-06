"""Helpers for reading raw Blender synthetic dataset folders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_PREVIEW_FILES = (
    "metadata.json",
    "sensor_data.npz",
    "rgb.png",
    "label_overlay.png",
    "depth_preview.png",
    "instance_mask.png",
    "normal_camera.png",
)

REQUIRED_NPZ_KEYS = (
    "depth_m",
    "instance_mask",
    "normal_world",
    "normal_camera",
    "points_camera",
    "points_world",
    "point_instance_ids",
    "point_semantic_ids",
    "point_pixels",
)


@dataclass(frozen=True)
class RawSample:
    """A single raw generated sample folder."""

    name: str
    path: Path

    @property
    def metadata_path(self) -> Path:
        return self.path / "metadata.json"

    @property
    def sensor_path(self) -> Path:
        return self.path / "sensor_data.npz"


def discover_samples(dataset_root: Path) -> list[RawSample]:
    """Return raw sample folders sorted by name."""

    root = Path(dataset_root)
    if not root.exists():
        return []
    return [
        RawSample(path.name, path)
        for path in sorted(root.iterdir())
        if path.is_dir() and path.name.startswith("sample_")
    ]


def load_metadata(sample: RawSample | Path) -> dict[str, Any]:
    """Load a sample metadata JSON file."""

    path = sample.metadata_path if isinstance(sample, RawSample) else Path(sample) / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz_arrays(sample: RawSample | Path, keys: tuple[str, ...] = REQUIRED_NPZ_KEYS) -> dict[str, np.ndarray]:
    """Load selected arrays from a compressed sensor_data.npz file.

    Arrays are copied out of the NpzFile so callers can use them after the file
    handle is closed.
    """

    path = sample.sensor_path if isinstance(sample, RawSample) else Path(sample) / "sensor_data.npz"
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in keys if key in data.files}


def visible_instance_ids_from_arrays(arrays: dict[str, np.ndarray]) -> set[int]:
    """Return visible non-background instance ids from point labels."""

    if "point_instance_ids" not in arrays:
        return set()
    return {int(value) for value in np.unique(arrays["point_instance_ids"]) if int(value) != 0}


def sample_basic_stats(sample: RawSample | Path) -> dict[str, Any]:
    """Return display-friendly stats for a raw sample."""

    raw_sample = sample if isinstance(sample, RawSample) else RawSample(Path(sample).name, Path(sample))
    metadata = load_metadata(raw_sample)
    arrays = load_npz_arrays(raw_sample, ("point_instance_ids",))
    visible_ids = visible_instance_ids_from_arrays(arrays)
    settings = metadata.get("settings", {})
    return {
        "sample": raw_sample.name,
        "object_count": int(metadata.get("object_count", len(metadata.get("instances", [])))),
        "visible_objects": len(visible_ids),
        "visible_points": int(metadata.get("visible_object_point_count", arrays.get("point_instance_ids", np.array([])).shape[0])),
        "seed": settings.get("seed"),
        "attempt_idx": settings.get("attempt_idx"),
        "collision_margin": settings.get("collision_margin"),
        "collision_shape": settings.get("collision_shape"),
        "out_of_bin_policy": settings.get("out_of_bin_policy", "unknown"),
        "min_visible_objects": settings.get("min_visible_objects"),
        "min_visible_points": settings.get("min_visible_points"),
    }
