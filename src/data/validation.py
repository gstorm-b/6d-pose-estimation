"""Validation for raw Blender synthetic datasets."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from .raw_dataset import REQUIRED_NPZ_KEYS, REQUIRED_PREVIEW_FILES, RawSample, discover_samples


STATUS_OK = "OK"
STATUS_MISSING_FILES = "MISSING_FILES"
STATUS_BAD_NPZ_SCHEMA = "BAD_NPZ_SCHEMA"
STATUS_BAD_INSTANCE_IDS = "BAD_INSTANCE_IDS"
STATUS_BAD_GENERATION_POLICY = "BAD_GENERATION_POLICY"
STATUS_SPARSE = "SPARSE"
STATUS_OUT_OF_BIN = "OUT_OF_BIN"

STATUS_PRIORITY = (
    STATUS_MISSING_FILES,
    STATUS_BAD_NPZ_SCHEMA,
    STATUS_BAD_INSTANCE_IDS,
    STATUS_OUT_OF_BIN,
    STATUS_BAD_GENERATION_POLICY,
    STATUS_SPARSE,
)


@dataclass(frozen=True)
class SampleValidationResult:
    sample: str
    status: str
    issues: tuple[str, ...]
    object_count: int
    visible_objects: int
    visible_points: int
    out_of_bin_count: int


@dataclass(frozen=True)
class DatasetValidationReport:
    dataset_root: Path
    sample_count: int
    ok_count: int
    status_counts: dict[str, int]
    summary: dict[str, Any]
    samples: tuple[SampleValidationResult, ...]


def validate_dataset(
    dataset_root: Path,
    project_root: Path | None = None,
    *,
    allow_legacy_filtering: bool = False,
) -> DatasetValidationReport:
    """Validate all sample folders in a raw dataset root."""

    root = Path(dataset_root)
    project = Path.cwd() if project_root is None else Path(project_root)
    results = tuple(
        validate_sample(
            sample,
            dataset_root=root,
            project_root=project,
            allow_legacy_filtering=allow_legacy_filtering,
        )
        for sample in discover_samples(root)
    )
    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
    summary = _dataset_summary(results)
    return DatasetValidationReport(
        dataset_root=root,
        sample_count=len(results),
        ok_count=status_counts.get(STATUS_OK, 0),
        status_counts=status_counts,
        summary=summary,
        samples=results,
    )


def validate_sample(
    sample: RawSample | Path,
    dataset_root: Path | None = None,
    project_root: Path | None = None,
    *,
    allow_legacy_filtering: bool = False,
) -> SampleValidationResult:
    """Validate a single raw sample folder."""

    raw_sample = sample if isinstance(sample, RawSample) else RawSample(Path(sample).name, Path(sample))
    root = Path(dataset_root) if dataset_root is not None else raw_sample.path.parent
    project = Path.cwd() if project_root is None else Path(project_root)
    issues: list[str] = []
    status_markers: set[str] = set()
    object_count = 0
    visible_objects = 0
    visible_points = 0
    out_of_bin_count = 0

    missing = [name for name in REQUIRED_PREVIEW_FILES if not (raw_sample.path / name).exists()]
    if missing:
        issues.append(f"missing files: {', '.join(missing)}")
        status_markers.add(STATUS_MISSING_FILES)
        return _result(raw_sample.name, status_markers, issues, object_count, visible_objects, visible_points, out_of_bin_count)

    try:
        metadata = json.loads(raw_sample.metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        issues.append(f"metadata unreadable: {exc}")
        status_markers.add(STATUS_BAD_NPZ_SCHEMA)
        return _result(raw_sample.name, status_markers, issues, object_count, visible_objects, visible_points, out_of_bin_count)

    try:
        with np.load(raw_sample.sensor_path) as data:
            arrays = {key: np.asarray(data[key]) for key in data.files}
    except Exception as exc:  # noqa: BLE001 - report validation failure, do not crash the GUI.
        issues.append(f"sensor_data.npz unreadable: {exc}")
        status_markers.add(STATUS_BAD_NPZ_SCHEMA)
        return _result(raw_sample.name, status_markers, issues, object_count, visible_objects, visible_points, out_of_bin_count)

    object_count = int(metadata.get("object_count", len(metadata.get("instances", []))))
    schema_issues = _check_npz_schema(arrays)
    if schema_issues:
        issues.extend(schema_issues)
        status_markers.add(STATUS_BAD_NPZ_SCHEMA)
        return _result(raw_sample.name, status_markers, issues, object_count, visible_objects, visible_points, out_of_bin_count)

    point_instance_ids = arrays["point_instance_ids"]
    visible_ids = {int(value) for value in np.unique(point_instance_ids) if int(value) != 0}
    visible_objects = len(visible_ids)
    visible_points = int(point_instance_ids.shape[0])

    id_issues = _check_instance_ids(metadata, arrays, visible_ids)
    if id_issues:
        issues.extend(id_issues)
        status_markers.add(STATUS_BAD_INSTANCE_IDS)

    policy_issues = _check_generation_policy(metadata, allow_legacy_filtering=allow_legacy_filtering)
    if policy_issues:
        issues.extend(policy_issues)
        status_markers.add(STATUS_BAD_GENERATION_POLICY)

    out_of_bin_ids, out_of_bin_issues = _check_out_of_bin(metadata, root, project)
    out_of_bin_count = len(out_of_bin_ids)
    if out_of_bin_issues:
        issues.extend(out_of_bin_issues)
        status_markers.add(STATUS_OUT_OF_BIN)
    elif out_of_bin_ids:
        issues.append(f"out-of-bin instance ids: {out_of_bin_ids}")
        status_markers.add(STATUS_OUT_OF_BIN)

    sparse_issues = _check_sparse(metadata, visible_objects, visible_points)
    if sparse_issues:
        issues.extend(sparse_issues)
        status_markers.add(STATUS_SPARSE)

    return _result(raw_sample.name, status_markers, issues, object_count, visible_objects, visible_points, out_of_bin_count)


def _result(
    sample_name: str,
    markers: set[str],
    issues: list[str],
    object_count: int,
    visible_objects: int,
    visible_points: int,
    out_of_bin_count: int,
) -> SampleValidationResult:
    status = STATUS_OK
    for candidate in STATUS_PRIORITY:
        if candidate in markers:
            status = candidate
            break
    return SampleValidationResult(
        sample=sample_name,
        status=status,
        issues=tuple(issues),
        object_count=object_count,
        visible_objects=visible_objects,
        visible_points=visible_points,
        out_of_bin_count=out_of_bin_count,
    )


def _check_npz_schema(arrays: dict[str, np.ndarray]) -> list[str]:
    issues: list[str] = []
    missing = [key for key in REQUIRED_NPZ_KEYS if key not in arrays]
    if missing:
        issues.append(f"missing npz keys: {', '.join(missing)}")
        return issues

    depth = arrays["depth_m"]
    mask = arrays["instance_mask"]
    normal_world = arrays["normal_world"]
    normal_camera = arrays["normal_camera"]
    points_camera = arrays["points_camera"]
    points_world = arrays["points_world"]
    point_instance_ids = arrays["point_instance_ids"]
    point_semantic_ids = arrays["point_semantic_ids"]
    point_pixels = arrays["point_pixels"]

    if depth.ndim != 2:
        issues.append(f"depth_m must be 2D, got {depth.shape}")
        return issues
    if mask.shape != depth.shape:
        issues.append(f"instance_mask shape {mask.shape} does not match depth_m {depth.shape}")
    for key, array in (("normal_world", normal_world), ("normal_camera", normal_camera)):
        if array.shape != depth.shape + (3,):
            issues.append(f"{key} shape {array.shape} does not match depth_m + 3")
    for key, array in (("points_camera", points_camera), ("points_world", points_world)):
        if array.ndim != 2 or array.shape[1] != 3:
            issues.append(f"{key} must have shape (N, 3), got {array.shape}")
    n_points = points_camera.shape[0] if points_camera.ndim == 2 else -1
    if points_world.shape[0] != n_points:
        issues.append("points_world length does not match points_camera")
    if point_instance_ids.shape != (n_points,):
        issues.append("point_instance_ids length does not match points_camera")
    if point_semantic_ids.shape != (n_points,):
        issues.append("point_semantic_ids length does not match points_camera")
    if point_pixels.shape != (n_points, 2):
        issues.append("point_pixels shape does not match points_camera")
    if not np.all(np.isfinite(points_camera)) or not np.all(np.isfinite(points_world)):
        issues.append("point arrays contain non-finite values")
    return issues


def _check_instance_ids(metadata: dict[str, Any], arrays: dict[str, np.ndarray], visible_ids: set[int]) -> list[str]:
    issues: list[str] = []
    metadata_ids = {int(instance.get("instance_id")) for instance in metadata.get("instances", []) if "instance_id" in instance}
    mask_ids = {int(value) for value in np.unique(arrays["instance_mask"]) if int(value) != 0}
    point_ids = visible_ids
    if not point_ids.issubset(metadata_ids):
        issues.append(f"point ids missing from metadata: {sorted(point_ids - metadata_ids)}")
    if not mask_ids.issubset(metadata_ids):
        issues.append(f"mask ids missing from metadata: {sorted(mask_ids - metadata_ids)}")
    if not point_ids.issubset(mask_ids):
        issues.append(f"point ids missing from instance_mask: {sorted(point_ids - mask_ids)}")
    semantic_ids = set(int(value) for value in np.unique(arrays["point_semantic_ids"]))
    if not semantic_ids.issubset({0, int(metadata.get("class_id", 1))}):
        issues.append(f"unexpected semantic ids: {sorted(semantic_ids)}")
    return issues


def _check_sparse(metadata: dict[str, Any], visible_objects: int, visible_points: int) -> list[str]:
    settings = metadata.get("settings", {})
    issues: list[str] = []
    min_visible_objects = settings.get("min_visible_objects")
    min_visible_points = settings.get("min_visible_points")
    if min_visible_objects is not None and visible_objects < int(min_visible_objects):
        issues.append(f"visible object count {visible_objects} below min_visible_objects {min_visible_objects}")
    if min_visible_points is not None and visible_points < int(min_visible_points):
        issues.append(f"visible point count {visible_points} below min_visible_points {min_visible_points}")
    return issues


def _check_generation_policy(metadata: dict[str, Any], *, allow_legacy_filtering: bool) -> list[str]:
    settings = metadata.get("settings", {})
    issues: list[str] = []
    if not allow_legacy_filtering and settings.get("allow_out_of_bin_filtering") is True:
        issues.append("legacy out-of-bin filtering was enabled; regenerate with strict reject policy")
    if not allow_legacy_filtering and settings.get("out_of_bin_policy") == "filter":
        issues.append("out_of_bin_policy is filter; expected reject_attempt for training datasets")
    return issues


def _check_out_of_bin(metadata: dict[str, Any], dataset_root: Path, project_root: Path) -> tuple[list[int], list[str]]:
    model_path = _resolve_model_path(metadata.get("model_path"), dataset_root, project_root)
    if model_path is None:
        return [], [f"model path not found: {metadata.get('model_path')}"]
    try:
        corners = _stl_bbox_corners(model_path)
    except Exception as exc:  # noqa: BLE001 - validation should report the sample-level issue.
        return [], [f"could not read STL bbox: {exc}"]

    settings = metadata.get("settings", {})
    bin_x = float(settings.get("bin_x", 0.24))
    bin_y = float(settings.get("bin_y", 0.24))
    tolerance = float(settings.get("out_of_bin_tolerance", 0.015))
    x_limit = bin_x / 2.0 + tolerance
    y_limit = bin_y / 2.0 + tolerance
    min_z = -0.04

    out_of_bin_ids: list[int] = []
    for instance in metadata.get("instances", []):
        instance_id = int(instance.get("instance_id", -1))
        if "object_to_world" not in instance:
            out_of_bin_ids.append(instance_id)
            continue
        object_to_world = np.asarray(instance["object_to_world"], dtype=np.float64)
        if object_to_world.shape != (4, 4) or not np.all(np.isfinite(object_to_world)):
            out_of_bin_ids.append(instance_id)
            continue
        world_corners = (object_to_world @ corners.T).T[:, :3]
        in_bounds = (
            float(world_corners[:, 0].min()) >= -x_limit
            and float(world_corners[:, 0].max()) <= x_limit
            and float(world_corners[:, 1].min()) >= -y_limit
            and float(world_corners[:, 1].max()) <= y_limit
            and float(world_corners[:, 2].min()) >= min_z
        )
        if not in_bounds:
            out_of_bin_ids.append(instance_id)
    return out_of_bin_ids, []


def _resolve_model_path(value: Any, dataset_root: Path, project_root: Path) -> Path | None:
    if not value:
        return None
    raw_path = Path(str(value))
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.extend((project_root / raw_path, dataset_root / raw_path, dataset_root.parent / raw_path))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _stl_bbox_corners(path: Path) -> np.ndarray:
    vertices = _read_stl_vertices(path)
    if vertices.size == 0:
        raise ValueError("STL has no vertices")
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    return np.array(
        [[x, y, z, 1.0] for x in (mins[0], maxs[0]) for y in (mins[1], maxs[1]) for z in (mins[2], maxs[2])],
        dtype=np.float64,
    )


def _read_stl_vertices(path: Path) -> np.ndarray:
    data = path.read_bytes()
    vertices: list[tuple[float, float, float]] = []
    if len(data) >= 84:
        tri_count = struct.unpack("<I", data[80:84])[0]
        expected_size = 84 + tri_count * 50
        if expected_size == len(data):
            offset = 84
            for _ in range(tri_count):
                triangle = struct.unpack("<12fH", data[offset : offset + 50])
                offset += 50
                vertices.extend((triangle[3:6], triangle[6:9], triangle[9:12]))
            return np.asarray(vertices, dtype=np.float64)

    text = data.decode("utf-8", errors="ignore")
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return np.asarray(vertices, dtype=np.float64)


def _dataset_summary(results: tuple[SampleValidationResult, ...]) -> dict[str, Any]:
    object_counts = [result.object_count for result in results]
    visible_objects = [result.visible_objects for result in results]
    visible_points = [result.visible_points for result in results]
    return {
        "object_count": _min_median_max(object_counts),
        "visible_objects": _min_median_max(visible_objects),
        "visible_points": _min_median_max(visible_points),
        "out_of_bin_samples": sum(1 for result in results if result.out_of_bin_count > 0),
        "out_of_bin_objects": sum(result.out_of_bin_count for result in results),
    }


def _min_median_max(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {
        "min": min(values),
        "median": median(values),
        "max": max(values),
    }
