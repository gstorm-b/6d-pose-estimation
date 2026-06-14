"""Geometry helpers for pose estimation in explicit object/camera frames."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def load_stl_vertices_m(path: str | Path, *, scale: float = 1.0) -> np.ndarray:
    """Load STL vertices and return them in meters."""

    data = Path(path).read_bytes()
    vertices = _load_ascii_stl_vertices(data)
    if vertices is None:
        vertices = _load_binary_stl_vertices(data)
    return (vertices.astype(np.float32) * float(scale)).astype(np.float32)


def sample_model_points(
    vertices_m: np.ndarray,
    count: int,
    *,
    seed: int = 7,
) -> np.ndarray:
    """Deterministically sample model vertices for ADD-style losses."""

    vertices = np.asarray(vertices_m, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"expected vertices with shape (N, 3), got {vertices.shape}")
    if vertices.shape[0] == 0:
        raise ValueError("model vertices are empty")
    rng = np.random.default_rng(seed)
    replace = vertices.shape[0] < int(count)
    indices = rng.choice(vertices.shape[0], size=int(count), replace=replace)
    return vertices[indices].astype(np.float32)


def model_diameter_m(vertices_m: np.ndarray) -> float:
    """Return bbox diagonal diameter in meters."""

    vertices = np.asarray(vertices_m, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        raise ValueError(f"expected non-empty vertices with shape (N, 3), got {vertices.shape}")
    return float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))


def symmetry_matrices_for_name(name: str | None) -> np.ndarray:
    """Return object-frame finite symmetry rotations as 3x3 matrices (built-in fallback)."""

    normalized = (name or "none").lower().replace("-", "_")
    if normalized in {"k41144", "k41144_y_180"}:
        return np.stack([np.eye(3, dtype=np.float32), rotation_y_np(np.pi)], axis=0)
    if normalized in {"none", "identity", "bending_pipe"}:
        return np.eye(3, dtype=np.float32).reshape(1, 3, 3)
    raise ValueError(f"Unknown symmetry set: {name}")


SYMMETRY_SCHEMA_VERSION = 1
SYMMETRY_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "symmetry"
DEFAULT_CONTINUOUS_SAMPLES = 72


def axis_angle_to_matrix(axis: Any, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation matrix for a unit-normalized axis and angle in radians."""

    a = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(a))
    if norm < 1e-12:
        raise ValueError("symmetry axis must be non-zero")
    a = a / norm
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]], dtype=np.float64)
    matrix = np.eye(3, dtype=np.float64) + np.sin(angle_rad) * k + (1.0 - np.cos(angle_rad)) * (k @ k)
    return matrix.astype(np.float32)


@dataclass
class SymmetryDefinition:
    """Object-frame symmetry as an explicit set of equivalent rotations."""

    name: str
    matrices: np.ndarray  # (M, 3, 3), identity first
    has_continuous: bool
    source: str

    def __post_init__(self) -> None:
        self.matrices = np.asarray(self.matrices, dtype=np.float32).reshape(-1, 3, 3)


def _expand_symmetry(
    finite_rotations: list[dict[str, Any]] | None,
    continuous_axes: list[dict[str, Any]] | None,
    continuous_samples: int,
) -> tuple[np.ndarray, bool]:
    matrices = [np.eye(3, dtype=np.float32)]
    for entry in finite_rotations or []:
        axis = entry["axis"]
        angle_deg = float(entry.get("angle_deg", entry.get("angle", 0.0)))
        matrices.append(axis_angle_to_matrix(axis, np.radians(angle_deg)))
    has_continuous = bool(continuous_axes)
    for entry in continuous_axes or []:
        axis = entry["axis"]
        samples = int(entry.get("samples", continuous_samples))
        # Approximate a continuous (revolute) symmetry with a dense finite set.
        for theta in np.linspace(0.0, 2.0 * np.pi, num=max(2, samples), endpoint=False):
            if abs(float(theta)) < 1e-9:
                continue
            matrices.append(axis_angle_to_matrix(axis, float(theta)))
    return np.stack(matrices, axis=0).astype(np.float32), has_continuous


def load_symmetry(source: Any, *, continuous_samples: int = DEFAULT_CONTINUOUS_SAMPLES) -> SymmetryDefinition:
    """Load a symmetry definition from a dict, a JSON path, or a named config/built-in.

    Resolution order for a string `name`: `configs/symmetry/<name>.json` if it
    exists, otherwise the built-in `symmetry_matrices_for_name` fallback. This
    keeps `symmetry_name: k41144` working as a deprecated alias.
    """

    if isinstance(source, SymmetryDefinition):
        return source
    if isinstance(source, np.ndarray):
        return SymmetryDefinition(name="array", matrices=source, has_continuous=False, source="array")

    data: dict[str, Any] | None = None
    name = "none"
    origin = "builtin"
    if isinstance(source, dict):
        data = source
        name = str(source.get("name", "inline"))
        origin = "dict"
    elif isinstance(source, (str, Path)):
        text = str(source)
        path = Path(text)
        if not path.suffix and not path.exists():
            candidate = SYMMETRY_CONFIG_DIR / f"{text}.json"
            if candidate.exists():
                path = candidate
        if path.suffix == ".json" or path.exists():
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            name = str(data.get("name", Path(path).stem))
            origin = str(path)
        else:
            matrices = symmetry_matrices_for_name(text)
            return SymmetryDefinition(name=text, matrices=matrices, has_continuous=False, source="builtin")
    else:
        raise TypeError(f"unsupported symmetry source: {type(source)!r}")

    matrices, has_continuous = _expand_symmetry(
        data.get("finite_rotations"), data.get("continuous_axes"), continuous_samples
    )
    return SymmetryDefinition(name=name, matrices=matrices, has_continuous=has_continuous, source=origin)


def resolve_symmetry_matrices(source: Any, *, continuous_samples: int = DEFAULT_CONTINUOUS_SAMPLES) -> np.ndarray:
    """Return (M, 3, 3) symmetry matrices for a dict/path/name/array source."""

    return load_symmetry(source, continuous_samples=continuous_samples).matrices


def rotation_y_np(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def _load_ascii_stl_vertices(data: bytes) -> np.ndarray | None:
    header = data[:512].decode("ascii", errors="ignore").lower()
    if not header.lstrip().startswith("solid") or "facet normal" not in header:
        return None
    vertices: list[list[float]] = []
    for line in data.decode("ascii", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        return None
    return np.asarray(vertices, dtype=np.float32)


def _load_binary_stl_vertices(data: bytes) -> np.ndarray:
    if len(data) < 84:
        raise ValueError("STL file is too small")
    triangle_count = struct.unpack("<I", data[80:84])[0]
    expected_size = 84 + triangle_count * 50
    if len(data) < expected_size:
        raise ValueError(f"Binary STL truncated: expected at least {expected_size} bytes, got {len(data)}")
    vertices = np.zeros((triangle_count * 3, 3), dtype=np.float32)
    offset = 84
    cursor = 0
    for _ in range(triangle_count):
        values = struct.unpack("<12fH", data[offset : offset + 50])
        offset += 50
        vertices[cursor] = values[3:6]
        vertices[cursor + 1] = values[6:9]
        vertices[cursor + 2] = values[9:12]
        cursor += 3
    return vertices


def rotation_6d_to_matrix(rotation_6d: Any):
    """Convert Zhou-style 6D rotation representation to a matrix."""

    import torch
    from torch.nn import functional as F

    x = rotation_6d.reshape(*rotation_6d.shape[:-1], 3, 2)
    a1 = x[..., :, 0]
    a2 = x[..., :, 1]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def matrix_to_rotation_6d(rotation_matrix: Any):
    """Return the first two rotation matrix columns as a 6D representation."""

    return rotation_matrix[..., :, :2].reshape(*rotation_matrix.shape[:-2], 6)


def transform_model_points(model_points: Any, rotation_object_to_camera: Any, object_origin_camera: Any):
    """Apply `object_to_camera` rotation and translation to model points."""

    return model_points @ rotation_object_to_camera.transpose(-1, -2) + object_origin_camera.unsqueeze(-2)


def symmetry_aware_add(
    *,
    rotation_pred_object_to_camera: Any,
    origin_pred_camera: Any,
    rotation_gt_object_to_camera: Any,
    origin_gt_camera: Any,
    model_points_object: Any,
    symmetry_matrices_object: Any,
) -> Any:
    """Return per-sample ADD under the best finite object-frame symmetry."""

    import torch

    pred_points = transform_model_points(model_points_object.unsqueeze(0), rotation_pred_object_to_camera, origin_pred_camera)
    candidate_errors = []
    for sym_idx in range(symmetry_matrices_object.shape[0]):
        sym = symmetry_matrices_object[sym_idx].to(device=rotation_gt_object_to_camera.device, dtype=rotation_gt_object_to_camera.dtype)
        rotation_gt_equiv = rotation_gt_object_to_camera @ sym
        gt_points = transform_model_points(model_points_object.unsqueeze(0), rotation_gt_equiv, origin_gt_camera)
        candidate_errors.append(torch.linalg.norm(pred_points - gt_points, dim=-1).mean(dim=-1))
    return torch.stack(candidate_errors, dim=-1).min(dim=-1).values


def symmetry_aware_rotation_error_deg(
    *,
    rotation_pred_object_to_camera: Any,
    rotation_gt_object_to_camera: Any,
    symmetry_matrices_object: Any,
) -> Any:
    """Return geodesic rotation error in degrees under finite symmetries."""

    import torch

    errors = []
    for sym_idx in range(symmetry_matrices_object.shape[0]):
        sym = symmetry_matrices_object[sym_idx].to(device=rotation_gt_object_to_camera.device, dtype=rotation_gt_object_to_camera.dtype)
        rotation_gt_equiv = rotation_gt_object_to_camera @ sym
        delta = rotation_pred_object_to_camera.transpose(-1, -2) @ rotation_gt_equiv
        trace = delta.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        cos_angle = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
        errors.append(torch.rad2deg(torch.arccos(cos_angle)))
    return torch.stack(errors, dim=-1).min(dim=-1).values
