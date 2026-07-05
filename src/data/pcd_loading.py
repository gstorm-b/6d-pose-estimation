"""Loading real depth-camera .pcd clouds into the pipeline's conventions.

Real cameras (e.g. the Basler stereo STA series) deliver organized clouds in
millimetres with the camera looking down -z and ~30-60% of pixels invalid.
Everything downstream of this module expects metres in the positive-forward
camera frame with normals matching the synthetic `normal_camera` sign. This is
the single place those conventions are applied, shared by the viewer, the CLI
scripts, and the real-eval importer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_PCD_TYPE_MAP = {
    ("F", 4): "<f4", ("F", 8): "<f8",
    ("U", 1): "<u1", ("U", 2): "<u2", ("U", 4): "<u4", ("U", 8): "<u8",
    ("I", 1): "<i1", ("I", 2): "<i2", ("I", 4): "<i4", ("I", 8): "<i8",
}


@dataclass(frozen=True)
class OrganizedCloud:
    """Raw contents of a .pcd file, before any unit/frame conversion."""

    xyz: np.ndarray                 # (N, 3) float64, N = width * height for organized clouds
    width: int
    height: int
    fields: dict[str, np.ndarray] = field(default_factory=dict)  # extra per-point fields (e.g. intensity)

    @property
    def organized(self) -> bool:
        return self.height > 1 and self.xyz.shape[0] == self.width * self.height

    def valid_mask(self) -> np.ndarray:
        """Finite, non-zero points (invalid pixels are NaN or all-zero)."""

        finite = np.isfinite(self.xyz).all(axis=1)
        nonzero = (np.abs(self.xyz) > 1e-9).any(axis=1)
        return finite & nonzero

    def pixel_grid(self) -> np.ndarray:
        """(N, 2) int64 (u, v) source pixel per point (organized clouds only)."""

        if not self.organized:
            raise ValueError("pixel_grid() requires an organized cloud")
        index = np.arange(self.xyz.shape[0], dtype=np.int64)
        return np.stack([index % self.width, index // self.width], axis=1)


@dataclass(frozen=True)
class SceneCloud:
    """A camera cloud converted to pipeline conventions (metres, positive-forward)."""

    points: np.ndarray              # (N, 3) float32 valid points only
    normals: np.ndarray | None      # (N, 3) float32, synthetic normal_camera sign
    pixels: np.ndarray | None       # (N, 2) int64 (u, v) when the source was organized
    info: dict[str, Any]            # applied conversions + source stats


def read_pcd(path: str | Path) -> OrganizedCloud:
    """Parse an ascii or binary .pcd, returning xyz plus any extra fields."""

    path = Path(path)
    with open(path, "rb") as handle:
        header: dict[str, str] = {}
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path.name}: unexpected EOF while reading header")
            text = line.decode("ascii", errors="strict").strip()
            if not text or text.startswith("#"):
                continue
            key, _, rest = text.partition(" ")
            header[key] = rest
            if key == "DATA":
                break

        fields = header["FIELDS"].split()
        sizes = [int(s) for s in header["SIZE"].split()]
        types = header["TYPE"].split()
        counts = [int(c) for c in header.get("COUNT", " ".join(["1"] * len(fields))).split()]
        width = int(header["WIDTH"])
        height = int(header["HEIGHT"])
        n_points = int(header["POINTS"])
        data_kind = header["DATA"].strip()

        dtype_fields = []
        for name, size, typ, count in zip(fields, sizes, types, counts):
            np_type = _PCD_TYPE_MAP[(typ, size)]
            dtype_fields.append((name, np_type) if count == 1 else (name, np_type, count))
        dtype = np.dtype(dtype_fields)

        if data_kind == "binary":
            raw = np.fromfile(handle, dtype=dtype, count=n_points)
        elif data_kind == "ascii":
            raw = np.loadtxt(handle, dtype=dtype, max_rows=n_points)
        else:
            raise ValueError(f"{path.name}: unsupported DATA kind '{data_kind}'")
    if raw.size != n_points:
        raise ValueError(f"{path.name}: truncated data ({raw.size}/{n_points} points read)")

    xyz = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float64)
    extra = {
        name: np.asarray(raw[name])
        for name in raw.dtype.names
        if name not in ("x", "y", "z")
    }
    return OrganizedCloud(xyz=xyz, width=width, height=height, fields=extra)


def convert_to_positive_forward(points: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert raw camera coordinates to metres, positive-forward.

    Heuristics (each recorded in the returned info dict):
    - median |p| > 10 means millimetres -> divide by 1000;
    - median z < 0 means the camera looks down -z -> negate z.
    """

    converted = np.asarray(points, dtype=np.float64).copy()
    in_mm = float(np.median(np.linalg.norm(converted, axis=1))) > 10.0
    if in_mm:
        converted /= 1000.0
    flip_z = float(np.median(converted[:, 2])) < 0.0
    if flip_z:
        converted[:, 2] *= -1.0
    return converted, {"in_mm": in_mm, "flip_z": flip_z}


def estimate_camera_normals(
    points: np.ndarray,
    *,
    radius: float = 0.02,
    max_nn: int = 30,
) -> np.ndarray:
    """Estimate per-point normals with the synthetic `normal_camera` sign.

    The synthetic normals the models trained on point away from the sensor, so
    normals are oriented toward the camera origin and then negated (this sign is
    what makes the instance model fire on real clouds - verified empirically).
    Requires open3d.
    """

    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    cloud.orient_normals_towards_camera_location(np.zeros(3))
    return (-np.asarray(cloud.normals)).astype(np.float32)


def load_scene_pcd(
    path: str | Path,
    *,
    with_normals: bool = True,
    normal_radius: float = 0.02,
    normal_max_nn: int = 30,
) -> SceneCloud:
    """Read a camera .pcd into pipeline conventions, keeping pixels when organized."""

    cloud = read_pcd(path)
    valid = cloud.valid_mask()
    if not valid.any():
        raise ValueError(f"{Path(path).name}: no valid points")
    points, conversions = convert_to_positive_forward(cloud.xyz[valid])
    pixels = cloud.pixel_grid()[valid] if cloud.organized else None
    normals = None
    if with_normals:
        normals = estimate_camera_normals(points, radius=normal_radius, max_nn=normal_max_nn)
    info = {
        **conversions,
        "source_points": int(cloud.xyz.shape[0]),
        "valid_points": int(points.shape[0]),
        "organized": cloud.organized,
        "width": cloud.width,
        "height": cloud.height,
    }
    return SceneCloud(points=points.astype(np.float32), normals=normals, pixels=pixels, info=info)
