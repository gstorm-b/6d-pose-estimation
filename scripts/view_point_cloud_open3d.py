"""View generated synthetic point clouds with Open3D.

Examples:

    python scripts/view_point_cloud_open3d.py synthetic-data/K41144/sample_000001
    python scripts/view_point_cloud_open3d.py synthetic-data/K41144/sample_000001 --world --save-ply preview.ply
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample", help="Sample directory or path to sensor_data.npz")
    parser.add_argument("--world", action="store_true", help="Show points in world coordinates instead of camera coordinates")
    parser.add_argument("--voxel-size", type=float, default=0.0, help="Optional voxel downsample size in meters")
    parser.add_argument("--save-ply", default=None, help="Optional output PLY path")
    parser.add_argument("--no-window", action="store_true", help="Only print stats/save PLY without opening the Open3D window")
    return parser.parse_args()


def resolve_paths(sample_arg: str) -> tuple[Path, Path | None]:
    path = Path(sample_arg)
    if path.is_dir():
        return path / "sensor_data.npz", path / "metadata.json"
    metadata = path.with_name("metadata.json")
    return path, metadata if metadata.exists() else None


def color_by_instance(instance_ids: np.ndarray) -> np.ndarray:
    colors = np.zeros((instance_ids.shape[0], 3), dtype=np.float64)
    unique_ids = sorted(int(i) for i in np.unique(instance_ids) if i != 0)
    for instance_id in unique_ids:
        rng = np.random.default_rng(instance_id * 7919 + 17)
        colors[instance_ids == instance_id] = rng.uniform(0.25, 1.0, size=3)
    return colors


def make_point_cloud(points: np.ndarray, instance_ids: np.ndarray, voxel_size: float) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(color_by_instance(instance_ids))
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    return pcd


def print_stats(npz_path: Path, metadata_path: Path | None, points: np.ndarray, instance_ids: np.ndarray) -> None:
    print(f"sensor_data: {npz_path}")
    if metadata_path and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        print(f"metadata: {metadata_path}")
        print(f"objects in metadata: {metadata.get('object_count')}")
        settings = metadata.get("settings", {})
        if settings:
            print(
                "settings: "
                f"collision_margin={settings.get('collision_margin')} m, "
                f"collision_shape={settings.get('collision_shape')}, "
                f"drop_height_min={settings.get('drop_height_min')}, "
                f"drop_height_max={settings.get('drop_height_max')}"
            )
    unique, counts = np.unique(instance_ids, return_counts=True)
    per_instance = {int(i): int(c) for i, c in zip(unique, counts)}
    print(f"points: {points.shape[0]}")
    print(f"bounds min: {points.min(axis=0)}")
    print(f"bounds max: {points.max(axis=0)}")
    print(f"points per instance: {per_instance}")


def main() -> None:
    args = parse_args()
    npz_path, metadata_path = resolve_paths(args.sample)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    data = np.load(npz_path)
    points_key = "points_world" if args.world else "points_camera"
    points = data[points_key]
    instance_ids = data["point_instance_ids"]
    print_stats(npz_path, metadata_path, points, instance_ids)

    pcd = make_point_cloud(points, instance_ids, args.voxel_size)
    if args.save_ply:
        out_path = Path(args.save_ply)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(out_path), pcd)
        print(f"saved: {out_path}")

    if not args.no_window:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        o3d.visualization.draw_geometries([pcd, frame], window_name=f"{npz_path.parent.name} ({points_key})")


if __name__ == "__main__":
    main()
