"""Review how train-time noise/augmentation changes a processed point cloud.

Applies the exact same augmentation the DataLoader uses
(`src/data/augmentation.py`) to a processed sample, then shows the clean cloud
and the noisy cloud side by side in Open3D so the effect can be inspected.

Examples:

    python scripts/view_augmented_point_cloud.py processed-data/pointnet2_semseg_k41144/test/sample_000011.npz --config configs/train/pointnet2_semseg_k41144.yaml
    python scripts/view_augmented_point_cloud.py processed-data/pointnet2_semseg_k41144/test/sample_000011.npz --point-dropout-prob 0.1 --outlier-ratio 0.01
    python scripts/view_augmented_point_cloud.py <sample.npz> --config <train.yaml> --save-ply-clean clean.ply --save-ply-noisy noisy.ply --no-window
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augmentation import PointCloudAugmentationConfig, augment_point_cloud  # noqa: E402

# Sentinel so we can tell "flag not passed" from an explicit value.
_UNSET = object()

# RGB for label-based coloring.
_OBJECT_RGB = (0.20, 0.80, 0.35)
_BACKGROUND_RGB = (0.55, 0.55, 0.55)
_CHANGED_RGB = (0.95, 0.25, 0.20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sample", help="Path to a processed sample .npz (points/features/semantic_labels/instance_labels).")
    parser.add_argument("--config", help="Train YAML to read the augment block from. CLI flags override it.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible noise.")
    parser.add_argument("--color", choices=("semantic", "instance"), default="semantic", help="Base coloring of the clouds.")
    parser.add_argument("--highlight-changes", action="store_true", help="Tint points the augmentation moved or relabeled in red on the noisy cloud.")
    parser.add_argument("--gap", type=float, default=0.05, help="Extra gap in meters between the two clouds in the side-by-side view.")
    parser.add_argument("--save-ply-clean", help="Optional PLY path for the clean cloud.")
    parser.add_argument("--save-ply-noisy", help="Optional PLY path for the noisy cloud.")
    parser.add_argument("--no-window", action="store_true", help="Only print stats / save PLYs without opening the Open3D window.")

    overrides = parser.add_argument_group("augmentation overrides (override --config)")
    overrides.add_argument("--enabled", dest="enabled", action="store_true", default=_UNSET, help="Force-enable augmentation.")
    overrides.add_argument("--xyz-jitter-std", type=float, default=_UNSET)
    overrides.add_argument("--xyz-jitter-clip", type=float, default=_UNSET)
    overrides.add_argument("--depth-noise-std", type=float, default=_UNSET)
    overrides.add_argument("--point-dropout-prob", type=float, default=_UNSET)
    overrides.add_argument("--outlier-ratio", type=float, default=_UNSET)
    overrides.add_argument("--outlier-std", type=float, default=_UNSET)
    overrides.add_argument("--normal-jitter-std", type=float, default=_UNSET)
    overrides.add_argument("--random-z-rotation", dest="random_z_rotation", action="store_true", default=_UNSET)
    return parser.parse_args()


def find_augment_block(node: Any) -> dict[str, Any] | None:
    """Recursively locate an `augment` mapping anywhere in the loaded YAML."""

    if isinstance(node, dict):
        if isinstance(node.get("augment"), dict):
            return node["augment"]
        for value in node.values():
            found = find_augment_block(value)
            if found is not None:
                return found
    return None


def build_config(args: argparse.Namespace) -> PointCloudAugmentationConfig:
    base: dict[str, Any] = {}
    if args.config:
        import yaml  # local import so the viewer works without the train deps unless --config is used

        config_data = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        block = find_augment_block(config_data)
        if block is None:
            print(f"[warn] no `augment` block found in {args.config}; using defaults")
        else:
            base.update(block)

    override_keys = (
        "enabled",
        "xyz_jitter_std",
        "xyz_jitter_clip",
        "depth_noise_std",
        "point_dropout_prob",
        "outlier_ratio",
        "outlier_std",
        "normal_jitter_std",
        "random_z_rotation",
    )
    any_override = False
    for key in override_keys:
        value = getattr(args, key)
        if value is not _UNSET:
            base[key] = value
            any_override = True

    # Make the tool useful by default: if nothing turned augmentation on, enable it.
    if not base.get("enabled", False):
        if any_override or args.config is None:
            base["enabled"] = True
    return PointCloudAugmentationConfig.from_mapping(base)


def color_by_semantic(semantic: np.ndarray) -> np.ndarray:
    colors = np.tile(np.array(_BACKGROUND_RGB, dtype=np.float64), (semantic.shape[0], 1))
    colors[np.asarray(semantic) > 0] = _OBJECT_RGB
    return colors


def color_by_instance(instance_ids: np.ndarray) -> np.ndarray:
    colors = np.tile(np.array(_BACKGROUND_RGB, dtype=np.float64), (instance_ids.shape[0], 1))
    for instance_id in sorted(int(i) for i in np.unique(instance_ids) if int(i) != 0):
        rng = np.random.default_rng(instance_id * 7919 + 17)
        colors[np.asarray(instance_ids) == instance_id] = rng.uniform(0.25, 1.0, size=3)
    return colors


def make_point_cloud(points: np.ndarray, colors: np.ndarray, offset_x: float = 0.0) -> o3d.geometry.PointCloud:
    shifted = points.astype(np.float64).copy()
    shifted[:, 0] += offset_x
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(shifted)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    return pcd


def print_stats(
    config: PointCloudAugmentationConfig,
    clean_points: np.ndarray,
    noisy_points: np.ndarray,
    clean_semantic: np.ndarray,
    noisy_semantic: np.ndarray,
    moved_mask: np.ndarray,
) -> None:
    n = clean_points.shape[0]
    displacement = np.linalg.norm(noisy_points - clean_points, axis=1)
    label_changed = int(np.sum(np.asarray(clean_semantic) != np.asarray(noisy_semantic)))
    became_background = int(np.sum((np.asarray(clean_semantic) > 0) & (np.asarray(noisy_semantic) == 0)))
    print("augment config:")
    for field, value in config.__dict__.items():
        print(f"  {field}: {value}")
    print(f"points: {n}")
    print(f"points moved (> 1e-6 m): {int(moved_mask.sum())} ({100.0 * moved_mask.mean():.1f}%)")
    print(f"mean displacement (moved): {float(displacement[moved_mask].mean()) * 1000.0 if moved_mask.any() else 0.0:.3f} mm")
    print(f"max displacement: {float(displacement.max()) * 1000.0:.3f} mm")
    print(f"semantic labels changed: {label_changed} (object->background: {became_background})")


def main() -> None:
    args = parse_args()
    npz_path = Path(args.sample)
    if not npz_path.is_file():
        raise FileNotFoundError(f"Expected a processed sample .npz file: {npz_path}")

    with np.load(npz_path) as data:
        if "points" not in data.files:
            raise KeyError(f"{npz_path} has no `points` array; pass a processed dataset .npz, not a raw sensor_data.npz")
        points = np.asarray(data["points"], dtype=np.float32)
        semantic = np.asarray(data["semantic_labels"], dtype=np.int64)
        instance = np.asarray(data["instance_labels"], dtype=np.int64)
        features = np.asarray(data["features"], dtype=np.float32) if "features" in data.files else None

    config = build_config(args)
    rng = np.random.default_rng(args.seed)
    noisy_points, _noisy_features, noisy_semantic, noisy_instance = augment_point_cloud(
        points, features, semantic, instance, config, rng
    )

    moved_mask = np.linalg.norm(noisy_points - points, axis=1) > 1e-6
    print_stats(config, points, noisy_points, semantic, noisy_semantic, moved_mask)

    if args.color == "instance":
        clean_colors = color_by_instance(instance)
        noisy_colors = color_by_instance(noisy_instance)
    else:
        clean_colors = color_by_semantic(semantic)
        noisy_colors = color_by_semantic(noisy_semantic)
    if args.highlight_changes:
        changed = moved_mask | (np.asarray(semantic) != np.asarray(noisy_semantic))
        noisy_colors[changed] = _CHANGED_RGB

    span_x = float(points[:, 0].max() - points[:, 0].min())
    offset_x = span_x + max(0.0, args.gap)
    clean_pcd = make_point_cloud(points, clean_colors, offset_x=0.0)
    noisy_pcd = make_point_cloud(noisy_points, noisy_colors, offset_x=offset_x)

    for save_arg, pcd, label in ((args.save_ply_clean, clean_pcd, "clean"), (args.save_ply_noisy, noisy_pcd, "noisy")):
        if save_arg:
            out_path = Path(save_arg)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            o3d.io.write_point_cloud(str(out_path), pcd)
            print(f"saved {label}: {out_path}")

    if not args.no_window:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        frame_noisy = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05).translate((offset_x, 0.0, 0.0))
        print("\nleft = clean, right = noisy. Close the window to exit.")
        o3d.visualization.draw_geometries(
            [clean_pcd, noisy_pcd, frame, frame_noisy],
            window_name=f"{npz_path.name}  clean | noisy",
        )


if __name__ == "__main__":
    main()
