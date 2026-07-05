"""Run instance segmentation on a scene with a trained model or the classical baseline.

Segments a bin into object instances (per-point labels + clusters + centroids)
using the instance stage only - no pose. Load the model from a packaged bundle
(--sku), straight from a training checkpoint (--checkpoint), or skip learning
entirely with --method classical (floor-plane removal + Euclidean clustering,
ranked topmost-first).

Inputs (pick one):
  --depth depth.npy --intrinsics intr.json [--normals normal.npy]
  --points scene.npz        # auto-detects points / points_camera (+ features / normal_camera)
  --points scene.pcd        # real camera cloud; auto mm->m, flip-z, estimated normals

Workspace crop (recommended for real scenes):
  --crop auto               # RANSAC bin-floor plane + eroded hull
  --crop bounds --crop-bounds x0,x1,y0,y1,z0,z1   # calibrated box (metres, blank = open)

Outputs (to --out):
  instance_labels.npy       # (N,) int per-point labels (0 = background)
  instances.json            # instance count + per-instance summary + timings
  segmentation.ply          # with --save-cloud: points colored by instance
  instance_mask.png         # with --save-mask (depth/organized-pcd input): 2D instance mask

Examples:
  python scripts/run_instance_segmentation.py --sku bending_pipe \
      --points processed-data/pointnet2_semseg_bending_pipe_wave2/test/<s>.npz \
      --out out/seg --save-cloud
  python scripts/run_instance_segmentation.py --method classical \
      --points real-data/bending/<frame>-points_intensity.pcd \
      --crop auto --out out/seg --save-cloud
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# A fixed, readable palette (background is gray); cycled for many instances.
_PALETTE = np.array(
    [[230, 25, 75], [60, 180, 75], [0, 130, 200], [245, 130, 48], [145, 30, 180],
     [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212], [0, 128, 128],
     [220, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0], [170, 255, 195],
     [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128], [255, 225, 25]],
    dtype=np.uint8,
)


def _instance_colors(labels: np.ndarray) -> np.ndarray:
    colors = np.full((labels.shape[0], 3), 90, dtype=np.uint8)  # background gray
    fg = labels > 0
    colors[fg] = _PALETTE[(labels[fg] - 1) % len(_PALETTE)]
    return colors


def _load_segmenter(args: argparse.Namespace):
    from src.inference.instance_segmentation import InstanceSegmenter

    clustering_override = {
        k: v for k, v in {
            "object_probability_threshold": args.object_probability_threshold,
            "dbscan_eps_m": args.dbscan_eps_m,
            "dbscan_min_samples": args.dbscan_min_samples,
            "min_cluster_points": args.min_cluster_points,
        }.items() if v is not None
    }
    if args.checkpoint:
        segmenter = InstanceSegmenter.from_checkpoint(
            args.checkpoint, device=args.device, clustering=clustering_override or None
        )
        model_id = f"checkpoint:{Path(args.checkpoint).name}"
    else:
        from src.inference.device import resolve_device
        from src.registry.model_registry import ModelRegistry

        registry = ModelRegistry(args.registry_root)
        bundle = registry.resolve(args.sku, args.version)
        loaded = registry.load(bundle, device=resolve_device(args.device))
        segmenter = InstanceSegmenter.from_loaded_bundle(loaded)
        model_id = f"{args.sku}:{bundle.version}"
    # Apply any CLI clustering overrides on top of the loaded config
    # (the config dataclass is frozen, so build a replacement).
    if clustering_override:
        from dataclasses import replace

        segmenter.clustering = replace(segmenter.clustering, **clustering_override)
    return segmenter, model_id


def _load_scene(args: argparse.Namespace):
    """Return (points, features, pixels, scene_info) for any supported input."""

    if args.depth:
        from src.data.depth_unprojection import unproject_depth_to_camera

        depth = np.load(args.depth)
        if isinstance(depth, np.lib.npyio.NpzFile):
            depth = depth["depth_m"]
        intrinsics = json.loads(Path(args.intrinsics).read_text(encoding="utf-8"))
        points, pixels = unproject_depth_to_camera(np.asarray(depth, dtype=np.float32), intrinsics)
        features = None
        if args.normals:
            normal = np.asarray(np.load(args.normals), dtype=np.float32)
            features = normal[pixels[:, 1].astype(np.int64), pixels[:, 0].astype(np.int64)]
        return points, features, pixels.astype(np.int64), {"input": "depth"}

    points_path = Path(args.points)
    if points_path.suffix.lower() == ".pcd":
        from src.data.pcd_loading import load_scene_pcd

        scene = load_scene_pcd(points_path, with_normals=not args.no_normals)
        return scene.points, scene.normals, scene.pixels, {"input": "pcd", **scene.info}

    data = np.load(args.points)
    keys = data.files if isinstance(data, np.lib.npyio.NpzFile) else None
    if keys is None:
        return np.asarray(data, dtype=np.float32), None, None, {"input": "npy"}
    points_key = next((k for k in ("points", "points_camera") if k in keys), None)
    if points_key is None:
        raise KeyError(f"{args.points} has no 'points' or 'points_camera' array (has {keys})")
    points = np.asarray(data[points_key], dtype=np.float32)
    feature_key = next((k for k in ("features", "normal_camera", "normals") if k in keys), None)
    features = np.asarray(data[feature_key], dtype=np.float32) if feature_key else None
    if features is not None and features.shape[0] != points.shape[0]:
        features = None  # e.g. a full-frame normal map, not per-point; skip
    return points, features, None, {"input": "npz"}


def _build_crop_config(args: argparse.Namespace):
    from src.inference.workspace_crop import WorkspaceCropConfig

    if args.crop == "none":
        return WorkspaceCropConfig(mode="none")
    if args.crop == "auto":
        return WorkspaceCropConfig(mode="auto_plane")
    if args.crop == "bounds":
        if not args.crop_bounds:
            raise SystemExit("--crop bounds requires --crop-bounds x0,x1,y0,y1,z0,z1 (blank entries = open)")
        parts = [p.strip() for p in args.crop_bounds.split(",")]
        if len(parts) != 6:
            raise SystemExit("--crop-bounds needs exactly 6 comma-separated values")
        values = [float(p) if p else None for p in parts]
        return WorkspaceCropConfig(
            mode="bounds",
            x_min=values[0], x_max=values[1],
            y_min=values[2], y_max=values[3],
            z_min=values[4], z_max=values[5],
        )
    raise SystemExit(f"unknown --crop mode: {args.crop}")


def _segment(args: argparse.Namespace, segmenter):
    from src.inference.workspace_crop import crop_workspace

    points, features, pixels, scene_info = _load_scene(args)
    crop_info = None
    if args.crop != "none":
        keep, crop_info = crop_workspace(points, _build_crop_config(args))
        points = points[keep]
        features = None if features is None else features[keep]
        pixels = None if pixels is None else pixels[keep]
        if points.shape[0] == 0:
            raise SystemExit("workspace crop removed every point; check --crop settings")

    if args.method == "classical":
        from src.inference.classical_segmentation import ClassicalSegmentationConfig, segment_scene_classical

        overrides = {
            k: v for k, v in {
                "cluster_eps_m": args.dbscan_eps_m,
                "cluster_min_samples": args.dbscan_min_samples,
                "min_cluster_points": args.min_cluster_points,
            }.items() if v is not None
        }
        result = segment_scene_classical(points, ClassicalSegmentationConfig.from_mapping(overrides), pixels=pixels)
    else:
        result = segmenter.segment_points(points, features, pixels=pixels)
    return result, scene_info, crop_info


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    lines = [f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}" for p, c in zip(points, colors)]
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


def _mask_shape(args: argparse.Namespace, scene_info: dict) -> tuple[int, int] | None:
    if args.intrinsics:
        intr = json.loads(Path(args.intrinsics).read_text(encoding="utf-8"))
        return int(intr["height"]), int(intr["width"])
    if scene_info.get("organized"):
        return int(scene_info["height"]), int(scene_info["width"])
    return None


def _write_mask(path: Path, result, shape: tuple[int, int]) -> bool:
    if result.pixels is None:
        return False
    mask = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    colors = _instance_colors(result.instance_labels)
    mask[result.pixels[:, 1].astype(int), result.pixels[:, 0].astype(int)] = colors
    try:
        from PIL import Image
    except Exception:  # noqa: BLE001 - PIL optional; fall back to npy
        np.save(path.with_suffix(".npy"), mask)
        return False
    Image.fromarray(mask).save(path)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run instance segmentation with a trained model")
    parser.add_argument("--method", choices=("model", "classical"), default="model",
                        help="'model' = learned instance seg; 'classical' = plane removal + Euclidean clustering.")
    model = parser.add_mutually_exclusive_group(required=False)
    model.add_argument("--sku", help="Segment with this SKU's registry bundle.")
    model.add_argument("--checkpoint", help="Segment with an instance-seg checkpoint directly.")
    parser.add_argument("--version", default=None, help="Bundle version (default: latest).")
    parser.add_argument("--registry-root", default="models")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--depth", help="Depth .npy/.npz (metres).")
    parser.add_argument("--intrinsics", help="Intrinsics JSON (required with --depth).")
    parser.add_argument("--normals", help="Optional (H,W,3) normal map .npy for the depth input.")
    parser.add_argument("--points", help="Scene .npz with points/points_camera (+ optional features), or a camera .pcd.")
    parser.add_argument("--no-normals", action="store_true", help="Skip normal estimation for .pcd inputs.")
    parser.add_argument("--crop", choices=("none", "auto", "bounds"), default="none",
                        help="Workspace crop before segmentation (auto = RANSAC bin floor + hull).")
    parser.add_argument("--crop-bounds", default=None,
                        help="x0,x1,y0,y1,z0,z1 metres for --crop bounds (blank entry = open side).")

    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--save-cloud", action="store_true", help="Write a per-instance colored PLY.")
    parser.add_argument("--save-mask", action="store_true", help="Write a 2D instance mask (depth input only).")

    parser.add_argument("--object-probability-threshold", type=float, default=None)
    parser.add_argument("--dbscan-eps-m", type=float, default=None)
    parser.add_argument("--dbscan-min-samples", type=int, default=None)
    parser.add_argument("--min-cluster-points", type=int, default=None)
    args = parser.parse_args(argv)

    if args.depth and not args.intrinsics:
        parser.error("--intrinsics is required with --depth")
    if not args.depth and not args.points:
        parser.error("provide an input: --depth (+ --intrinsics) or --points")
    if args.method == "model" and not (args.sku or args.checkpoint):
        parser.error("--method model requires --sku or --checkpoint")

    if args.method == "classical":
        segmenter, model_id = None, "classical"
    else:
        segmenter, model_id = _load_segmenter(args)
    result, scene_info, crop_info = _segment(args, segmenter)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "instance_labels.npy", result.instance_labels)
    summary = {
        "model": model_id,
        "device": segmenter.device if segmenter is not None else "cpu",
        "scene": scene_info,
        "crop": crop_info,
        "instance_count": result.instance_count,
        "scene_points": int(result.points_camera.shape[0]),
        "source_point_count": int(result.source_point_count),
        "timings_ms": {k: round(float(v), 2) for k, v in result.timings_ms.items()},
        "clustering": None if segmenter is None else {
            "object_probability_threshold": segmenter.clustering.object_probability_threshold,
            "dbscan_eps_m": segmenter.clustering.dbscan_eps_m,
            "dbscan_min_samples": segmenter.clustering.dbscan_min_samples,
            "min_cluster_points": segmenter.clustering.min_cluster_points,
        },
        "instances": [
            {
                "instance_id": inst.instance_id,
                "point_count": inst.point_count,
                "centroid_camera": [round(float(v), 5) for v in inst.centroid_camera],
                "bbox_min_camera": [round(float(v), 5) for v in inst.bbox_min_camera],
                "bbox_max_camera": [round(float(v), 5) for v in inst.bbox_max_camera],
                "mean_object_probability": round(inst.mean_object_probability, 4),
            }
            for inst in result.instances
        ],
    }
    (out_dir / "instances.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.save_cloud:
        _write_ply(out_dir / "segmentation.ply", result.points_camera, _instance_colors(result.instance_labels))
    if args.save_mask:
        shape = _mask_shape(args, scene_info)
        if shape is None or result.pixels is None:
            print("  (--save-mask skipped: needs a --depth or organized .pcd input)")
        else:
            _write_mask(out_dir / "instance_mask.png", result, shape)

    fg = int((result.instance_labels > 0).sum())
    print(
        f"[instance-seg] {model_id}: {result.instance_count} instances, "
        f"{fg}/{result.points_camera.shape[0]} foreground points, "
        f"timings {' + '.join(f'{k} {v:.0f}ms' for k, v in result.timings_ms.items())} -> {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
