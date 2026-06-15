"""Run instance segmentation on a scene with a trained instance model.

Segments a bin into object instances (per-point labels + clusters + centroids)
using the instance stage only - no pose. Load the model from a packaged bundle
(--sku) or straight from a training checkpoint (--checkpoint).

Inputs (pick one):
  --depth depth.npy --intrinsics intr.json [--normals normal.npy]
  --points scene.npz        # auto-detects points / points_camera (+ features / normal_camera)

Outputs (to --out):
  instance_labels.npy       # (N,) int per-point labels (0 = background)
  instances.json            # instance count + per-instance summary + timings
  segmentation.ply          # with --save-cloud: points colored by instance
  instance_mask.png         # with --save-mask (depth input only): 2D instance mask

Examples:
  python scripts/run_instance_segmentation.py --sku bending_pipe \
      --points processed-data/pointnet2_semseg_bending_pipe_wave2/test/<s>.npz \
      --out out/seg --save-cloud
  python scripts/run_instance_segmentation.py \
      --checkpoint experiments/<run>/checkpoints/best.pt \
      --depth frame/depth.npy --intrinsics frame/intrinsics.json --normals frame/normal.npy \
      --out out/seg --save-cloud --save-mask
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
    # Apply any CLI clustering overrides on top of the loaded config.
    for key, value in clustering_override.items():
        setattr(segmenter.clustering, key, value)
    return segmenter, model_id


def _segment(args: argparse.Namespace, segmenter):
    if args.depth:
        depth = np.load(args.depth)
        if isinstance(depth, np.lib.npyio.NpzFile):
            depth = depth["depth_m"]
        intrinsics = json.loads(Path(args.intrinsics).read_text(encoding="utf-8"))
        normal = np.load(args.normals) if args.normals else None
        return segmenter.segment_depth(np.asarray(depth, dtype=np.float32), intrinsics, normal)

    data = np.load(args.points)
    keys = data.files if isinstance(data, np.lib.npyio.NpzFile) else None
    if keys is None:
        return segmenter.segment_points(np.asarray(data, dtype=np.float32))
    points_key = next((k for k in ("points", "points_camera") if k in keys), None)
    if points_key is None:
        raise KeyError(f"{args.points} has no 'points' or 'points_camera' array (has {keys})")
    points = np.asarray(data[points_key], dtype=np.float32)
    feature_key = next((k for k in ("features", "normal_camera", "normals") if k in keys), None)
    features = np.asarray(data[feature_key], dtype=np.float32) if feature_key else None
    if features is not None and features.shape[0] != points.shape[0]:
        features = None  # e.g. a full-frame normal map, not per-point; skip
    return segmenter.segment_points(points, features)


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    lines = [f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}" for p, c in zip(points, colors)]
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


def _write_mask(path: Path, result, intrinsics_path: str) -> bool:
    if result.pixels is None:
        return False
    try:
        from PIL import Image
    except Exception:  # noqa: BLE001 - PIL optional; fall back to npy
        intr = json.loads(Path(intrinsics_path).read_text(encoding="utf-8"))
        mask = np.zeros((int(intr["height"]), int(intr["width"]), 3), dtype=np.uint8)
        colors = _instance_colors(result.instance_labels)
        mask[result.pixels[:, 1].astype(int), result.pixels[:, 0].astype(int)] = colors
        np.save(path.with_suffix(".npy"), mask)
        return False
    intr = json.loads(Path(intrinsics_path).read_text(encoding="utf-8"))
    mask = np.zeros((int(intr["height"]), int(intr["width"]), 3), dtype=np.uint8)
    colors = _instance_colors(result.instance_labels)
    mask[result.pixels[:, 1].astype(int), result.pixels[:, 0].astype(int)] = colors
    Image.fromarray(mask).save(path)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run instance segmentation with a trained model")
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--sku", help="Segment with this SKU's registry bundle.")
    model.add_argument("--checkpoint", help="Segment with an instance-seg checkpoint directly.")
    parser.add_argument("--version", default=None, help="Bundle version (default: latest).")
    parser.add_argument("--registry-root", default="models")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--depth", help="Depth .npy/.npz (metres).")
    parser.add_argument("--intrinsics", help="Intrinsics JSON (required with --depth).")
    parser.add_argument("--normals", help="Optional (H,W,3) normal map .npy for the depth input.")
    parser.add_argument("--points", help="Scene .npz with points/points_camera (+ optional features).")

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

    segmenter, model_id = _load_segmenter(args)
    result = _segment(args, segmenter)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "instance_labels.npy", result.instance_labels)
    summary = {
        "model": model_id,
        "device": segmenter.device,
        "instance_count": result.instance_count,
        "scene_points": int(result.points_camera.shape[0]),
        "source_point_count": int(result.source_point_count),
        "timings_ms": {k: round(float(v), 2) for k, v in result.timings_ms.items()},
        "clustering": {
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
        wrote = _write_mask(out_dir / "instance_mask.png", result, args.intrinsics or "")
        if not wrote and result.pixels is None:
            print("  (--save-mask skipped: needs a --depth input with pixels)")

    fg = int((result.instance_labels > 0).sum())
    print(
        f"[instance-seg] {model_id} device={segmenter.device}: {result.instance_count} instances, "
        f"{fg}/{result.points_camera.shape[0]} foreground points, "
        f"inference {result.timings_ms.get('inference_ms', 0):.0f} ms + clustering "
        f"{result.timings_ms.get('clustering_ms', 0):.0f} ms -> {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
