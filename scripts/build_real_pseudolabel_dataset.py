"""Build pseudo-labeled processed samples from real-eval frames (domain adaptation).

The classical baseline (floor-plane removal + crest-split clustering) produces
reliable instance labels on *sparse* real frames - the frames where parts lie
flat and separated. This script turns those frames into processed .npz samples
in the exact PointNet++ format so they can be mixed into a synthetic dataset
and used to fine-tune the instance model on the deployment sensor's real
statistics (tote floor ribs, correlated stereo ripple, true dropout patterns).

Only feed it frames where the classical labels are correct (check the
label_preview.png it writes). Keep pile frames out - hold them out as the
honest test of what fine-tuning bought.

    python scripts/build_real_pseudolabel_dataset.py \
        --set real-data/bending_real --frames 0000 0001 0002 0005 \
        --out processed-data/real_bending_pseudolabel --copies 50
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

from src.data.depth_unprojection import unproject_depth_to_camera  # noqa: E402
from src.data.pointnet2_processing import voxel_sample_indices  # noqa: E402
from src.inference.classical_segmentation import ClassicalSegmentationConfig, segment_scene_classical  # noqa: E402

_PALETTE = np.array(
    [[230, 25, 75], [60, 180, 75], [0, 130, 200], [245, 130, 48], [145, 30, 180],
     [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212], [0, 128, 128],
     [220, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0], [170, 255, 195]],
    dtype=np.uint8,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", required=True, help="Real-eval set folder (import-pcd output).")
    parser.add_argument("--frames", nargs="+", required=True, help="Frame-name substrings to include.")
    parser.add_argument("--out", required=True, help="Output processed dataset root (train/ is created).")
    parser.add_argument("--copies", type=int, default=50, help="Sampled clouds per frame (different subsets).")
    parser.add_argument("--val-copies", type=int, default=5, help="Additional copies per frame for val/.")
    parser.add_argument("--num-points", type=int, default=16384)
    parser.add_argument("--voxel-size-m", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    set_dir = Path(args.set)
    frame_dirs = [
        p for p in sorted(set_dir.iterdir())
        if p.is_dir() and (p / "depth.npy").exists() and any(tag in p.name for tag in args.frames)
    ]
    if not frame_dirs:
        print(f"no frames matching {args.frames} under {set_dir}")
        return 1

    out_root = Path(args.out)
    (out_root / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "val").mkdir(parents=True, exist_ok=True)
    (out_root / "test").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    manifest = []
    for frame_dir in frame_dirs:
        depth = np.load(frame_dir / "depth.npy").astype(np.float32)
        intrinsics = json.loads((frame_dir / "intrinsics.json").read_text(encoding="utf-8"))
        points, pixels = unproject_depth_to_camera(depth, intrinsics)
        features = None
        if (frame_dir / "normal.npy").exists():
            normal = np.asarray(np.load(frame_dir / "normal.npy"), dtype=np.float32)
            features = normal[pixels[:, 1].astype(np.int64), pixels[:, 0].astype(np.int64)]

        result = segment_scene_classical(points, ClassicalSegmentationConfig(), pixels=pixels.astype(np.int64))
        labels = result.instance_labels
        n_instances = result.instance_count
        fg = float((labels > 0).mean())
        print(f"  {frame_dir.name}: {n_instances} pseudo instances, fg {fg:.1%}")

        # Visual check artifact: instance colors on the pixel grid.
        preview = np.zeros((int(intrinsics["height"]), int(intrinsics["width"]), 3), dtype=np.uint8)
        colors = np.full((labels.shape[0], 3), 60, dtype=np.uint8)
        fg_mask = labels > 0
        colors[fg_mask] = _PALETTE[(labels[fg_mask] - 1) % len(_PALETTE)]
        preview[pixels[:, 1].astype(int), pixels[:, 0].astype(int)] = colors
        try:
            from PIL import Image

            Image.fromarray(preview).save(frame_dir / "label_preview.png")
        except Exception:  # noqa: BLE001
            pass

        total = args.copies + args.val_copies
        for copy in range(total):
            indices = voxel_sample_indices(points, num_points=args.num_points, voxel_size_m=args.voxel_size_m, rng=rng)
            arrays = {
                "points": points[indices].astype(np.float32),
                "semantic_labels": (labels[indices] > 0).astype(np.int64),
                "instance_labels": labels[indices].astype(np.int64),
                "point_pixels": pixels[indices].astype(np.uint16),
                "raw_sample": np.asarray(f"real__{frame_dir.name}"),
            }
            if features is not None:
                arrays["features"] = features[indices].astype(np.float32)
            split = "train" if copy < args.copies else "val"
            out_path = out_root / split / f"real_{frame_dir.name}_{copy:03d}.npz"
            np.savez_compressed(out_path, **arrays)
        manifest.append({"frame": frame_dir.name, "instances": n_instances, "foreground": round(fg, 4),
                         "train_copies": args.copies, "val_copies": args.val_copies})

    (out_root / "pseudolabel_manifest.json").write_text(
        json.dumps({"set": str(set_dir), "frames": manifest, "source": "classical_segmentation"}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {sum(m['train_copies'] + m['val_copies'] for m in manifest)} samples to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
