"""Turn rendered stereo pairs into SGM depth - the Route A sensor simulation.

Runs OpenCV StereoSGBM (the same algorithm family as the Basler stereo ace's
onboard SGM) on each sample's `stereo_left.png` / `stereo_right.png`, converts
the fixed-point disparity (1/16 px, like the real camera streams) to metric
depth, and rewrites `sensor_data.npz` with:

    depth_m      <- SGM depth (0 = matching failure)  [what training sees]
    depth_gt_m   <- the original raycast-perfect depth [kept for reference]
    disparity_16 <- raw int16 fixed-point disparity

Running as a separate pass (not inside Blender) means SGBM parameters can be
retuned against the real captures without re-rendering anything.

    python scripts/stereo_depth_postprocess.py --raw synthetic-data/bending_pipe_stereo_pilot
    python scripts/stereo_depth_postprocess.py --raw <root> --stats-only   # report, don't write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_matcher(fx: float, baseline_m: float, args: argparse.Namespace) -> tuple[cv2.StereoSGBM, int, int]:
    fxb = fx * baseline_m
    disp_min = int(np.floor(fxb / args.max_z))
    disp_max = int(np.ceil(fxb / args.min_z))
    num_disparities = int(np.ceil((disp_max - disp_min) / 16.0)) * 16
    block = int(args.block_size)
    matcher = cv2.StereoSGBM_create(
        minDisparity=disp_min,
        numDisparities=num_disparities,
        blockSize=block,
        P1=8 * block * block,
        P2=32 * block * block,
        disp12MaxDiff=args.disp12_max_diff,
        uniquenessRatio=args.uniqueness_ratio,
        speckleWindowSize=args.speckle_window,
        speckleRange=args.speckle_range,
        mode=cv2.STEREO_SGBM_MODE_HH if args.mode == "hh" else cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return matcher, disp_min, num_disparities


def process_sample(sample_dir: Path, args: argparse.Namespace) -> dict | None:
    left_path = sample_dir / "stereo_left.png"
    right_path = sample_dir / "stereo_right.png"
    meta_path = sample_dir / "metadata.json"
    if not (left_path.exists() and right_path.exists() and meta_path.exists()):
        return None
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    stereo_meta = metadata.get("stereo") or {}
    baseline = float(stereo_meta.get("baseline_m", args.baseline))
    fx = float(metadata["camera_intrinsics"]["fx"])

    left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
    matcher, disp_min, num_disp = build_matcher(fx, baseline, args)
    disparity = matcher.compute(left, right)  # int16, 1/16 px

    # The real camera's SGM stack rejects low-confidence speckle; without this
    # the synthetic cloud keeps garbage matches spread across the whole search
    # range (measured: 0.65 m z-span vs 0.13 m in the real clouds). filterSpeckles
    # works in fixed-point disparity units, hence the *16.
    if args.speckle_size > 0:
        cv2.filterSpeckles(disparity, 0, int(args.speckle_size), int(args.speckle_diff_px * 16))

    valid = disparity > (disp_min * 16)
    depth = np.zeros(disparity.shape, dtype=np.float32)
    with np.errstate(divide="ignore"):
        depth[valid] = fx * baseline / (disparity[valid].astype(np.float32) / 16.0)
    out_of_range = (depth < args.min_z) | (depth > args.max_z)
    depth[out_of_range] = 0.0
    valid &= ~out_of_range

    stats = {
        "sample": sample_dir.name,
        "valid_fraction": round(float(valid.mean()), 4),
        "depth_min": round(float(depth[valid].min()), 4) if valid.any() else None,
        "depth_max": round(float(depth[valid].max()), 4) if valid.any() else None,
        "disp_min": disp_min,
        "num_disparities": num_disp,
    }

    npz_path = sample_dir / "sensor_data.npz"
    if npz_path.exists():
        with np.load(npz_path) as data:
            arrays = {key: np.asarray(data[key]) for key in data.files}
        gt_depth = arrays.get("depth_gt_m", arrays["depth_m"])  # idempotent re-runs keep the raycast GT
        if gt_depth.shape != depth.shape:
            raise ValueError(
                f"{sample_dir.name}: GT depth {gt_depth.shape} != SGM depth {depth.shape}; "
                "generate with matching width/height"
            )
        # Agreement where both valid: SGM should track GT within a few mm.
        both = valid & (gt_depth > 0)
        if both.any():
            err = np.abs(depth[both] - gt_depth[both])
            stats["abs_err_mm_median"] = round(float(np.median(err)) * 1000.0, 3)
            stats["abs_err_mm_p90"] = round(float(np.percentile(err, 90)) * 1000.0, 3)
        if not args.stats_only:
            arrays["depth_gt_m"] = gt_depth
            arrays["depth_m"] = depth
            arrays["disparity_16"] = disparity.astype(np.int16)
            np.savez_compressed(npz_path, **arrays)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, help="Raw dataset root (or a single sample folder).")
    parser.add_argument("--baseline", type=float, default=0.0998, help="Fallback baseline if metadata lacks it.")
    parser.add_argument("--min-z", type=float, default=0.35, help="Nearest valid depth (m).")
    parser.add_argument("--max-z", type=float, default=1.1, help="Farthest valid depth (m).")
    parser.add_argument("--block-size", type=int, default=7)
    parser.add_argument("--uniqueness-ratio", type=int, default=10)
    parser.add_argument("--speckle-window", type=int, default=100)
    parser.add_argument("--speckle-range", type=int, default=2)
    parser.add_argument("--disp12-max-diff", type=int, default=1)
    parser.add_argument("--mode", choices=("sgbm3way", "hh"), default="hh",
                        help="hh = full 8-direction SGM (closest to the camera), sgbm3way = faster.")
    parser.add_argument("--speckle-size", type=int, default=400,
                        help="Remove connected disparity blobs smaller than this (px). 0 = off.")
    parser.add_argument("--speckle-diff-px", type=float, default=2.0,
                        help="Disparity continuity threshold (px) for the speckle filter.")
    parser.add_argument("--stats-only", action="store_true", help="Report stats without rewriting npz files.")
    args = parser.parse_args()

    root = Path(args.raw)
    sample_dirs = [root] if (root / "metadata.json").exists() else sorted(
        p for p in root.iterdir() if p.is_dir() and (p / "metadata.json").exists()
    )
    results = []
    for sample_dir in sample_dirs:
        stats = process_sample(sample_dir, args)
        if stats is None:
            continue
        results.append(stats)
        err = f", err med {stats.get('abs_err_mm_median')} mm p90 {stats.get('abs_err_mm_p90')} mm" \
            if "abs_err_mm_median" in stats else ""
        print(f"  {stats['sample']}: valid {stats['valid_fraction']:.1%}{err}")
    if not results:
        print(f"no samples with stereo pairs under {root}")
        return 1
    fractions = [r["valid_fraction"] for r in results]
    print(f"[stereo-sgm] {len(results)} samples | valid fraction mean {np.mean(fractions):.1%} "
          f"(min {min(fractions):.1%}, max {max(fractions):.1%})"
          f"{' [stats-only]' if args.stats_only else ''}")
    (root / "stereo_sgm_report.json").write_text(
        json.dumps({"args": vars(args), "samples": results}, indent=2, default=str), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
