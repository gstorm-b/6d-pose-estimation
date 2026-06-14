"""Audit an object's rotational symmetry from its STL and propose a symmetry.json.

Samples mesh vertices, recenters them on the bbox center, and measures a one-way
Chamfer distance between the mesh and rotated copies for candidate finite
rotations and dense continuous probes. Writes a proposed config with
`requires_human_confirmation: true` so an operator can sign it off.

Example:

    python scripts/audit_object_symmetry.py --model object-model/K41144.stl --out configs/symmetry/k41144_proposed.json
    python scripts/audit_object_symmetry.py --model object-model/bending_pipe.stl --model-scale 0.001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.pose_geometry import axis_angle_to_matrix, load_stl_vertices_m  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="STL path.")
    parser.add_argument("--model-scale", type=float, default=1.0)
    parser.add_argument("--name", default=None, help="Symmetry name (defaults to STL stem).")
    parser.add_argument("--sample-points", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--finite-tol-mm", type=float, default=1.0, help="Mean Chamfer below this accepts a finite symmetry.")
    parser.add_argument("--continuous-tol-mm", type=float, default=1.0, help="Mean Chamfer below this for a dense axis probe accepts a continuous axis.")
    parser.add_argument("--out", default=None, help="Output proposed symmetry JSON path.")
    return parser.parse_args()


def one_way_chamfer_mm(query: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Nearest-neighbor distance from each query point to the dense target set."""

    # Chunk to bound memory; target stays dense for accurate nearest neighbors.
    nearest = np.empty(query.shape[0], dtype=np.float64)
    chunk = 512
    for start in range(0, query.shape[0], chunk):
        block = query[start : start + chunk]
        diff = block[:, None, :] - target[None, :, :]
        nearest[start : start + block.shape[0]] = np.sqrt(np.sum(diff * diff, axis=2)).min(axis=1)
    return {
        "mean": float(nearest.mean()) * 1000.0,
        "p95": float(np.percentile(nearest, 95)) * 1000.0,
        "max": float(nearest.max()) * 1000.0,
    }


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    name = args.name or model_path.stem

    vertices = load_stl_vertices_m(model_path, scale=args.model_scale)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    centered = (vertices - center).astype(np.float64)
    rng = np.random.default_rng(args.seed)
    if centered.shape[0] > args.sample_points:
        idx = rng.choice(centered.shape[0], size=args.sample_points, replace=False)
        points = centered[idx]
    else:
        points = centered
    # Keep the target set dense (up to 12000 verts) so nearest neighbors are accurate.
    if centered.shape[0] > 12000:
        target = centered[rng.choice(centered.shape[0], size=12000, replace=False)]
    else:
        target = centered
    dims = (vertices.max(axis=0) - vertices.min(axis=0)) * 1000.0
    print(f"object: {name}  dims_mm=[{dims[0]:.1f}, {dims[1]:.1f}, {dims[2]:.1f}]  query_points={points.shape[0]} target_points={target.shape[0]}")

    axes = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
    finite_candidates = [("x", 180), ("y", 180), ("z", 180), ("y", 90), ("y", 270), ("y", 120), ("y", 240)]

    print("\nfinite rotation candidates (one-way Chamfer):")
    print(f"  {'rotation':<12}{'mean_mm':>10}{'p95_mm':>10}{'max_mm':>10}{'symmetric':>11}")
    accepted_finite: list[dict[str, Any]] = []
    for axis_name, angle_deg in finite_candidates:
        rot = axis_angle_to_matrix(axes[axis_name], np.radians(angle_deg))
        stats = one_way_chamfer_mm(points, target @ rot.T)
        # A real symmetry aligns ALL points, so the worst-case matters, not just the mean.
        symmetric = stats["mean"] <= args.finite_tol_mm and stats["max"] <= 3.0 * args.finite_tol_mm
        print(f"  {axis_name+' '+str(angle_deg):<12}{stats['mean']:>10.3f}{stats['p95']:>10.3f}{stats['max']:>10.3f}{str(symmetric):>11}")
        if symmetric and angle_deg in (180, 90, 120):
            accepted_finite.append({"axis": axes[axis_name], "angle_deg": angle_deg})

    print("\ncontinuous axis probe (mean over dense angles):")
    accepted_continuous: list[dict[str, Any]] = []
    for axis_name, axis in axes.items():
        worst_mean = 0.0
        worst_max = 0.0
        for angle_deg in range(30, 360, 30):
            rot = axis_angle_to_matrix(axis, np.radians(angle_deg))
            stats = one_way_chamfer_mm(points, target @ rot.T)
            worst_mean = max(worst_mean, stats["mean"])
            worst_max = max(worst_max, stats["max"])
        # Continuous requires every probe angle to align well, worst-case included.
        is_continuous = worst_mean <= args.continuous_tol_mm and worst_max <= 3.0 * args.continuous_tol_mm
        print(f"  axis {axis_name}: worst_mean_mm={worst_mean:.3f}  worst_max_mm={worst_max:.3f}  continuous={is_continuous}")
        if is_continuous:
            accepted_continuous.append({"axis": axis, "samples": 72})

    # A continuous axis subsumes finite rotations about the same axis.
    if accepted_continuous:
        cont_axes = {tuple(c["axis"]) for c in accepted_continuous}
        accepted_finite = [f for f in accepted_finite if tuple(f["axis"]) not in cont_axes]

    proposed = {
        "schema_version": 1,
        "name": name,
        "finite_rotations": accepted_finite,
        "continuous_axes": accepted_continuous,
        "requires_human_confirmation": True,
        "notes": f"Auto-proposed by audit_object_symmetry.py (finite_tol={args.finite_tol_mm} mm, continuous_tol={args.continuous_tol_mm} mm). Review before training.",
    }
    print("\nproposed symmetry:")
    print(json.dumps(proposed, indent=2))

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(proposed, indent=2), encoding="utf-8")
        print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
