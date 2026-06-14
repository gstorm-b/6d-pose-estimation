"""Package a per-SKU model bundle for the pose-estimation backend registry.

Reads the object geometry from the pose checkpoint's embedded config, samples
model points/keypoints from the STL, copies the instance and pose checkpoints,
and writes a self-contained bundle under models/<sku>/<version>/.

Example:

    python scripts/package_model_bundle.py --sku K41144 \
        --instance-checkpoint experiments/pointnet2_instance_k41144_20260607_165729/checkpoints/best.pt \
        --pose-checkpoint experiments/pointnet2_pose_k41144_20260608_023605/checkpoints/best_add.pt \
        --dbscan-eps-m 0.004 --min-cluster-points 96 --object-probability-threshold 0.5
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pose_crop_dataset import farthest_point_keypoints  # noqa: E402
from src.registry.bundle_schema import BundleManifest, ModelInfo, write_manifest  # noqa: E402
from src.training.checkpoint import load_checkpoint  # noqa: E402
from src.training.pose_geometry import (  # noqa: E402
    load_stl_vertices_m,
    model_diameter_m,
    sample_model_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sku", required=True, help="SKU identifier, e.g. K41144.")
    parser.add_argument("--version", default="v1", help="Bundle version directory name. Default v1.")
    parser.add_argument("--instance-checkpoint", required=True, help="Instance-segmentation checkpoint (best.pt).")
    parser.add_argument("--pose-checkpoint", required=True, help="Pose-voting checkpoint (best_add.pt).")
    parser.add_argument("--root", default="models", help="Registry root directory. Default models.")
    parser.add_argument("--object-probability-threshold", type=float, default=0.5)
    parser.add_argument("--dbscan-eps-m", type=float, default=0.005)
    parser.add_argument("--dbscan-min-samples", type=int, default=8)
    parser.add_argument("--min-cluster-points", type=int, default=64)
    parser.add_argument("--num-points", type=int, default=16384, help="Scene points the instance model expects.")
    parser.add_argument("--refinement-preset", default="configs/eval/pose_refinement_defaults.json", help="Optional refinement preset JSON.")
    parser.add_argument("--eval-metrics", default=None, help="Optional eval evidence JSON to embed.")
    parser.add_argument("--gates", default=None, help="Optional gates JSON to embed.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing bundle version directory.")
    return parser.parse_args()


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - commit is best-effort metadata
        return None


def load_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    if not candidate.exists():
        print(f"[warn] file not found, skipping: {path}")
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def resolve_path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)


def main() -> int:
    args = parse_args()
    pose_ckpt_path = resolve_path(args.pose_checkpoint)
    instance_ckpt_path = resolve_path(args.instance_checkpoint)
    if not pose_ckpt_path.exists():
        raise FileNotFoundError(pose_ckpt_path)
    if not instance_ckpt_path.exists():
        raise FileNotFoundError(instance_ckpt_path)

    pose_ckpt = load_checkpoint(pose_ckpt_path, map_location="cpu")
    pose_config = dict(pose_ckpt.get("config", {}))
    object_config = dict(pose_config.get("object", {}))
    model_config = dict(pose_config.get("model", {}))
    if "model_path" not in object_config:
        raise KeyError("pose checkpoint config has no object.model_path; cannot package bundle")

    model_path = str(object_config["model_path"])
    model_scale = float(object_config.get("model_scale", 1.0))
    symmetry_name = str(object_config.get("symmetry_name", "none"))
    model_point_count = int(object_config.get("model_point_count", 1024))
    keypoint_count = int(model_config.get("keypoint_count", object_config.get("keypoint_count", 8)))
    seed = int(pose_config.get("seed", pose_config.get("train", {}).get("seed", 7)))
    class_name = str(object_config.get("class_name", args.sku))

    stl_path = resolve_path(model_path)
    if not stl_path.exists():
        raise FileNotFoundError(f"object STL not found: {stl_path}")
    vertices_m = load_stl_vertices_m(stl_path, scale=model_scale)
    diameter = float(model_diameter_m(vertices_m))
    model_points = sample_model_points(vertices_m, model_point_count, seed=seed).astype(np.float32)
    keypoints = farthest_point_keypoints(vertices_m, keypoint_count, seed=seed).astype(np.float32)

    bundle_dir = resolve_path(args.root) / args.sku / args.version
    if bundle_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"bundle exists (use --overwrite): {bundle_dir}")
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(instance_ckpt_path, bundle_dir / "instance_seg.pt")
    shutil.copy2(pose_ckpt_path, bundle_dir / "pose_voting.pt")
    shutil.copy2(stl_path, bundle_dir / "model.stl")
    np.save(bundle_dir / "model_points.npy", model_points)
    np.save(bundle_dir / "keypoints.npy", keypoints)

    inference = {
        "num_points": int(args.num_points),
        "object_probability_threshold": float(args.object_probability_threshold),
        "dbscan_eps_m": float(args.dbscan_eps_m),
        "dbscan_min_samples": int(args.dbscan_min_samples),
        "min_cluster_points": int(args.min_cluster_points),
        "refinement": load_json(args.refinement_preset) or {},
    }

    manifest = BundleManifest(
        schema_version=1,
        sku=args.sku,
        version=args.version,
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        git_commit=git_commit(),
        model=ModelInfo(
            class_name=class_name,
            model_path=model_path,
            model_scale=model_scale,
            symmetry_name=symmetry_name,
            diameter_m=diameter,
            model_point_count=model_point_count,
            keypoint_count=keypoint_count,
        ),
        files={
            "instance_seg": "instance_seg.pt",
            "pose_voting": "pose_voting.pt",
            "model_points": "model_points.npy",
            "keypoints": "keypoints.npy",
            "model_stl": "model.stl",
        },
        inference=inference,
        gates=load_json(args.gates) or {},
        eval_metrics=load_json(args.eval_metrics),
    )
    write_manifest(bundle_dir, manifest)
    print(f"[bundle] wrote {bundle_dir}")
    print(f"[bundle] class={class_name} symmetry={symmetry_name} diameter={diameter * 1000:.1f} mm "
          f"model_points={model_points.shape[0]} keypoints={keypoints.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
