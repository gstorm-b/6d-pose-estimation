"""Run PointNet++ semantic segmentation inference for one processed or raw sample."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.pointnet2_semseg_infer import (  # noqa: E402
    load_pointnet2_semseg_model,
    prepare_processed_sample,
    prepare_raw_sample,
    run_inference,
    save_inference_npz,
    save_inference_summary,
    write_prediction_ply,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a PointNet++ checkpoint.")
    parser.add_argument("--sample", required=True, help="Processed .npz file or raw sample folder.")
    parser.add_argument("--sample-type", choices=("auto", "processed", "raw"), default="auto")
    parser.add_argument("--out", required=True, help="Output folder.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ops-backend", default="pytorch3d")
    parser.add_argument("--num-points", type=int, help="Point count for raw sample inference.")
    parser.add_argument("--normalize", help="Override normalization mode.")
    parser.add_argument("--seed", type=int, help="Override inference RNG seed.")
    parser.add_argument("--save-ply", action="store_true", help="Write colored predicted-label PLY.")
    return parser.parse_args()


def infer_sample_type(path: Path, sample_type: str) -> str:
    if sample_type != "auto":
        return sample_type
    if path.is_file() and path.suffix.lower() == ".npz":
        return "processed"
    if path.is_dir() and (path / "sensor_data.npz").exists() and (path / "metadata.json").exists():
        return "raw"
    raise ValueError(f"Cannot infer sample type for {path}; pass --sample-type explicitly")


def main() -> int:
    args = parse_args()
    sample_path = Path(args.sample)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, config, device = load_pointnet2_semseg_model(
        args.checkpoint,
        device=args.device,
        ops_backend=args.ops_backend,
        seed=args.seed,
    )

    resolved_type = infer_sample_type(sample_path, args.sample_type)
    seed = int(args.seed if args.seed is not None else config.get("runtime_inference", {}).get("seed", 7))
    if resolved_type == "processed":
        inference_input = prepare_processed_sample(sample_path, config, normalize=args.normalize)
    else:
        inference_input = prepare_raw_sample(
            sample_path,
            config,
            num_points=args.num_points,
            normalize=args.normalize,
            seed=seed,
        )

    result = run_inference(model, inference_input, device=device)
    npz_path = output_dir / "prediction.npz"
    summary_path = output_dir / "summary.json"
    save_inference_npz(npz_path, result)
    save_inference_summary(summary_path, result, config)
    if args.save_ply:
        write_prediction_ply(output_dir / "prediction.ply", result.input.points_camera, result.predicted_labels)

    object_count = int((result.predicted_labels == 1).sum())
    print(
        f"[infer] sample={result.input.raw_sample} type={resolved_type} "
        f"points={result.input.points_camera.shape[0]} object_points={object_count} "
        f"elapsed={result.elapsed_seconds:.4f}s backend={result.ops_backend_effective}"
    )
    print(f"[infer] wrote: {npz_path}")
    print(f"[infer] wrote: {summary_path}")
    if args.save_ply:
        print(f"[infer] wrote: {output_dir / 'prediction.ply'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
