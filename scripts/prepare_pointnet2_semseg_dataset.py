"""Convert raw Blender synthetic data into PointNet++ semantic samples."""

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

from src.data.pointnet2_processing import (  # noqa: E402
    PointNet2ConversionConfig,
    convert_raw_sample,
    split_samples,
    summarize_converted_samples,
    write_schema,
)
from src.data.raw_dataset import REQUIRED_PREVIEW_FILES, discover_samples  # noqa: E402
from src.data.validation import validate_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, nargs="+", help="One or more raw Blender dataset roots (merged; sample names are prefixed by root).")
    parser.add_argument("--out", required=True, help="Output processed dataset root.")
    parser.add_argument("--config", help="Optional JSON/YAML config file.")
    parser.add_argument("--num-points", type=int, help="Fixed point count per processed sample.")
    parser.add_argument("--use-normals", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--object-fraction-target", type=float, help="Target fraction of sampled object points.")
    parser.add_argument("--train-ratio", type=float, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, help="Test split ratio.")
    parser.add_argument("--seed", type=int, help="Random seed for splits and sampling.")
    parser.add_argument("--skip-raw-validation", action="store_true", help="Skip raw dataset validation gate.")
    parser.add_argument(
        "--skip-invalid-samples",
        action="store_true",
        help="Skip raw sample folders missing required files when merging trusted multi-root datasets.",
    )
    return parser.parse_args()


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as exc:
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            return parse_simple_yaml_mapping(text)
        raise RuntimeError("PyYAML is required for this config format. Install pyyaml or use CLI args.") from exc
    loaded = yaml.safe_load(text)
    return loaded or {}


def parse_simple_yaml_mapping(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by this project's default configs."""

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML line: {raw_line}")
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if not raw_value:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_scalar(raw_value)
    return root


def parse_scalar(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")


def build_conversion_config(args: argparse.Namespace) -> tuple[PointNet2ConversionConfig, dict[str, Any]]:
    file_config = load_config(args.config)
    conversion_section = file_config.get("conversion", file_config.get("dataset", file_config))
    config_values: dict[str, Any] = dict(conversion_section or {})

    cli_overrides = {
        "num_points": args.num_points,
        "use_normals": args.use_normals,
        "object_fraction_target": args.object_fraction_target,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            config_values[key] = value

    config = PointNet2ConversionConfig.from_mapping(config_values)
    total_ratio = config.train_ratio + config.val_ratio + config.test_ratio
    if total_ratio <= 0:
        raise ValueError("train_ratio + val_ratio + test_ratio must be positive")
    if abs(total_ratio - 1.0) > 1e-6:
        config = PointNet2ConversionConfig(
            num_points=config.num_points,
            use_normals=config.use_normals,
            object_fraction_target=config.object_fraction_target,
            train_ratio=config.train_ratio / total_ratio,
            val_ratio=config.val_ratio / total_ratio,
            test_ratio=config.test_ratio / total_ratio,
            seed=config.seed,
        )
    return config, file_config


def ensure_output_root(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (path / split).mkdir(exist_ok=True)


def missing_required_files(sample: Any) -> list[str]:
    return [name for name in REQUIRED_PREVIEW_FILES if not (sample.path / name).exists()]


def main() -> int:
    args = parse_args()
    raw_roots = [Path(r) for r in args.raw]
    out_root = Path(args.out)
    config, source_config = build_conversion_config(args)

    from src.data.raw_dataset import RawSample

    samples: list[RawSample] = []
    skipped_samples: list[dict[str, Any]] = []
    for raw_root in raw_roots:
        if not args.skip_raw_validation:
            report = validate_dataset(raw_root, project_root=PROJECT_ROOT)
            if report.sample_count == 0:
                raise RuntimeError(f"No raw samples found in {raw_root}")
            if report.ok_count != report.sample_count:
                raise RuntimeError(
                    f"Raw validation failed for {raw_root}: {report.ok_count}/{report.sample_count} OK, "
                    f"status_counts={report.status_counts}"
                )
        root_samples = discover_samples(raw_root)
        if not root_samples:
            raise RuntimeError(f"No raw sample folders found in {raw_root}")
        # Prefix names with the root so merged samples stay unique.
        prefix = raw_root.name if len(raw_roots) > 1 else ""
        for sample in root_samples:
            missing = missing_required_files(sample)
            if missing and args.skip_invalid_samples:
                skipped_samples.append(
                    {
                        "raw_root": str(raw_root),
                        "sample": sample.name,
                        "reason": "missing_required_files",
                        "missing_files": missing,
                    }
                )
                continue
            unique_name = f"{prefix}__{sample.name}" if prefix else sample.name
            samples.append(RawSample(unique_name, sample.path))
    if not samples:
        raise RuntimeError("No valid raw samples found after filtering")
    raw_root = raw_roots[0]
    print(f"[pointnet2] merged {len(samples)} samples from {len(raw_roots)} root(s)")
    if skipped_samples:
        print(f"[pointnet2] skipped {len(skipped_samples)} invalid sample(s)")

    ensure_output_root(out_root)
    splits = split_samples(samples, config)
    split_names = {name: [sample.name for sample in split_samples_] for name, split_samples_ in splits.items()}

    rng = np.random.default_rng(config.seed)
    all_stats: list[dict[str, Any]] = []
    for split_name, split_samples_ in splits.items():
        for sample in split_samples_:
            converted = convert_raw_sample(sample, config, rng)
            out_path = out_root / split_name / f"{sample.name}.npz"
            np.savez_compressed(out_path, **converted.arrays)
            stats = {"split": split_name, "file": str(out_path.relative_to(out_root)), **converted.stats}
            all_stats.append(stats)
            print(
                f"[pointnet2] {split_name}/{sample.name}: "
                f"object={stats['sampled_object_points']} background={stats['sampled_background_points']}"
            )

    conversion_config = {
        "raw_root": str(raw_root),
        "raw_roots": [str(r) for r in raw_roots],
        "out_root": str(out_root),
        "source_config": source_config,
        "conversion": {
            "num_points": config.num_points,
            "use_normals": config.use_normals,
            "object_fraction_target": config.object_fraction_target,
            "train_ratio": config.train_ratio,
            "val_ratio": config.val_ratio,
            "test_ratio": config.test_ratio,
            "seed": config.seed,
        },
        "skipped_samples": skipped_samples,
    }
    (out_root / "conversion_config.json").write_text(json.dumps(conversion_config, indent=2), encoding="utf-8")
    stats = summarize_converted_samples(all_stats, split_names)
    (out_root / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    write_schema(out_root / "schema.md", config)
    print(f"[pointnet2] wrote processed dataset: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
