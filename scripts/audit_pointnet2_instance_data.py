"""Audit processed PointNet++ data for instance-segmentation training."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REQUIRED_ARRAYS = {
    "points",
    "semantic_labels",
    "instance_labels",
    "point_pixels",
    "raw_sample",
}
OPTIONAL_ARRAYS = {"features"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class AuditThresholds:
    min_points_per_instance: int = 64
    ignore_gt_instances_below_points: int = 32
    min_large_instance_fraction: float = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Processed PointNet++ dataset root.")
    parser.add_argument("--out", required=True, help="Output audit folder.")
    parser.add_argument("--model-name", help="Optional object/model name to record in the report.")
    parser.add_argument("--model-path", help="Optional STL/CAD path to verify and record.")
    parser.add_argument("--min-points-per-instance", type=int, default=64)
    parser.add_argument("--ignore-gt-instances-below-points", type=int, default=32)
    parser.add_argument("--min-large-instance-fraction", type=float, default=0.95)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data)
    out_root = Path(args.out)
    thresholds = AuditThresholds(
        min_points_per_instance=args.min_points_per_instance,
        ignore_gt_instances_below_points=args.ignore_gt_instances_below_points,
        min_large_instance_fraction=args.min_large_instance_fraction,
    )

    report = audit_dataset(
        data_root=data_root,
        thresholds=thresholds,
        model_name=args.model_name,
        model_path=Path(args.model_path) if args.model_path else None,
    )
    out_root.mkdir(parents=True, exist_ok=True)
    report_path = out_root / "audit_report.json"
    summary_path = out_root / "audit_summary.md"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_path.write_text(format_markdown_summary(report), encoding="utf-8")

    status = report["status"]
    totals = report["totals"]
    print(
        f"[instance-audit] status={status} samples={totals['sample_count']} "
        f"instances={totals['instance_count']} small_lt_{thresholds.min_points_per_instance}="
        f"{totals['instances_below_min_points']}"
    )
    print(f"[instance-audit] wrote {report_path}")
    print(f"[instance-audit] wrote {summary_path}")
    return 0 if status != "fail" else 1


def audit_dataset(
    *,
    data_root: Path,
    thresholds: AuditThresholds,
    model_name: str | None,
    model_path: Path | None,
) -> dict[str, Any]:
    if not data_root.exists():
        raise FileNotFoundError(f"Processed dataset root does not exist: {data_root}")

    model_info = build_model_info(model_name, model_path)
    sample_reports: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    split_counts: dict[str, int] = {}

    for split in SPLITS:
        split_dir = data_root / split
        if not split_dir.exists():
            errors.append(f"Missing split folder: {split_dir}")
            split_counts[split] = 0
            continue
        sample_paths = sorted(split_dir.glob("*.npz"))
        split_counts[split] = len(sample_paths)
        if not sample_paths:
            warnings.append(f"Split has no samples: {split}")
            continue
        for sample_path in sample_paths:
            sample_reports.append(audit_sample(sample_path, split, thresholds))

    for sample_report in sample_reports:
        errors.extend(sample_report["errors"])
        warnings.extend(sample_report["warnings"])

    totals = summarize_samples(sample_reports, thresholds)
    if totals["instance_count"] > 0:
        large_fraction = totals["instances_at_or_above_min_points"] / totals["instance_count"]
        if large_fraction < thresholds.min_large_instance_fraction:
            warnings.append(
                "Large-instance fraction below target: "
                f"{large_fraction:.4f} < {thresholds.min_large_instance_fraction:.4f}"
            )

    status = "fail" if errors else ("warn" if warnings else "pass")
    return {
        "status": status,
        "data_root": str(data_root),
        "model": model_info,
        "thresholds": {
            "min_points_per_instance": thresholds.min_points_per_instance,
            "ignore_gt_instances_below_points": thresholds.ignore_gt_instances_below_points,
            "min_large_instance_fraction": thresholds.min_large_instance_fraction,
        },
        "split_counts": split_counts,
        "totals": totals,
        "errors": errors,
        "warnings": warnings,
        "samples": sample_reports,
    }


def build_model_info(model_name: str | None, model_path: Path | None) -> dict[str, Any]:
    info: dict[str, Any] = {"name": model_name, "path": str(model_path) if model_path else None}
    if model_path is not None:
        info["exists"] = model_path.exists()
        info["size_bytes"] = model_path.stat().st_size if model_path.exists() else None
    return info


def audit_sample(sample_path: Path, split: str, thresholds: AuditThresholds) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    with np.load(sample_path, allow_pickle=False) as data:
        files = set(data.files)
        missing = sorted(REQUIRED_ARRAYS - files)
        unknown = sorted(files - REQUIRED_ARRAYS - OPTIONAL_ARRAYS)
        if missing:
            errors.append(f"{sample_path}: missing arrays {missing}")
        if unknown:
            warnings.append(f"{sample_path}: unknown arrays {unknown}")

        if missing:
            return empty_sample_report(sample_path, split, errors, warnings)

        points = np.asarray(data["points"])
        semantic_labels = np.asarray(data["semantic_labels"])
        instance_labels = np.asarray(data["instance_labels"])
        point_pixels = np.asarray(data["point_pixels"])
        features = np.asarray(data["features"]) if "features" in files else None
        raw_sample = str(np.asarray(data["raw_sample"]).item())

    if points.ndim != 2 or points.shape[1] != 3:
        errors.append(f"{sample_path}: points must have shape (N, 3), got {points.shape}")
    if semantic_labels.ndim != 1:
        errors.append(f"{sample_path}: semantic_labels must have shape (N,), got {semantic_labels.shape}")
    if instance_labels.ndim != 1:
        errors.append(f"{sample_path}: instance_labels must have shape (N,), got {instance_labels.shape}")
    if point_pixels.ndim != 2 or point_pixels.shape[1] != 2:
        errors.append(f"{sample_path}: point_pixels must have shape (N, 2), got {point_pixels.shape}")
    if features is not None and (features.ndim != 2 or features.shape[0] != points.shape[0]):
        errors.append(f"{sample_path}: features must have shape (N, C), got {features.shape}")

    n_points = int(points.shape[0]) if points.ndim >= 1 else 0
    for name, array in {
        "semantic_labels": semantic_labels,
        "instance_labels": instance_labels,
        "point_pixels": point_pixels,
    }.items():
        if array.shape[0] != n_points:
            errors.append(f"{sample_path}: {name} first dimension {array.shape[0]} != points {n_points}")

    if not np.issubdtype(points.dtype, np.floating):
        warnings.append(f"{sample_path}: points dtype is {points.dtype}, expected floating")
    if not np.issubdtype(semantic_labels.dtype, np.integer):
        errors.append(f"{sample_path}: semantic_labels dtype is {semantic_labels.dtype}, expected integer")
    if not np.issubdtype(instance_labels.dtype, np.integer):
        errors.append(f"{sample_path}: instance_labels dtype is {instance_labels.dtype}, expected integer")
    if not np.all(np.isfinite(points)):
        errors.append(f"{sample_path}: points contain non-finite values")
    if features is not None and not np.all(np.isfinite(features)):
        errors.append(f"{sample_path}: features contain non-finite values")

    semantic_unique = sorted(int(v) for v in np.unique(semantic_labels)) if semantic_labels.size else []
    instance_unique = sorted(int(v) for v in np.unique(instance_labels)) if instance_labels.size else []
    object_mask = semantic_labels == 1
    background_mask = semantic_labels == 0
    invalid_semantic_values = [value for value in semantic_unique if value not in {0, 1}]
    if invalid_semantic_values:
        errors.append(f"{sample_path}: invalid semantic labels {invalid_semantic_values}")
    if np.any(object_mask & (instance_labels <= 0)):
        errors.append(f"{sample_path}: object points contain non-positive instance ids")
    if np.any(background_mask & (instance_labels != 0)):
        errors.append(f"{sample_path}: background points contain positive instance ids")
    if not np.any(object_mask):
        errors.append(f"{sample_path}: no object points")
    if not np.any(background_mask):
        warnings.append(f"{sample_path}: no background points")

    instance_counts = count_instances(instance_labels)
    small_counts = [count for count in instance_counts.values() if count < thresholds.min_points_per_instance]
    ignored_counts = [count for count in instance_counts.values() if count < thresholds.ignore_gt_instances_below_points]
    if ignored_counts:
        warnings.append(
            f"{sample_path}: {len(ignored_counts)} instances below ignore threshold "
            f"{thresholds.ignore_gt_instances_below_points}"
        )

    return {
        "split": split,
        "file": str(sample_path),
        "raw_sample": raw_sample,
        "point_count": n_points,
        "feature_dim": int(features.shape[1]) if features is not None and features.ndim == 2 else None,
        "semantic_labels": semantic_unique,
        "instance_count": len(instance_counts),
        "object_point_count": int(np.count_nonzero(object_mask)),
        "background_point_count": int(np.count_nonzero(background_mask)),
        "instance_point_counts": {str(key): int(value) for key, value in instance_counts.items()},
        "instance_point_count_stats": summarize_numeric(list(instance_counts.values())),
        "instances_below_min_points": len(small_counts),
        "instances_below_ignore_threshold": len(ignored_counts),
        "errors": errors,
        "warnings": warnings,
    }


def empty_sample_report(sample_path: Path, split: str, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    return {
        "split": split,
        "file": str(sample_path),
        "raw_sample": None,
        "point_count": 0,
        "feature_dim": None,
        "semantic_labels": [],
        "instance_count": 0,
        "object_point_count": 0,
        "background_point_count": 0,
        "instance_point_counts": {},
        "instance_point_count_stats": summarize_numeric([]),
        "instances_below_min_points": 0,
        "instances_below_ignore_threshold": 0,
        "errors": errors,
        "warnings": warnings,
    }


def count_instances(instance_labels: np.ndarray) -> dict[int, int]:
    positive = instance_labels[instance_labels > 0]
    if positive.size == 0:
        return {}
    ids, counts = np.unique(positive, return_counts=True)
    return {int(instance_id): int(count) for instance_id, count in zip(ids, counts, strict=True)}


def summarize_samples(sample_reports: list[dict[str, Any]], thresholds: AuditThresholds) -> dict[str, Any]:
    all_instance_counts: list[int] = []
    for sample in sample_reports:
        all_instance_counts.extend(int(v) for v in sample["instance_point_counts"].values())

    sample_point_counts = [int(sample["point_count"]) for sample in sample_reports]
    instance_counts_per_sample = [int(sample["instance_count"]) for sample in sample_reports]
    object_points_per_sample = [int(sample["object_point_count"]) for sample in sample_reports]
    background_points_per_sample = [int(sample["background_point_count"]) for sample in sample_reports]
    below_min = [count for count in all_instance_counts if count < thresholds.min_points_per_instance]
    below_ignore = [count for count in all_instance_counts if count < thresholds.ignore_gt_instances_below_points]

    return {
        "sample_count": len(sample_reports),
        "instance_count": len(all_instance_counts),
        "sample_point_count_stats": summarize_numeric(sample_point_counts),
        "instances_per_sample_stats": summarize_numeric(instance_counts_per_sample),
        "object_points_per_sample_stats": summarize_numeric(object_points_per_sample),
        "background_points_per_sample_stats": summarize_numeric(background_points_per_sample),
        "instance_point_count_stats": summarize_numeric(all_instance_counts),
        "instances_below_min_points": len(below_min),
        "instances_at_or_above_min_points": len(all_instance_counts) - len(below_min),
        "instances_below_ignore_threshold": len(below_ignore),
    }


def summarize_numeric(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {
            "min": None,
            "p10": None,
            "median": None,
            "p90": None,
            "max": None,
            "mean": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": int(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": int(np.max(array)),
        "mean": float(np.mean(array)),
    }


def format_markdown_summary(report: dict[str, Any]) -> str:
    totals = report["totals"]
    thresholds = report["thresholds"]
    instance_stats = totals["instance_point_count_stats"]
    lines = [
        "# PointNet++ Instance Dataset Audit",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Dataset",
        "",
        f"- Data root: `{report['data_root']}`",
        f"- Model name: `{report['model'].get('name')}`",
        f"- Model path: `{report['model'].get('path')}`",
        f"- Model path exists: `{report['model'].get('exists')}`",
        f"- Split counts: `{report['split_counts']}`",
        "",
        "## Thresholds",
        "",
        f"- min_points_per_instance: `{thresholds['min_points_per_instance']}`",
        f"- ignore_gt_instances_below_points: `{thresholds['ignore_gt_instances_below_points']}`",
        f"- min_large_instance_fraction: `{thresholds['min_large_instance_fraction']}`",
        "",
        "## Totals",
        "",
        f"- Samples: `{totals['sample_count']}`",
        f"- Instances: `{totals['instance_count']}`",
        f"- Instances below min points: `{totals['instances_below_min_points']}`",
        f"- Instances below ignore threshold: `{totals['instances_below_ignore_threshold']}`",
        f"- Instance point-count stats: `{instance_stats}`",
        "",
        "## Errors",
        "",
    ]
    lines.extend(f"- {error}" for error in report["errors"])
    if not report["errors"]:
        lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in report["warnings"])
    if not report["warnings"]:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
