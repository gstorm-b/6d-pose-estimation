"""Validate a raw Blender synthetic dataset from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.validation import DatasetValidationReport, STATUS_OK, validate_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", help="Raw dataset folder. Equivalent to --data.")
    parser.add_argument("--data", dest="data", help="Raw dataset folder.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT), help="Project root used to resolve model paths.")
    parser.add_argument("--json-output", help="Optional path to write a machine-readable validation report.")
    parser.add_argument("--max-issues", type=int, default=20, help="Maximum failing sample rows to print.")
    parser.add_argument(
        "--allow-legacy-filtering",
        action="store_true",
        help="Do not fail samples generated with --allow-out-of-bin-filtering.",
    )
    return parser.parse_args()


def report_to_dict(report: DatasetValidationReport) -> dict[str, Any]:
    return {
        "dataset_root": str(report.dataset_root),
        "sample_count": report.sample_count,
        "ok_count": report.ok_count,
        "status_counts": report.status_counts,
        "summary": report.summary,
        "samples": [
            {
                "sample": sample.sample,
                "status": sample.status,
                "issues": list(sample.issues),
                "object_count": sample.object_count,
                "visible_objects": sample.visible_objects,
                "visible_points": sample.visible_points,
                "out_of_bin_count": sample.out_of_bin_count,
            }
            for sample in report.samples
        ],
    }


def print_report(report: DatasetValidationReport, max_issues: int) -> None:
    print(f"Dataset: {report.dataset_root}")
    print(f"Samples: {report.sample_count}")
    print(f"OK: {report.ok_count}")
    print(f"Status counts: {json.dumps(report.status_counts, sort_keys=True)}")
    print(f"Summary: {json.dumps(report.summary, sort_keys=True)}")

    failing = [sample for sample in report.samples if sample.status != STATUS_OK]
    if not failing:
        print("Validation: OK")
        return

    print(f"Validation: FAILED ({len(failing)} failing samples)")
    for sample in failing[:max_issues]:
        issues = "; ".join(sample.issues) if sample.issues else "no issue details"
        print(f"- {sample.sample}: {sample.status} - {issues}")
    if len(failing) > max_issues:
        print(f"... {len(failing) - max_issues} more failing samples not shown")


def main() -> int:
    args = parse_args()
    dataset_value = args.data or args.dataset
    if not dataset_value:
        raise SystemExit("Provide a dataset folder with --data or as a positional argument.")

    dataset_root = Path(dataset_value)
    project_root = Path(args.project_root)
    report = validate_dataset(
        dataset_root,
        project_root=project_root,
        allow_legacy_filtering=args.allow_legacy_filtering,
    )

    print_report(report, max_issues=args.max_issues)

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report_to_dict(report), indent=2), encoding="utf-8")
        print(f"Wrote JSON report: {output_path}")

    if report.sample_count == 0:
        return 2
    return 0 if report.ok_count == report.sample_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
