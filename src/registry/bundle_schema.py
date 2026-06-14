"""Model bundle manifest schema for the pose-estimation backend.

A bundle is a self-contained, versioned directory holding everything needed to
serve one SKU: the instance-segmentation and pose-voting checkpoints, the object
geometry (STL + sampled model points + keypoints + diameter + symmetry), the
inference parameters, and the evaluation evidence that gated its release.

This module is pure-Python (no torch) so the schema can be validated anywhere.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

# Logical file keys every bundle must reference, resolved relative to the bundle dir.
REQUIRED_FILE_KEYS = ("instance_seg", "pose_voting", "model_points", "keypoints", "model_stl")


class BundleValidationError(ValueError):
    """Raised when a bundle manifest is malformed or its files are missing."""


@dataclass
class ModelInfo:
    """Object geometry and identity for one SKU."""

    class_name: str
    model_path: str
    model_scale: float
    symmetry_name: str
    diameter_m: float
    model_point_count: int
    keypoint_count: int

    def validate(self) -> None:
        if not self.class_name:
            raise BundleValidationError("model.class_name is empty")
        if not self.model_path:
            raise BundleValidationError("model.model_path is empty")
        if not (self.model_scale > 0.0):
            raise BundleValidationError(f"model.model_scale must be > 0, got {self.model_scale}")
        if not (self.diameter_m > 0.0):
            raise BundleValidationError(f"model.diameter_m must be > 0, got {self.diameter_m}")
        if self.model_point_count <= 0:
            raise BundleValidationError(f"model.model_point_count must be > 0, got {self.model_point_count}")
        if self.keypoint_count <= 0:
            raise BundleValidationError(f"model.keypoint_count must be > 0, got {self.keypoint_count}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelInfo":
        try:
            return cls(
                class_name=str(data["class_name"]),
                model_path=str(data["model_path"]),
                model_scale=float(data["model_scale"]),
                symmetry_name=str(data.get("symmetry_name", "none")),
                diameter_m=float(data["diameter_m"]),
                model_point_count=int(data["model_point_count"]),
                keypoint_count=int(data["keypoint_count"]),
            )
        except KeyError as exc:
            raise BundleValidationError(f"model missing field: {exc}") from exc


@dataclass
class BundleManifest:
    """Top-level manifest written as manifest.json in each bundle directory."""

    schema_version: int
    sku: str
    version: str
    created: str
    model: ModelInfo
    files: dict[str, str]
    inference: dict[str, Any] = field(default_factory=dict)
    gates: dict[str, Any] = field(default_factory=dict)
    eval_metrics: dict[str, Any] | None = None
    git_commit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["model"] = self.model.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BundleManifest":
        try:
            return cls(
                schema_version=int(data["schema_version"]),
                sku=str(data["sku"]),
                version=str(data["version"]),
                created=str(data.get("created", "")),
                model=ModelInfo.from_dict(data["model"]),
                files=dict(data["files"]),
                inference=dict(data.get("inference", {})),
                gates=dict(data.get("gates", {})),
                eval_metrics=data.get("eval_metrics"),
                git_commit=data.get("git_commit"),
            )
        except KeyError as exc:
            raise BundleValidationError(f"manifest missing field: {exc}") from exc

    def validate(self, bundle_dir: str | Path | None = None, *, check_files: bool = True) -> None:
        """Validate schema and, when `check_files`, that referenced files exist."""

        if self.schema_version != SCHEMA_VERSION:
            raise BundleValidationError(
                f"unsupported schema_version {self.schema_version}, expected {SCHEMA_VERSION}"
            )
        if not self.sku:
            raise BundleValidationError("sku is empty")
        if not self.version:
            raise BundleValidationError("version is empty")
        self.model.validate()
        for key in REQUIRED_FILE_KEYS:
            if key not in self.files or not self.files[key]:
                raise BundleValidationError(f"files is missing required key: {key}")
        if check_files:
            if bundle_dir is None:
                raise BundleValidationError("bundle_dir is required when check_files=True")
            root = Path(bundle_dir)
            for key, name in self.files.items():
                if not (root / name).exists():
                    raise BundleValidationError(f"referenced file not found: {key} -> {name}")

    def file_path(self, bundle_dir: str | Path, key: str) -> Path:
        if key not in self.files:
            raise BundleValidationError(f"unknown file key: {key}")
        return Path(bundle_dir) / self.files[key]


def write_manifest(bundle_dir: str | Path, manifest: BundleManifest) -> Path:
    manifest.validate(bundle_dir, check_files=True)
    path = Path(bundle_dir) / "manifest.json"
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_manifest(bundle_dir: str | Path, *, check_files: bool = True) -> BundleManifest:
    path = Path(bundle_dir) / "manifest.json"
    if not path.exists():
        raise BundleValidationError(f"manifest.json not found in {bundle_dir}")
    manifest = BundleManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
    manifest.validate(bundle_dir, check_files=check_files)
    return manifest
