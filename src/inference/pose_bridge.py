"""Bridge instance segmentation outputs into pose-stage crop inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PoseInstanceCrop:
    """One object instance crop in camera coordinates."""

    sample_name: str
    source: str
    instance_id: int
    point_indices: np.ndarray
    points_camera: np.ndarray
    centroid_camera: np.ndarray
    bbox_min_camera: np.ndarray
    bbox_max_camera: np.ndarray
    features: np.ndarray | None = None
    matched_gt_instance_id: int | None = None
    matched_gt_iou: float | None = None
    object_to_camera: np.ndarray | None = None
    class_name: str | None = None
    model_path: str | None = None

    @property
    def point_count(self) -> int:
        return int(self.point_indices.shape[0])


def load_raw_metadata(raw_root: str | Path | None, raw_sample: str | None) -> dict[str, Any] | None:
    """Load raw synthetic metadata for a processed sample when available."""

    if raw_root is None or not raw_sample:
        return None
    metadata_path = Path(raw_root) / raw_sample / "metadata.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def build_pose_instance_crops(
    *,
    sample_name: str,
    source: str,
    points_camera: np.ndarray,
    instance_labels: np.ndarray,
    point_features: np.ndarray | None = None,
    target_instance_labels: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
    min_points: int = 1,
) -> list[PoseInstanceCrop]:
    """Create pose-stage crops from either predicted or GT instance labels.

    `source` should be `predicted` or `gt`. Predicted crops can be matched back
    to GT labels to attach synthetic `object_to_camera` supervision.
    """

    points = np.asarray(points_camera, dtype=np.float32)
    labels = np.asarray(instance_labels, dtype=np.int64).reshape(-1)
    if points.shape[0] != labels.shape[0]:
        raise ValueError(f"points and instance_labels length mismatch: {points.shape[0]} vs {labels.shape[0]}")
    features = None if point_features is None else np.asarray(point_features, dtype=np.float32)
    if features is not None and features.shape[0] != points.shape[0]:
        raise ValueError(f"point_features length mismatch: {features.shape[0]} vs {points.shape[0]}")
    targets = None if target_instance_labels is None else np.asarray(target_instance_labels, dtype=np.int64).reshape(-1)
    if targets is not None and targets.shape != labels.shape:
        raise ValueError(f"target_instance_labels shape mismatch: {targets.shape} vs {labels.shape}")

    metadata_by_id = _metadata_instances_by_id(metadata)
    crops: list[PoseInstanceCrop] = []
    for instance_id in sorted(int(value) for value in np.unique(labels) if int(value) > 0):
        indices = np.flatnonzero(labels == instance_id).astype(np.int64)
        if int(indices.shape[0]) < int(min_points):
            continue
        crop_points = points[indices]
        crop_features = None if features is None else features[indices]
        matched_gt_id, matched_iou = _match_crop_to_gt(labels, targets, instance_id)
        pose_gt_id = instance_id if source == "gt" else matched_gt_id
        instance_metadata = metadata_by_id.get(int(pose_gt_id)) if pose_gt_id is not None else None
        object_to_camera = _matrix_or_none(instance_metadata.get("object_to_camera")) if instance_metadata else None
        crops.append(
            PoseInstanceCrop(
                sample_name=sample_name,
                source=source,
                instance_id=instance_id,
                point_indices=indices,
                points_camera=crop_points.astype(np.float32),
                features=None if crop_features is None else crop_features.astype(np.float32),
                centroid_camera=crop_points.mean(axis=0).astype(np.float32),
                bbox_min_camera=crop_points.min(axis=0).astype(np.float32),
                bbox_max_camera=crop_points.max(axis=0).astype(np.float32),
                matched_gt_instance_id=pose_gt_id,
                matched_gt_iou=matched_iou,
                object_to_camera=object_to_camera,
                class_name=str(metadata.get("class_name")) if metadata and metadata.get("class_name") is not None else None,
                model_path=str(metadata.get("model_path")) if metadata and metadata.get("model_path") is not None else None,
            )
        )
    return crops


def save_pose_instance_crops_npz(path: str | Path, crops: list[PoseInstanceCrop]) -> None:
    """Save variable-length crops as a flat NPZ suitable for pose-stage loading."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not crops:
        np.savez_compressed(
            output_path,
            crop_points_camera=np.zeros((0, 3), dtype=np.float32),
            crop_features=np.zeros((0, 0), dtype=np.float32),
            crop_point_indices=np.zeros((0,), dtype=np.int64),
            crop_offsets=np.zeros((0, 2), dtype=np.int64),
            crop_instance_ids=np.zeros((0,), dtype=np.int64),
            crop_point_counts=np.zeros((0,), dtype=np.int64),
            crop_centroids_camera=np.zeros((0, 3), dtype=np.float32),
            crop_bbox_min_camera=np.zeros((0, 3), dtype=np.float32),
            crop_bbox_max_camera=np.zeros((0, 3), dtype=np.float32),
            crop_matched_gt_instance_ids=np.zeros((0,), dtype=np.int64),
            crop_matched_gt_ious=np.zeros((0,), dtype=np.float32),
            crop_object_to_camera=np.zeros((0, 4, 4), dtype=np.float32),
            sample_name=np.asarray("", dtype="U1"),
            source=np.asarray("", dtype="U1"),
        )
        return

    point_blocks = [crop.points_camera.astype(np.float32) for crop in crops]
    feature_blocks = [crop.features.astype(np.float32) for crop in crops if crop.features is not None]
    feature_dim = feature_blocks[0].shape[1] if feature_blocks else 0
    if feature_blocks and len(feature_blocks) != len(crops):
        raise ValueError("Either all pose crops must include features or none should include features")
    index_blocks = [crop.point_indices.astype(np.int64) for crop in crops]
    offsets = []
    cursor = 0
    for crop in crops:
        start = cursor
        cursor += crop.point_count
        offsets.append((start, cursor))
    object_to_camera = np.stack(
        [
            crop.object_to_camera.astype(np.float32)
            if crop.object_to_camera is not None
            else np.full((4, 4), np.nan, dtype=np.float32)
            for crop in crops
        ],
        axis=0,
    )
    np.savez_compressed(
        output_path,
        crop_points_camera=np.concatenate(point_blocks, axis=0),
        crop_features=np.concatenate(feature_blocks, axis=0) if feature_blocks else np.zeros((cursor, feature_dim), dtype=np.float32),
        crop_point_indices=np.concatenate(index_blocks, axis=0),
        crop_offsets=np.asarray(offsets, dtype=np.int64),
        crop_instance_ids=np.asarray([crop.instance_id for crop in crops], dtype=np.int64),
        crop_point_counts=np.asarray([crop.point_count for crop in crops], dtype=np.int64),
        crop_centroids_camera=np.stack([crop.centroid_camera for crop in crops], axis=0).astype(np.float32),
        crop_bbox_min_camera=np.stack([crop.bbox_min_camera for crop in crops], axis=0).astype(np.float32),
        crop_bbox_max_camera=np.stack([crop.bbox_max_camera for crop in crops], axis=0).astype(np.float32),
        crop_matched_gt_instance_ids=np.asarray(
            [crop.matched_gt_instance_id if crop.matched_gt_instance_id is not None else -1 for crop in crops],
            dtype=np.int64,
        ),
        crop_matched_gt_ious=np.asarray(
            [crop.matched_gt_iou if crop.matched_gt_iou is not None else np.nan for crop in crops],
            dtype=np.float32,
        ),
        crop_object_to_camera=object_to_camera,
        sample_name=np.asarray(crops[0].sample_name),
        source=np.asarray(crops[0].source),
    )


def pose_crop_summary(crop: PoseInstanceCrop) -> dict[str, Any]:
    """Return a JSON-safe crop summary."""

    return {
        "sample_name": crop.sample_name,
        "source": crop.source,
        "instance_id": crop.instance_id,
        "point_count": crop.point_count,
        "centroid_camera": crop.centroid_camera.tolist(),
        "bbox_min_camera": crop.bbox_min_camera.tolist(),
        "bbox_max_camera": crop.bbox_max_camera.tolist(),
        "matched_gt_instance_id": crop.matched_gt_instance_id,
        "matched_gt_iou": crop.matched_gt_iou,
        "has_object_to_camera": crop.object_to_camera is not None,
        "class_name": crop.class_name,
        "model_path": crop.model_path,
    }


def _metadata_instances_by_id(metadata: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    if not metadata:
        return {}
    return {
        int(instance["instance_id"]): instance
        for instance in metadata.get("instances", [])
        if "instance_id" in instance
    }


def _matrix_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape != (4, 4):
        return None
    return matrix


def _match_crop_to_gt(
    labels: np.ndarray,
    targets: np.ndarray | None,
    instance_id: int,
) -> tuple[int | None, float | None]:
    if targets is None:
        return None, None
    crop_mask = labels == int(instance_id)
    best_gt_id: int | None = None
    best_iou = 0.0
    for gt_id in sorted(int(value) for value in np.unique(targets[crop_mask]) if int(value) > 0):
        gt_mask = targets == gt_id
        intersection = int(np.count_nonzero(crop_mask & gt_mask))
        union = int(np.count_nonzero(crop_mask | gt_mask))
        iou = float(intersection / union) if union > 0 else 0.0
        if iou > best_iou:
            best_gt_id = gt_id
            best_iou = iou
    return best_gt_id, best_iou if best_gt_id is not None else None
