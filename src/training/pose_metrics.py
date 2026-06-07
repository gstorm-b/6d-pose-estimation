"""Pose-estimation metrics with finite object-frame symmetry support."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .pose_geometry import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    symmetry_aware_add,
    symmetry_aware_rotation_error_deg,
)


def pose_predictions_to_object_to_camera(outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Convert model outputs into explicit `object_to_camera` components."""

    if "predicted_points_object_normalized" in outputs:
        return voting_predictions_to_object_to_camera(outputs, batch)
    rotation = rotation_6d_to_matrix(outputs["rotation_6d"])
    offset_norm = outputs["object_origin_offset_from_crop_centroid_camera_normalized"]
    diameter = batch["model_diameter_m"].to(device=offset_norm.device, dtype=offset_norm.dtype).view(-1, 1)
    centroid = batch["crop_centroid_camera"].to(device=offset_norm.device, dtype=offset_norm.dtype)
    origin = centroid + offset_norm * diameter
    return {
        "rotation_object_to_camera": rotation,
        "object_origin_camera": origin,
        "object_origin_offset_from_crop_centroid_camera_normalized": offset_norm,
    }


def voting_predictions_to_object_to_camera(outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Estimate `object_to_camera` from dense object-coordinate votes."""

    pred_object_normalized = outputs["predicted_points_object_normalized"]
    diameter = batch["model_diameter_m"].to(device=pred_object_normalized.device, dtype=pred_object_normalized.dtype).view(-1, 1, 1)
    object_points = pred_object_normalized * diameter
    camera_points = batch["sampled_points_camera"].to(device=object_points.device, dtype=object_points.dtype)
    rotation, origin = weighted_kabsch_object_to_camera(object_points, camera_points)
    if "predicted_keypoint_offsets_camera_normalized" in outputs:
        keypoint_offsets = outputs["predicted_keypoint_offsets_camera_normalized"].to(
            device=camera_points.device,
            dtype=camera_points.dtype,
        )
        keypoint_votes = camera_points.unsqueeze(2) + keypoint_offsets * diameter.unsqueeze(2)
        keypoints_camera = keypoint_votes.mean(dim=1)
        keypoints_object = batch["keypoints_object"].to(device=camera_points.device, dtype=camera_points.dtype)
        rotation, origin = weighted_kabsch_object_to_camera(keypoints_object, keypoints_camera)
    elif "rotation_6d" in outputs:
        rotation = rotation_6d_to_matrix(outputs["rotation_6d"].to(device=object_points.device, dtype=object_points.dtype))
    if "predicted_origin_offsets_camera_normalized" in outputs:
        origin_votes = camera_points + outputs["predicted_origin_offsets_camera_normalized"].to(
            device=camera_points.device,
            dtype=camera_points.dtype,
        ) * diameter
        origin = origin_votes.mean(dim=1)
    centroid = batch["crop_centroid_camera"].to(device=origin.device, dtype=origin.dtype)
    offset = origin - centroid
    offset_norm = offset / diameter.view(-1, 1).clamp_min(1e-8)
    return {
        "rotation_object_to_camera": rotation,
        "object_origin_camera": origin,
        "object_origin_offset_from_crop_centroid_camera_normalized": offset_norm,
        "rotation_6d": matrix_to_rotation_6d(rotation),
    }


def weighted_kabsch_object_to_camera(
    object_points: torch.Tensor,
    camera_points: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit row-vector transform `camera = object @ R.T + t`."""

    if weights is None:
        weights = torch.ones(object_points.shape[:-1], dtype=object_points.dtype, device=object_points.device)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    object_mean = (object_points * weights.unsqueeze(-1)).sum(dim=1)
    camera_mean = (camera_points * weights.unsqueeze(-1)).sum(dim=1)
    object_centered = object_points - object_mean.unsqueeze(1)
    camera_centered = camera_points - camera_mean.unsqueeze(1)
    covariance = object_centered.transpose(1, 2) @ (camera_centered * weights.unsqueeze(-1))
    u, _s, vh = torch.linalg.svd(covariance)
    rotation = vh.transpose(-1, -2) @ u.transpose(-1, -2)
    det = torch.linalg.det(rotation)
    if torch.any(det < 0):
        correction = torch.ones((rotation.shape[0], 3), dtype=rotation.dtype, device=rotation.device)
        correction[:, -1] = torch.where(det < 0, -1.0, 1.0)
        rotation = vh.transpose(-1, -2) @ torch.diag_embed(correction) @ u.transpose(-1, -2)
    origin = camera_mean - object_mean @ rotation.transpose(-1, -2)
    return rotation, origin


def compute_pose_metrics(outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, float]:
    """Compute batch-mean pose metrics."""

    pred = pose_predictions_to_object_to_camera(outputs, batch)
    rotation_pred = pred["rotation_object_to_camera"]
    origin_pred = pred["object_origin_camera"]
    rotation_gt = batch["rotation_object_to_camera"].to(device=rotation_pred.device, dtype=rotation_pred.dtype)
    origin_gt = batch["object_origin_camera"].to(device=origin_pred.device, dtype=origin_pred.dtype)
    model_points = batch["model_points_object"].to(device=rotation_pred.device, dtype=rotation_pred.dtype)
    symmetry = batch["symmetry_matrices_object"].to(device=rotation_pred.device, dtype=rotation_pred.dtype)
    if model_points.ndim == 3:
        model_points = model_points[0]
    if symmetry.ndim == 4:
        symmetry = symmetry[0]
    add = symmetry_aware_add(
        rotation_pred_object_to_camera=rotation_pred,
        origin_pred_camera=origin_pred,
        rotation_gt_object_to_camera=rotation_gt,
        origin_gt_camera=origin_gt,
        model_points_object=model_points,
        symmetry_matrices_object=symmetry,
    )
    rotation_error = symmetry_aware_rotation_error_deg(
        rotation_pred_object_to_camera=rotation_pred,
        rotation_gt_object_to_camera=rotation_gt,
        symmetry_matrices_object=symmetry,
    )
    translation_error = torch.linalg.norm(origin_pred - origin_gt, dim=-1)
    diameter = batch["model_diameter_m"].to(device=add.device, dtype=add.dtype).view(-1)
    success_01d = add <= (0.1 * diameter)
    success_005d = add <= (0.05 * diameter)
    return {
        "translation_error_m": float(translation_error.mean().detach().cpu()),
        "translation_error_mm": float((translation_error * 1000.0).mean().detach().cpu()),
        "rotation_error_deg_symmetry_aware": float(rotation_error.mean().detach().cpu()),
        "add_m": float(add.mean().detach().cpu()),
        "add_mm": float((add * 1000.0).mean().detach().cpu()),
        "pose_success_add_0.1d": float(success_01d.float().mean().detach().cpu()),
        "pose_success_add_0.05d": float(success_005d.float().mean().detach().cpu()),
        "matched_crop_iou": float(batch["matched_gt_iou"].float().mean().detach().cpu()),
    }


class PoseMetricMeter:
    """Accumulate pose metrics over batches."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.weight = 0

    def update(self, outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> None:
        metrics = compute_pose_metrics(outputs, batch)
        first_output = next(iter(outputs.values()))
        batch_size = int(first_output.shape[0])
        self.weight += batch_size
        for key, value in metrics.items():
            self.totals[key] = self.totals.get(key, 0.0) + float(value) * batch_size

    def compute(self) -> dict[str, float]:
        divisor = max(self.weight, 1)
        return {key: value / divisor for key, value in sorted(self.totals.items())}


def add_auc(add_errors_m: np.ndarray, diameter_m: float, *, max_threshold_fraction: float = 0.1) -> float:
    """Compute simple ADD AUC up to a diameter-relative threshold."""

    errors = np.asarray(add_errors_m, dtype=np.float32)
    if errors.size == 0:
        return 0.0
    max_threshold = float(max_threshold_fraction) * float(diameter_m)
    thresholds = np.linspace(0.0, max_threshold, 101, dtype=np.float32)
    recalls = np.asarray([(errors <= threshold).mean() for threshold in thresholds], dtype=np.float32)
    return float(np.trapz(recalls, thresholds) / max(max_threshold, 1e-8))
