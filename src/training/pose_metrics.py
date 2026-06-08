"""Pose-estimation metrics with finite object-frame symmetry support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .pose_geometry import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    symmetry_aware_add,
    symmetry_aware_rotation_error_deg,
)


@dataclass(frozen=True)
class CenterVoteAggregationConfig:
    method: str = "mean"
    trim_fraction: float = 0.1
    max_residual_fraction: float | None = None
    geometric_median_iterations: int = 8
    eps: float = 1e-8


def pose_predictions_to_object_to_camera(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    center_vote_config: CenterVoteAggregationConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Convert model outputs into explicit `object_to_camera` components."""

    if "predicted_points_object_normalized" in outputs:
        return voting_predictions_to_object_to_camera(outputs, batch, center_vote_config=center_vote_config)
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


def voting_predictions_to_object_to_camera(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    center_vote_config: CenterVoteAggregationConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Estimate `object_to_camera` from dense object-coordinate votes."""

    center_vote_config = center_vote_config or CenterVoteAggregationConfig()
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
        origin = aggregate_center_votes(origin_votes, diameter.view(-1), center_vote_config)
    centroid = batch["crop_centroid_camera"].to(device=origin.device, dtype=origin.dtype)
    offset = origin - centroid
    offset_norm = offset / diameter.view(-1, 1).clamp_min(1e-8)
    return {
        "rotation_object_to_camera": rotation,
        "object_origin_camera": origin,
        "object_origin_offset_from_crop_centroid_camera_normalized": offset_norm,
        "rotation_6d": matrix_to_rotation_6d(rotation),
    }


def aggregate_center_votes(
    origin_votes: torch.Tensor,
    diameter: torch.Tensor,
    config: CenterVoteAggregationConfig | None = None,
) -> torch.Tensor:
    """Aggregate per-point object-origin votes into one origin per crop."""

    config = config or CenterVoteAggregationConfig()
    method = config.method.lower()
    if origin_votes.ndim != 3 or origin_votes.shape[-1] != 3:
        raise ValueError(f"expected origin votes with shape (B, N, 3), got {tuple(origin_votes.shape)}")
    if method == "mean":
        return origin_votes.mean(dim=1)
    median = origin_votes.median(dim=1).values
    if method == "median":
        return median
    if method == "geometric_median":
        return geometric_median(origin_votes, median, iterations=config.geometric_median_iterations, eps=config.eps)
    if method not in {"trimmed_mean", "weighted_trimmed_mean"}:
        raise ValueError(f"Unknown center-vote aggregation method: {config.method}")

    distances = torch.linalg.norm(origin_votes - median.unsqueeze(1), dim=-1)
    keep_count = max(1, int(round(origin_votes.shape[1] * (1.0 - float(config.trim_fraction)))))
    keep_count = min(keep_count, origin_votes.shape[1])
    nearest = torch.topk(distances, k=keep_count, dim=1, largest=False).indices
    keep_mask = torch.zeros_like(distances, dtype=torch.bool)
    keep_mask.scatter_(1, nearest, True)
    if config.max_residual_fraction is not None and config.max_residual_fraction > 0:
        max_residual = diameter.to(device=origin_votes.device, dtype=origin_votes.dtype).view(-1, 1) * float(config.max_residual_fraction)
        keep_mask = keep_mask & (distances <= max_residual)
    fallback_mask = keep_mask.any(dim=1, keepdim=True)
    keep_mask = torch.where(fallback_mask, keep_mask, torch.ones_like(keep_mask, dtype=torch.bool))
    if method == "weighted_trimmed_mean":
        weights = torch.where(keep_mask, 1.0 / distances.clamp_min(config.eps), torch.zeros_like(distances))
    else:
        weights = keep_mask.to(dtype=origin_votes.dtype)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(config.eps)
    return (origin_votes * weights.unsqueeze(-1)).sum(dim=1)


def geometric_median(
    points: torch.Tensor,
    initial: torch.Tensor,
    *,
    iterations: int = 8,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Batched Weiszfeld geometric median for small vote clouds."""

    current = initial
    for _ in range(max(1, int(iterations))):
        distances = torch.linalg.norm(points - current.unsqueeze(1), dim=-1).clamp_min(eps)
        weights = 1.0 / distances
        current = (points * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(eps)
    return current


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
    transformed_object_mean = (object_mean.unsqueeze(1) @ rotation.transpose(-1, -2)).squeeze(1)
    origin = camera_mean - transformed_object_mean
    return rotation, origin


def compute_pose_metrics_per_sample(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    center_vote_config: CenterVoteAggregationConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Compute per-crop pose metrics as tensors."""

    pred = pose_predictions_to_object_to_camera(outputs, batch, center_vote_config=center_vote_config)
    return compute_pose_metrics_per_sample_from_pose(
        rotation_pred_object_to_camera=pred["rotation_object_to_camera"],
        origin_pred_camera=pred["object_origin_camera"],
        batch=batch,
    )


def compute_pose_metrics_per_sample_from_pose(
    *,
    rotation_pred_object_to_camera: torch.Tensor,
    origin_pred_camera: torch.Tensor,
    batch: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Compute per-crop pose metrics for explicit pose tensors."""

    rotation_pred = rotation_pred_object_to_camera
    origin_pred = origin_pred_camera
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
    matched_iou = batch["matched_gt_iou"].to(device=add.device, dtype=add.dtype).view(-1)
    return {
        "translation_error_m": translation_error,
        "translation_error_mm": translation_error * 1000.0,
        "rotation_error_deg_symmetry_aware": rotation_error,
        "add_m": add,
        "add_mm": add * 1000.0,
        "pose_success_add_0.1d": success_01d.float(),
        "pose_success_add_0.05d": success_005d.float(),
        "matched_crop_iou": matched_iou,
    }


def compute_pose_metrics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    center_vote_config: CenterVoteAggregationConfig | None = None,
) -> dict[str, float]:
    """Compute batch-mean pose metrics."""

    metrics = compute_pose_metrics_per_sample(outputs, batch, center_vote_config=center_vote_config)
    return {
        key: float(value.mean().detach().cpu())
        for key, value in metrics.items()
    }


class PoseMetricMeter:
    """Accumulate pose metrics over batches."""

    def __init__(self, *, center_vote_config: CenterVoteAggregationConfig | None = None) -> None:
        self.center_vote_config = center_vote_config
        self.totals: dict[str, float] = {}
        self.weight = 0

    def update(self, outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> None:
        metrics = compute_pose_metrics(outputs, batch, center_vote_config=self.center_vote_config)
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
