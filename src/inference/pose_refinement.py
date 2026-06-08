"""Conservative pose refinement for crop-level 6D pose estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.training.pose_metrics import weighted_kabsch_object_to_camera


@dataclass(frozen=True)
class PoseRefinementConfig:
    method: str = "translation_only_icp"
    max_iterations: int = 10
    distance_threshold_fraction: float = 0.08
    max_translation_delta_fraction: float = 0.10
    max_rotation_delta_deg: float = 10.0
    accept_if_improved: bool = True
    eps: float = 1e-8


def refine_pose_batch(
    *,
    rotation_object_to_camera: torch.Tensor,
    origin_camera: torch.Tensor,
    batch: dict[str, Any],
    config: PoseRefinementConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Refine a batch of object-to-camera poses against visible crop points."""

    config = config or PoseRefinementConfig()
    method = config.method.lower()
    if method == "none":
        return _result(rotation_object_to_camera, origin_camera, origin_camera, rotation_object_to_camera, batch, config, accepted=None)
    if method == "translation_only_icp":
        return refine_translation_only(rotation_object_to_camera=rotation_object_to_camera, origin_camera=origin_camera, batch=batch, config=config)
    if method == "point_to_point_icp":
        return refine_point_to_point(rotation_object_to_camera=rotation_object_to_camera, origin_camera=origin_camera, batch=batch, config=config)
    if method == "hybrid":
        first = refine_translation_only(rotation_object_to_camera=rotation_object_to_camera, origin_camera=origin_camera, batch=batch, config=config)
        return refine_point_to_point(
            rotation_object_to_camera=first["rotation_object_to_camera"],
            origin_camera=first["origin_camera"],
            batch=batch,
            config=config,
        )
    raise ValueError(f"Unknown pose refinement method: {config.method}")


def refine_translation_only(
    *,
    rotation_object_to_camera: torch.Tensor,
    origin_camera: torch.Tensor,
    batch: dict[str, Any],
    config: PoseRefinementConfig,
) -> dict[str, torch.Tensor]:
    """Refine only translation using robust visible-crop-to-model residuals."""

    model_points = _model_points_for_batch(batch, rotation_object_to_camera)
    crop_points = batch["sampled_points_camera"].to(device=origin_camera.device, dtype=origin_camera.dtype)
    diameter = batch["model_diameter_m"].to(device=origin_camera.device, dtype=origin_camera.dtype).view(-1)
    model_camera = _transform(model_points, rotation_object_to_camera, origin_camera)
    before = _one_sided_chamfer(crop_points, model_camera)
    nearest_distances, nearest_indices = torch.cdist(crop_points, model_camera).min(dim=2)
    nearest_model = torch.gather(model_camera, 1, nearest_indices.unsqueeze(-1).expand(-1, -1, 3))
    residuals = crop_points - nearest_model
    threshold = diameter * float(config.distance_threshold_fraction)
    max_delta = diameter * float(config.max_translation_delta_fraction)
    deltas = []
    for batch_idx in range(residuals.shape[0]):
        inliers = residuals[batch_idx][nearest_distances[batch_idx] <= threshold[batch_idx]]
        if inliers.shape[0] == 0:
            inliers = residuals[batch_idx]
        delta = inliers.median(dim=0).values
        delta_norm = torch.linalg.norm(delta)
        if delta_norm > max_delta[batch_idx]:
            delta = delta * (max_delta[batch_idx] / delta_norm.clamp_min(config.eps))
        deltas.append(delta)
    delta = torch.stack(deltas, dim=0)
    refined_origin = origin_camera + delta
    refined_model_camera = _transform(model_points, rotation_object_to_camera, refined_origin)
    after = _one_sided_chamfer(crop_points, refined_model_camera)
    accepted = _accept_mask(before, after, delta, torch.zeros_like(diameter), diameter, config)
    final_origin = torch.where(accepted.unsqueeze(-1), refined_origin, origin_camera)
    return _result(rotation_object_to_camera, final_origin, origin_camera, rotation_object_to_camera, batch, config, accepted=accepted, before=before, after=after)


def refine_point_to_point(
    *,
    rotation_object_to_camera: torch.Tensor,
    origin_camera: torch.Tensor,
    batch: dict[str, Any],
    config: PoseRefinementConfig,
) -> dict[str, torch.Tensor]:
    """Run conservative point-to-point ICP updates with clamp-and-accept gates."""

    model_points = _model_points_for_batch(batch, rotation_object_to_camera)
    crop_points = batch["sampled_points_camera"].to(device=origin_camera.device, dtype=origin_camera.dtype)
    diameter = batch["model_diameter_m"].to(device=origin_camera.device, dtype=origin_camera.dtype).view(-1)
    rotation = rotation_object_to_camera
    origin = origin_camera
    before = _one_sided_chamfer(crop_points, _transform(model_points, rotation, origin))
    for _ in range(max(1, int(config.max_iterations))):
        transformed = _transform(model_points, rotation, origin)
        nearest_distances, nearest_indices = torch.cdist(crop_points, transformed).min(dim=2)
        threshold = diameter * float(config.distance_threshold_fraction)
        next_rotations = []
        next_origins = []
        for batch_idx in range(crop_points.shape[0]):
            inlier_mask = nearest_distances[batch_idx] <= threshold[batch_idx]
            if int(inlier_mask.sum()) < 3:
                inlier_mask = torch.ones_like(inlier_mask, dtype=torch.bool)
            source_object = model_points[batch_idx][nearest_indices[batch_idx][inlier_mask]].unsqueeze(0)
            target_camera = crop_points[batch_idx][inlier_mask].unsqueeze(0)
            fitted_rotation, fitted_origin = weighted_kabsch_object_to_camera(source_object, target_camera)
            next_rotations.append(fitted_rotation[0])
            next_origins.append(fitted_origin[0])
        candidate_rotation = torch.stack(next_rotations, dim=0)
        candidate_origin = torch.stack(next_origins, dim=0)
        rotation_delta = rotation_error_deg(candidate_rotation, rotation)
        origin_delta = torch.linalg.norm(candidate_origin - origin_camera, dim=-1)
        max_origin_delta = diameter * float(config.max_translation_delta_fraction)
        keep = (rotation_delta <= float(config.max_rotation_delta_deg)) & (origin_delta <= max_origin_delta)
        rotation = torch.where(keep.view(-1, 1, 1), candidate_rotation, rotation)
        origin = torch.where(keep.view(-1, 1), candidate_origin, origin)
    after = _one_sided_chamfer(crop_points, _transform(model_points, rotation, origin))
    total_origin_delta = origin - origin_camera
    total_rotation_delta = rotation_error_deg(rotation, rotation_object_to_camera)
    accepted = _accept_mask(before, after, total_origin_delta, total_rotation_delta, diameter, config)
    final_rotation = torch.where(accepted.view(-1, 1, 1), rotation, rotation_object_to_camera)
    final_origin = torch.where(accepted.view(-1, 1), origin, origin_camera)
    return _result(final_rotation, final_origin, origin_camera, rotation_object_to_camera, batch, config, accepted=accepted, before=before, after=after)


def _accept_mask(
    before: torch.Tensor,
    after: torch.Tensor,
    origin_delta: torch.Tensor,
    rotation_delta_deg: torch.Tensor,
    diameter: torch.Tensor,
    config: PoseRefinementConfig,
) -> torch.Tensor:
    if origin_delta.ndim == 2:
        origin_delta_norm = torch.linalg.norm(origin_delta, dim=-1)
    else:
        origin_delta_norm = origin_delta
    limits_ok = (
        origin_delta_norm <= diameter * float(config.max_translation_delta_fraction)
    ) & (
        rotation_delta_deg <= float(config.max_rotation_delta_deg)
    )
    if config.accept_if_improved:
        return limits_ok & (after <= before)
    return limits_ok


def _result(
    rotation: torch.Tensor,
    origin: torch.Tensor,
    raw_origin: torch.Tensor,
    raw_rotation: torch.Tensor,
    batch: dict[str, Any],
    config: PoseRefinementConfig,
    *,
    accepted: torch.Tensor | None,
    before: torch.Tensor | None = None,
    after: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    diameter = batch["model_diameter_m"].to(device=origin.device, dtype=origin.dtype).view(-1)
    if accepted is None:
        accepted = torch.zeros(origin.shape[0], dtype=torch.bool, device=origin.device)
    if before is None or after is None:
        model_points = _model_points_for_batch(batch, rotation)
        crop_points = batch["sampled_points_camera"].to(device=origin.device, dtype=origin.dtype)
        before = _one_sided_chamfer(crop_points, _transform(model_points, raw_rotation, raw_origin))
        after = _one_sided_chamfer(crop_points, _transform(model_points, rotation, origin))
    origin_delta = torch.linalg.norm(origin - raw_origin, dim=-1)
    rotation_delta = rotation_error_deg(rotation, raw_rotation)
    return {
        "rotation_object_to_camera": rotation,
        "origin_camera": origin,
        "accepted": accepted.float(),
        "alignment_before_m": before,
        "alignment_after_m": after,
        "alignment_delta_m": after - before,
        "translation_delta_m": origin_delta,
        "translation_delta_mm": origin_delta * 1000.0,
        "translation_delta_fraction": origin_delta / diameter.clamp_min(config.eps),
        "rotation_delta_deg": rotation_delta,
    }


def _model_points_for_batch(batch: dict[str, Any], reference: torch.Tensor) -> torch.Tensor:
    model_points = batch["model_points_object"].to(device=reference.device, dtype=reference.dtype)
    if model_points.ndim == 2:
        model_points = model_points.unsqueeze(0).expand(reference.shape[0], -1, -1)
    return model_points


def _transform(points_object: torch.Tensor, rotation_object_to_camera: torch.Tensor, origin_camera: torch.Tensor) -> torch.Tensor:
    return points_object @ rotation_object_to_camera.transpose(-1, -2) + origin_camera.unsqueeze(1)


def _one_sided_chamfer(points_camera: torch.Tensor, model_camera: torch.Tensor) -> torch.Tensor:
    return torch.cdist(points_camera, model_camera).min(dim=2).values.mean(dim=1)


def rotation_error_deg(rotation_a: torch.Tensor, rotation_b: torch.Tensor) -> torch.Tensor:
    delta = rotation_a.transpose(-1, -2) @ rotation_b
    trace = delta.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos_angle = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos_angle))
