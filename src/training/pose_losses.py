"""Losses for PointNet++ crop-based 6D pose estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .pose_geometry import matrix_to_rotation_6d, symmetry_aware_add
from .pose_metrics import pose_predictions_to_object_to_camera


@dataclass(frozen=True)
class PoseLossConfig:
    translation_weight: float = 1.0
    rotation_weight: float = 0.5
    model_points_weight: float = 1.0
    model_points_use_gt_translation: bool = True
    translation_smooth_l1_beta: float = 0.01

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "PoseLossConfig":
        config = value or {}
        return cls(
            translation_weight=float(config.get("translation_weight", cls.translation_weight)),
            rotation_weight=float(config.get("rotation_weight", cls.rotation_weight)),
            model_points_weight=float(config.get("model_points_weight", cls.model_points_weight)),
            model_points_use_gt_translation=bool(config.get("model_points_use_gt_translation", cls.model_points_use_gt_translation)),
            translation_smooth_l1_beta=float(config.get("translation_smooth_l1_beta", cls.translation_smooth_l1_beta)),
        )


class PointNet2PoseLoss(nn.Module):
    """Composite translation and symmetry-aware model-point loss."""

    def __init__(self, config: PoseLossConfig | dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = config if isinstance(config, PoseLossConfig) else PoseLossConfig.from_mapping(config)

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        pred = pose_predictions_to_object_to_camera(outputs, batch)
        target_offset = batch["object_origin_offset_from_crop_centroid_camera_normalized"].to(
            device=outputs["rotation_6d"].device,
            dtype=outputs["rotation_6d"].dtype,
        )
        pred_offset = pred["object_origin_offset_from_crop_centroid_camera_normalized"]
        translation_loss = F.smooth_l1_loss(
            pred_offset,
            target_offset,
            beta=self.config.translation_smooth_l1_beta,
        )
        rotation_gt = batch["rotation_object_to_camera"].to(device=outputs["rotation_6d"].device, dtype=outputs["rotation_6d"].dtype)
        origin_gt = batch["object_origin_camera"].to(device=outputs["rotation_6d"].device, dtype=outputs["rotation_6d"].dtype)
        model_points = batch["model_points_object"].to(device=outputs["rotation_6d"].device, dtype=outputs["rotation_6d"].dtype)
        symmetry = batch["symmetry_matrices_object"].to(device=outputs["rotation_6d"].device, dtype=outputs["rotation_6d"].dtype)
        if model_points.ndim == 3:
            model_points = model_points[0]
        if symmetry.ndim == 4:
            symmetry = symmetry[0]
        rotation_loss = symmetry_aware_rotation_matrix_loss(
            rotation_pred_object_to_camera=pred["rotation_object_to_camera"],
            rotation_pred_6d=outputs["rotation_6d"],
            rotation_gt_object_to_camera=rotation_gt,
            symmetry_matrices_object=symmetry,
        )
        add = symmetry_aware_add(
            rotation_pred_object_to_camera=pred["rotation_object_to_camera"],
            origin_pred_camera=origin_gt if self.config.model_points_use_gt_translation else pred["object_origin_camera"],
            rotation_gt_object_to_camera=rotation_gt,
            origin_gt_camera=origin_gt,
            model_points_object=model_points,
            symmetry_matrices_object=symmetry,
        )
        diameter = batch["model_diameter_m"].to(device=add.device, dtype=add.dtype).view(-1)
        model_points_loss = (add / diameter.clamp_min(1e-8)).mean()
        total_loss = (
            self.config.translation_weight * translation_loss
            + self.config.rotation_weight * rotation_loss
            + self.config.model_points_weight * model_points_loss
        )
        return {
            "total_loss": total_loss,
            "translation_loss": translation_loss,
            "rotation_loss": rotation_loss,
            "model_points_loss": model_points_loss,
        }


@dataclass(frozen=True)
class PoseVotingLossConfig:
    object_coord_weight: float = 1.0
    center_vote_weight: float = 1.0
    keypoint_vote_weight: float = 1.0
    rotation_weight: float = 1.0
    model_points_weight: float = 1.0
    model_points_use_gt_translation: bool = True
    smooth_l1_beta: float = 0.02

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "PoseVotingLossConfig":
        config = value or {}
        return cls(
            object_coord_weight=float(config.get("object_coord_weight", cls.object_coord_weight)),
            center_vote_weight=float(config.get("center_vote_weight", cls.center_vote_weight)),
            keypoint_vote_weight=float(config.get("keypoint_vote_weight", cls.keypoint_vote_weight)),
            rotation_weight=float(config.get("rotation_weight", cls.rotation_weight)),
            model_points_weight=float(config.get("model_points_weight", cls.model_points_weight)),
            model_points_use_gt_translation=bool(config.get("model_points_use_gt_translation", cls.model_points_use_gt_translation)),
            smooth_l1_beta=float(config.get("smooth_l1_beta", cls.smooth_l1_beta)),
        )


class PointNet2PoseVotingLoss(nn.Module):
    """Dense object-coordinate vote supervision."""

    def __init__(self, config: PoseVotingLossConfig | dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = config if isinstance(config, PoseVotingLossConfig) else PoseVotingLossConfig.from_mapping(config)

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        pred = outputs["predicted_points_object_normalized"]
        target = batch["sampled_points_object_normalized"].to(device=pred.device, dtype=pred.dtype)
        object_coord_loss = F.smooth_l1_loss(pred, target, beta=self.config.smooth_l1_beta)
        center_pred = outputs["predicted_origin_offsets_camera_normalized"]
        center_target = batch["sampled_origin_offsets_camera_normalized"].to(device=center_pred.device, dtype=center_pred.dtype)
        center_vote_loss = F.smooth_l1_loss(center_pred, center_target, beta=self.config.smooth_l1_beta)
        keypoint_vote_loss = pred.sum() * 0.0
        if "predicted_keypoint_offsets_camera_normalized" in outputs:
            keypoint_pred = outputs["predicted_keypoint_offsets_camera_normalized"]
            keypoint_target = batch["sampled_keypoint_offsets_camera_normalized"].to(device=keypoint_pred.device, dtype=keypoint_pred.dtype)
            keypoint_vote_loss = F.smooth_l1_loss(keypoint_pred, keypoint_target, beta=self.config.smooth_l1_beta)
        zero = pred.sum() * 0.0
        rotation_loss = zero
        model_points_loss = zero
        if "rotation_6d" in outputs and (self.config.rotation_weight > 0.0 or self.config.model_points_weight > 0.0):
            pose_pred = pose_predictions_to_object_to_camera(outputs, batch)
            rotation_gt = batch["rotation_object_to_camera"].to(device=pred.device, dtype=pred.dtype)
            origin_gt = batch["object_origin_camera"].to(device=pred.device, dtype=pred.dtype)
            model_points = batch["model_points_object"].to(device=pred.device, dtype=pred.dtype)
            symmetry = batch["symmetry_matrices_object"].to(device=pred.device, dtype=pred.dtype)
            if model_points.ndim == 3:
                model_points = model_points[0]
            if symmetry.ndim == 4:
                symmetry = symmetry[0]
            rotation_loss = symmetry_aware_rotation_matrix_loss(
                rotation_pred_object_to_camera=pose_pred["rotation_object_to_camera"],
                rotation_pred_6d=pose_pred.get("rotation_6d", outputs["rotation_6d"]),
                rotation_gt_object_to_camera=rotation_gt,
                symmetry_matrices_object=symmetry,
            )
            add = symmetry_aware_add(
                rotation_pred_object_to_camera=pose_pred["rotation_object_to_camera"],
                origin_pred_camera=origin_gt if self.config.model_points_use_gt_translation else pose_pred["object_origin_camera"],
                rotation_gt_object_to_camera=rotation_gt,
                origin_gt_camera=origin_gt,
                model_points_object=model_points,
                symmetry_matrices_object=symmetry,
            )
            diameter = batch["model_diameter_m"].to(device=add.device, dtype=add.dtype).view(-1)
            model_points_loss = (add / diameter.clamp_min(1e-8)).mean()
        total_loss = (
            self.config.object_coord_weight * object_coord_loss
            + self.config.center_vote_weight * center_vote_loss
            + self.config.keypoint_vote_weight * keypoint_vote_loss
            + self.config.rotation_weight * rotation_loss
            + self.config.model_points_weight * model_points_loss
        )
        return {
            "total_loss": total_loss,
            "object_coord_loss": object_coord_loss,
            "center_vote_loss": center_vote_loss,
            "keypoint_vote_loss": keypoint_vote_loss,
            "rotation_loss": rotation_loss,
            "model_points_loss": model_points_loss,
        }


def symmetry_aware_rotation_matrix_loss(
    *,
    rotation_pred_object_to_camera: torch.Tensor,
    rotation_pred_6d: torch.Tensor,
    rotation_gt_object_to_camera: torch.Tensor,
    symmetry_matrices_object: torch.Tensor,
) -> torch.Tensor:
    """Frobenius + 6D rotation loss under finite object-frame symmetries."""

    candidate_losses = []
    for sym_idx in range(symmetry_matrices_object.shape[0]):
        sym = symmetry_matrices_object[sym_idx].to(
            device=rotation_gt_object_to_camera.device,
            dtype=rotation_gt_object_to_camera.dtype,
        )
        rotation_gt_equiv = rotation_gt_object_to_camera @ sym
        matrix_loss = (rotation_pred_object_to_camera - rotation_gt_equiv).square().mean(dim=(-1, -2))
        target_6d = matrix_to_rotation_6d(rotation_gt_equiv)
        representation_loss = (rotation_pred_6d - target_6d).square().mean(dim=-1)
        candidate_losses.append(matrix_loss + representation_loss)
    return torch.stack(candidate_losses, dim=-1).min(dim=-1).values.mean()


def build_pointnet2_pose_loss_from_config(config: dict[str, Any]) -> PointNet2PoseLoss:
    if config.get("model", {}).get("name") == "pointnet2_pose_voting":
        return PointNet2PoseVotingLoss(config.get("loss", {}))
    return PointNet2PoseLoss(config.get("loss", {}))
