"""Losses for learned pose-translation refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .pose_geometry import symmetry_aware_add


@dataclass(frozen=True)
class PoseTranslationRefinerLossConfig:
    translation_weight: float = 1.0
    add_weight: float = 0.5
    smooth_l1_beta: float = 0.01

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "PoseTranslationRefinerLossConfig":
        config = value or {}
        return cls(
            translation_weight=float(config.get("translation_weight", cls.translation_weight)),
            add_weight=float(config.get("add_weight", cls.add_weight)),
            smooth_l1_beta=float(config.get("smooth_l1_beta", cls.smooth_l1_beta)),
        )


class PoseTranslationRefinerLoss(nn.Module):
    """Supervise normalized origin residual and ADD with raw rotation fixed."""

    def __init__(self, config: PoseTranslationRefinerLossConfig | dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = config if isinstance(config, PoseTranslationRefinerLossConfig) else PoseTranslationRefinerLossConfig.from_mapping(config)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, Any],
        *,
        raw_rotation_object_to_camera: torch.Tensor,
        raw_origin_camera: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pred_delta = outputs["origin_delta_camera_normalized"]
        origin_gt = batch["object_origin_camera"].to(device=pred_delta.device, dtype=pred_delta.dtype)
        diameter = batch["model_diameter_m"].to(device=pred_delta.device, dtype=pred_delta.dtype).view(-1, 1).clamp_min(1e-8)
        target_delta = (origin_gt - raw_origin_camera.to(device=pred_delta.device, dtype=pred_delta.dtype)) / diameter
        translation_loss = F.smooth_l1_loss(pred_delta, target_delta, beta=self.config.smooth_l1_beta)
        refined_origin = raw_origin_camera.to(device=pred_delta.device, dtype=pred_delta.dtype) + pred_delta * diameter
        rotation_gt = batch["rotation_object_to_camera"].to(device=pred_delta.device, dtype=pred_delta.dtype)
        model_points = batch["model_points_object"].to(device=pred_delta.device, dtype=pred_delta.dtype)
        symmetry = batch["symmetry_matrices_object"].to(device=pred_delta.device, dtype=pred_delta.dtype)
        if model_points.ndim == 3:
            model_points = model_points[0]
        if symmetry.ndim == 4:
            symmetry = symmetry[0]
        add = symmetry_aware_add(
            rotation_pred_object_to_camera=raw_rotation_object_to_camera.to(device=pred_delta.device, dtype=pred_delta.dtype),
            origin_pred_camera=refined_origin,
            rotation_gt_object_to_camera=rotation_gt,
            origin_gt_camera=origin_gt,
            model_points_object=model_points,
            symmetry_matrices_object=symmetry,
        )
        add_loss = (add / diameter.view(-1)).mean()
        total_loss = self.config.translation_weight * translation_loss + self.config.add_weight * add_loss
        return {
            "total_loss": total_loss,
            "translation_loss": translation_loss,
            "add_loss": add_loss,
        }


def build_pose_translation_refiner_loss_from_config(config: dict[str, Any]) -> PoseTranslationRefinerLoss:
    return PoseTranslationRefinerLoss(config.get("loss", {}))

