"""Learned translation residual refiner for frozen crop-level pose checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from src.training.pose_geometry import matrix_to_rotation_6d
from src.training.pose_metrics import CenterVoteAggregationConfig, pose_predictions_to_object_to_camera


POSE_REFINER_FEATURE_DIM = 41


@dataclass(frozen=True)
class PoseRefinerFeatureConfig:
    center_vote_config: CenterVoteAggregationConfig | None = None
    eps: float = 1e-8


class PointNet2PoseTranslationRefiner(nn.Module):
    """Predict a small normalized camera-frame origin residual from raw pose diagnostics."""

    def __init__(
        self,
        input_dim: int = POSE_REFINER_FEATURE_DIM,
        hidden_dim: int = 128,
        depth: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        layers: list[nn.Module] = []
        current_dim = self.input_dim
        for _ in range(max(1, int(depth))):
            layers.extend(
                [
                    nn.Linear(current_dim, int(hidden_dim), bias=False),
                    nn.BatchNorm1d(int(hidden_dim)),
                    nn.ReLU(inplace=True),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            current_dim = int(hidden_dim)
        layers.append(nn.Linear(current_dim, 3))
        self.net = nn.Sequential(*layers)
        self._init_delta_head()

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"origin_delta_camera_normalized": self.net(features)}

    def _init_delta_head(self) -> None:
        final = self.net[-1]
        if isinstance(final, nn.Linear):
            nn.init.normal_(final.weight, mean=0.0, std=1e-4)
            nn.init.zeros_(final.bias)


def build_pointnet2_pose_refiner_from_config(config: dict[str, Any]) -> PointNet2PoseTranslationRefiner:
    model_config = config.get("refiner", config.get("model", {}))
    return PointNet2PoseTranslationRefiner(
        input_dim=int(model_config.get("input_dim", POSE_REFINER_FEATURE_DIM)),
        hidden_dim=int(model_config.get("hidden_dim", 128)),
        depth=int(model_config.get("depth", 3)),
        dropout=float(model_config.get("dropout", 0.1)),
    )


def build_pose_refiner_features(
    *,
    pose_outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    raw_pose: dict[str, torch.Tensor] | None = None,
    feature_config: PoseRefinerFeatureConfig | None = None,
) -> torch.Tensor:
    """Build stable raw-pose diagnostics for the translation refiner."""

    feature_config = feature_config or PoseRefinerFeatureConfig()
    raw_pose = raw_pose or pose_predictions_to_object_to_camera(
        pose_outputs,
        batch,
        center_vote_config=feature_config.center_vote_config,
    )
    origin = raw_pose["object_origin_camera"]
    dtype = origin.dtype
    device = origin.device
    diameter = batch["model_diameter_m"].to(device=device, dtype=dtype).view(-1, 1).clamp_min(feature_config.eps)
    centroid = batch["crop_centroid_camera"].to(device=device, dtype=dtype)
    raw_offset_norm = (origin - centroid) / diameter
    rotation_6d = raw_pose.get("rotation_6d")
    if rotation_6d is None:
        rotation_6d = matrix_to_rotation_6d(raw_pose["rotation_object_to_camera"])
    rotation_6d = rotation_6d.to(device=device, dtype=dtype)
    aux = batch.get("crop_aux_features")
    if aux is None:
        aux = torch.zeros(origin.shape[0], 18, dtype=dtype, device=device)
    aux = aux.to(device=device, dtype=dtype)
    if aux.shape[1] < 18:
        aux = torch.cat([aux, torch.zeros(aux.shape[0], 18 - aux.shape[1], dtype=dtype, device=device)], dim=1)
    elif aux.shape[1] > 18:
        aux = aux[:, :18]

    vote_features = vote_diagnostics(
        pose_outputs=pose_outputs,
        batch=batch,
        raw_origin_camera=origin,
        diameter=diameter,
        eps=feature_config.eps,
    )
    crop_count = batch.get("crop_point_count")
    if crop_count is None:
        count_feature = torch.ones(origin.shape[0], 1, dtype=dtype, device=device)
    else:
        count_feature = torch.log1p(crop_count.to(device=device, dtype=dtype).view(-1, 1)) / 10.0

    return torch.cat([raw_offset_norm, rotation_6d, aux, vote_features, count_feature], dim=1)


def vote_diagnostics(
    *,
    pose_outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    raw_origin_camera: torch.Tensor,
    diameter: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    dtype = raw_origin_camera.dtype
    device = raw_origin_camera.device
    if "predicted_origin_offsets_camera_normalized" not in pose_outputs:
        return torch.zeros(raw_origin_camera.shape[0], 13, dtype=dtype, device=device)
    camera_points = batch["sampled_points_camera"].to(device=device, dtype=dtype)
    offsets = pose_outputs["predicted_origin_offsets_camera_normalized"].to(device=device, dtype=dtype)
    origin_votes = camera_points + offsets * diameter.view(-1, 1, 1)
    mean = origin_votes.mean(dim=1)
    median = origin_votes.median(dim=1).values
    std = origin_votes.std(dim=1, unbiased=False)
    residual = torch.linalg.norm(origin_votes - raw_origin_camera.unsqueeze(1), dim=-1) / diameter.view(-1, 1).clamp_min(eps)
    residual_stats = torch.stack(
        [
            residual.mean(dim=1),
            residual.std(dim=1, unbiased=False),
            residual.max(dim=1).values,
            residual.median(dim=1).values,
        ],
        dim=1,
    )
    return torch.cat(
        [
            (mean - raw_origin_camera) / diameter,
            (median - raw_origin_camera) / diameter,
            std / diameter,
            residual_stats,
        ],
        dim=1,
    )

