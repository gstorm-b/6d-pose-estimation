"""PointNet++ crop-level 6D pose estimation model."""

from __future__ import annotations

import torch
from torch import nn

from .pointnet2_semseg import PointNet2Backbone
from .pointnet2_pose_voting import build_pointnet2_pose_voting_from_config


def mlp_head(
    in_channels: int,
    out_channels: int,
    *,
    hidden_channels: int = 256,
    dropout: float = 0.2,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Linear(in_channels, hidden_channels, bias=False),
        nn.BatchNorm1d(hidden_channels),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.extend(
        [
            nn.Linear(hidden_channels, hidden_channels // 2, bias=False),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels // 2, out_channels),
        ]
    )
    return nn.Sequential(*layers)


class PointNet2Pose(nn.Module):
    """Predict crop pose as rotation 6D and translation residual from crop centroid."""

    def __init__(
        self,
        input_channels: int = 3,
        sa_npoints: tuple[int, int, int] = (256, 64, 16),
        sa_nsamples: tuple[int, int, int] = (32, 32, 32),
        global_feature_dim: int = 256,
        aux_channels: int = 6,
        dropout: float = 0.2,
        ops_backend: str = "pure_torch",
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.backbone = PointNet2Backbone(
            input_channels=input_channels,
            sa_npoints=sa_npoints,
            sa_nsamples=sa_nsamples,
            ops_backend=ops_backend,
        )
        self.ops_backend_name = self.backbone.ops_backend_name
        self.aux_channels = int(aux_channels)
        pooled_channels = self.backbone.output_channels * 2
        self.shared = nn.Sequential(
            nn.Linear(pooled_channels + self.aux_channels, global_feature_dim, bias=False),
            nn.BatchNorm1d(global_feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.translation_head = mlp_head(global_feature_dim, 3, hidden_channels=global_feature_dim, dropout=dropout)
        self.rotation_head = mlp_head(global_feature_dim, 6, hidden_channels=global_feature_dim, dropout=dropout)
        self._init_pose_heads()

    def forward(
        self,
        points_pose_input: torch.Tensor,
        crop_aux_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if points_pose_input.ndim != 3 or points_pose_input.shape[-1] < 3:
            raise ValueError(f"expected `(B, N, C>=3)`, got {tuple(points_pose_input.shape)}")
        features = self.backbone(points_pose_input)
        global_max = features.max(dim=-1).values
        global_mean = features.mean(dim=-1)
        pooled = torch.cat([global_max, global_mean], dim=1)
        if crop_aux_features is None:
            crop_aux_features = torch.zeros(
                points_pose_input.shape[0],
                self.aux_channels,
                dtype=points_pose_input.dtype,
                device=points_pose_input.device,
            )
        pooled = torch.cat([pooled, crop_aux_features.to(dtype=pooled.dtype, device=pooled.device)], dim=1)
        shared = self.shared(pooled)
        return {
            "object_origin_offset_from_crop_centroid_camera_normalized": self.translation_head(shared),
            "rotation_6d": self.rotation_head(shared),
        }

    def _init_pose_heads(self) -> None:
        translation_final = self.translation_head[-1]
        rotation_final = self.rotation_head[-1]
        if isinstance(translation_final, nn.Linear):
            nn.init.normal_(translation_final.weight, mean=0.0, std=1e-4)
            nn.init.zeros_(translation_final.bias)
        if isinstance(rotation_final, nn.Linear):
            nn.init.normal_(rotation_final.weight, mean=0.0, std=1e-4)
            with torch.no_grad():
                rotation_final.bias.copy_(torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]))


def build_pointnet2_pose_from_config(config: dict) -> PointNet2Pose:
    model_config = config.get("model", {})
    if model_config.get("name") == "pointnet2_pose_voting":
        return build_pointnet2_pose_voting_from_config(config)
    input_features = model_config.get("input_features", "xyz")
    if input_features == "xyz":
        input_channels = 3
    elif input_features == "xyz_normal":
        input_channels = 6
    else:
        input_channels = int(model_config.get("input_channels", 3))
    return PointNet2Pose(
        input_channels=input_channels,
        sa_npoints=tuple(model_config.get("sa_npoints", [256, 64, 16])),
        sa_nsamples=tuple(model_config.get("sa_nsamples", [32, 32, 32])),
        global_feature_dim=int(model_config.get("global_feature_dim", 256)),
        aux_channels=int(model_config.get("aux_channels", 6)),
        dropout=float(model_config.get("dropout", 0.2)),
        ops_backend=str(model_config.get("ops_backend", "pure_torch")),
    )
