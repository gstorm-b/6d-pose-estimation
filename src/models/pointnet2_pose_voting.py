"""PointNet++ dense object-coordinate voting for crop-level pose."""

from __future__ import annotations

import torch
from torch import nn

from .pointnet2_semseg import PointNet2Backbone


class PointNet2PoseVoting(nn.Module):
    """Predict dense pose votes plus a direct crop-level rotation."""

    def __init__(
        self,
        input_channels: int = 6,
        sa_npoints: tuple[int, int, int] = (256, 64, 16),
        sa_nsamples: tuple[int, int, int] = (32, 32, 32),
        aux_channels: int = 18,
        keypoint_count: int = 8,
        hidden_channels: int = 128,
        dropout: float = 0.1,
        ops_backend: str = "pure_torch",
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.aux_channels = int(aux_channels)
        self.keypoint_count = int(keypoint_count)
        self.backbone = PointNet2Backbone(
            input_channels=input_channels,
            sa_npoints=sa_npoints,
            sa_nsamples=sa_nsamples,
            ops_backend=ops_backend,
        )
        self.ops_backend_name = self.backbone.ops_backend_name
        per_point_channels = self.backbone.output_channels
        global_channels = self.backbone.output_channels * 2
        head_in_channels = per_point_channels + global_channels + self.aux_channels
        pooled_channels = global_channels + self.aux_channels
        self.vote_head = nn.Sequential(
            nn.Conv1d(head_in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, 6 + self.keypoint_count * 3, kernel_size=1),
        )
        self.rotation_head = nn.Sequential(
            nn.Linear(pooled_channels, hidden_channels, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_channels, hidden_channels // 2, bias=False),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels // 2, 6),
        )
        self._init_rotation_head()

    def forward(
        self,
        points_pose_input: torch.Tensor,
        crop_aux_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if points_pose_input.ndim != 3 or points_pose_input.shape[-1] < 3:
            raise ValueError(f"expected `(B, N, C>=3)`, got {tuple(points_pose_input.shape)}")
        features = self.backbone(points_pose_input)
        global_max = features.max(dim=-1, keepdim=True).values
        global_mean = features.mean(dim=-1, keepdim=True)
        global_features = torch.cat([global_max, global_mean], dim=1).expand(-1, -1, features.shape[-1])
        if crop_aux_features is None:
            crop_aux_features = torch.zeros(
                points_pose_input.shape[0],
                self.aux_channels,
                dtype=points_pose_input.dtype,
                device=points_pose_input.device,
            )
        aux = crop_aux_features.to(dtype=features.dtype, device=features.device).unsqueeze(-1).expand(-1, -1, features.shape[-1])
        head_input = torch.cat([features, global_features, aux], dim=1)
        votes = self.vote_head(head_input).transpose(1, 2).contiguous()
        pooled = torch.cat(
            [
                global_max.squeeze(-1),
                global_mean.squeeze(-1),
                crop_aux_features.to(dtype=features.dtype, device=features.device),
            ],
            dim=1,
        )
        return {
            "predicted_points_object_normalized": votes[..., :3],
            "predicted_origin_offsets_camera_normalized": votes[..., 3:6],
            "predicted_keypoint_offsets_camera_normalized": votes[..., 6:].reshape(
                votes.shape[0],
                votes.shape[1],
                self.keypoint_count,
                3,
            ),
            "rotation_6d": self.rotation_head(pooled),
        }

    def _init_rotation_head(self) -> None:
        rotation_final = self.rotation_head[-1]
        if isinstance(rotation_final, nn.Linear):
            nn.init.normal_(rotation_final.weight, mean=0.0, std=1e-4)
            with torch.no_grad():
                rotation_final.bias.copy_(torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]))


def pose_voting_input_channels(model_config: dict) -> int:
    input_features = model_config.get("input_features", "xyz_normal")
    if input_features == "xyz":
        return 3
    if input_features == "xyz_normal":
        return 6
    return int(model_config.get("input_channels", 6))


def build_pointnet2_pose_voting_from_config(config: dict) -> PointNet2PoseVoting:
    model_config = config.get("model", {})
    return PointNet2PoseVoting(
        input_channels=pose_voting_input_channels(model_config),
        sa_npoints=tuple(model_config.get("sa_npoints", [256, 64, 16])),
        sa_nsamples=tuple(model_config.get("sa_nsamples", [32, 32, 32])),
        aux_channels=int(model_config.get("aux_channels", 18)),
        keypoint_count=int(model_config.get("keypoint_count", 8)),
        hidden_channels=int(model_config.get("hidden_channels", 128)),
        dropout=float(model_config.get("dropout", 0.1)),
        ops_backend=str(model_config.get("ops_backend", "pure_torch")),
    )
