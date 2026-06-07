"""PointNet++ instance segmentation model with semantic, offset, and embedding heads."""

from __future__ import annotations

import torch
from torch import nn

from .pointnet2_semseg import PointNet2Backbone


def point_head(
    in_channels: int,
    out_channels: int,
    *,
    hidden_channels: int = 64,
    dropout: float = 0.0,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv1d(in_channels, hidden_channels, kernel_size=1, bias=False),
        nn.BatchNorm1d(hidden_channels),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Conv1d(hidden_channels, out_channels, kernel_size=1))
    return nn.Sequential(*layers)


class PointNet2InstanceSeg(nn.Module):
    """PointNet++ point-wise instance segmentation.

    Args:
        point_features: `(B, N, C)` with xyz in the first 3 channels.

    Returns:
        A dict containing:
        - `semantic_logits`: `(B, N, num_classes)`
        - `offsets`: `(B, N, 3)`
        - `embeddings`: `(B, N, embedding_dim)`
    """

    def __init__(
        self,
        num_classes: int = 2,
        input_channels: int = 6,
        sa_npoints: tuple[int, int, int] = (4096, 1024, 256),
        sa_nsamples: tuple[int, int, int] = (32, 32, 32),
        dropout: float = 0.4,
        embedding_dim: int = 8,
        ops_backend: str = "pure_torch",
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.input_channels = int(input_channels)
        self.embedding_dim = int(embedding_dim)
        self.backbone = PointNet2Backbone(
            input_channels=input_channels,
            sa_npoints=sa_npoints,
            sa_nsamples=sa_nsamples,
            ops_backend=ops_backend,
        )
        self.ops_backend_name = self.backbone.ops_backend_name
        feature_channels = self.backbone.output_channels
        self.semantic_head = point_head(feature_channels, num_classes, dropout=dropout)
        self.offset_head = point_head(feature_channels, 3, dropout=dropout)
        self.embedding_head = point_head(feature_channels, embedding_dim, dropout=dropout)

    def forward(self, point_features: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(point_features)
        semantic_logits = self.semantic_head(features).transpose(1, 2).contiguous()
        offsets = self.offset_head(features).transpose(1, 2).contiguous()
        embeddings = self.embedding_head(features).transpose(1, 2).contiguous()
        return {
            "semantic_logits": semantic_logits,
            "offsets": offsets,
            "embeddings": embeddings,
        }


def build_pointnet2_instance_seg_from_config(config: dict) -> PointNet2InstanceSeg:
    model_config = config.get("model", {})
    input_features = model_config.get("input_features", "xyz_normal")
    input_channels = 6 if input_features == "xyz_normal" else 3
    return PointNet2InstanceSeg(
        num_classes=int(model_config.get("num_classes", 2)),
        input_channels=input_channels,
        sa_npoints=tuple(model_config.get("sa_npoints", [4096, 1024, 256])),
        sa_nsamples=tuple(model_config.get("sa_nsamples", [32, 32, 32])),
        dropout=float(model_config.get("dropout", 0.4)),
        embedding_dim=int(model_config.get("embedding_dim", 8)),
        ops_backend=str(model_config.get("ops_backend", "pure_torch")),
    )
