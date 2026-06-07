"""PointNet++ semantic segmentation MVP."""

from __future__ import annotations

import torch
from torch import nn

from .pointnet2_ops import PointNet2Ops, build_pointnet2_ops_backend, index_points


class SharedMlp2d(nn.Sequential):
    def __init__(self, channels: list[int]) -> None:
        layers: list[nn.Module] = []
        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
        super().__init__(*layers)


class SharedMlp1d(nn.Sequential):
    def __init__(self, channels: list[int], *, final_activation: bool = True) -> None:
        layers: list[nn.Module] = []
        pairs = list(zip(channels[:-1], channels[1:]))
        for idx, (in_channels, out_channels) in enumerate(pairs):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=idx == len(pairs) - 1))
            if final_activation or idx < len(pairs) - 1:
                layers.extend([nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True)])
        super().__init__(*layers)


class SetAbstraction(nn.Module):
    def __init__(
        self,
        npoint: int,
        nsample: int,
        in_channels: int,
        mlp_channels: list[int],
        ops_backend: PointNet2Ops,
    ) -> None:
        super().__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.ops_backend = ops_backend
        self.mlp = SharedMlp2d([in_channels, *mlp_channels])

    def forward(
        self,
        xyz: torch.Tensor,
        features: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fps_indices = self.ops_backend.farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_indices)
        group_indices = self.ops_backend.knn_indices(self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, group_indices)
        grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

        if features is None:
            grouped_features = grouped_xyz_norm
        else:
            features_bnc = features.transpose(1, 2).contiguous()
            grouped_point_features = index_points(features_bnc, group_indices)
            grouped_features = torch.cat([grouped_xyz_norm, grouped_point_features], dim=-1)

        grouped_features = grouped_features.permute(0, 3, 2, 1).contiguous()
        new_features = self.mlp(grouped_features).max(dim=2)[0]
        return new_xyz, new_features


class FeaturePropagation(nn.Module):
    def __init__(self, in_channels: int, mlp_channels: list[int], ops_backend: PointNet2Ops) -> None:
        super().__init__()
        self.ops_backend = ops_backend
        self.mlp = SharedMlp1d([in_channels, *mlp_channels])

    def forward(
        self,
        target_xyz: torch.Tensor,
        source_xyz: torch.Tensor,
        target_features: torch.Tensor | None,
        source_features: torch.Tensor,
    ) -> torch.Tensor:
        interpolated = self.ops_backend.three_nn_interpolate(source_xyz, target_xyz, source_features)
        if target_features is not None:
            interpolated = torch.cat([target_features, interpolated], dim=1)
        return self.mlp(interpolated)


class PointNet2Backbone(nn.Module):
    """Reusable PointNet++ encoder-decoder that returns per-point features."""

    output_channels = 64

    def __init__(
        self,
        input_channels: int = 6,
        sa_npoints: tuple[int, int, int] = (4096, 1024, 256),
        sa_nsamples: tuple[int, int, int] = (32, 32, 32),
        ops_backend: str = "pure_torch",
    ) -> None:
        super().__init__()
        if input_channels < 3:
            raise ValueError("input_channels must include xyz and be at least 3")
        extra_channels = input_channels - 3
        self.input_channels = input_channels
        self.ops_backend = build_pointnet2_ops_backend(ops_backend)
        self.ops_backend_name = self.ops_backend.name

        self.sa1 = SetAbstraction(sa_npoints[0], sa_nsamples[0], 3 + extra_channels, [64, 64, 64], self.ops_backend)
        self.sa2 = SetAbstraction(sa_npoints[1], sa_nsamples[1], 3 + 64, [128, 128, 128], self.ops_backend)
        self.sa3 = SetAbstraction(sa_npoints[2], sa_nsamples[2], 3 + 128, [256, 256, 256], self.ops_backend)

        self.fp3 = FeaturePropagation(128 + 256, [256, 128], self.ops_backend)
        self.fp2 = FeaturePropagation(64 + 128, [128, 64], self.ops_backend)
        self.fp1 = FeaturePropagation(extra_channels + 64, [64, 64], self.ops_backend)

    def forward(self, point_features: torch.Tensor) -> torch.Tensor:
        """Return per-point features with shape `(B, 64, N)`."""

        if point_features.ndim != 3 or point_features.shape[-1] < 3:
            raise ValueError(f"expected point tensor `(B, N, C>=3)`, got {tuple(point_features.shape)}")
        xyz0 = point_features[:, :, :3].contiguous()
        features0 = point_features[:, :, 3:].transpose(1, 2).contiguous() if point_features.shape[-1] > 3 else None

        xyz1, features1 = self.sa1(xyz0, features0)
        xyz2, features2 = self.sa2(xyz1, features1)
        xyz3, features3 = self.sa3(xyz2, features2)

        features2_up = self.fp3(xyz2, xyz3, features2, features3)
        features1_up = self.fp2(xyz1, xyz2, features1, features2_up)
        return self.fp1(xyz0, xyz1, features0, features1_up)


class PointNet2SemSeg(nn.Module):
    """PointNet++ semantic segmentation for `(B, N, C)` point tensors."""

    def __init__(
        self,
        num_classes: int = 2,
        input_channels: int = 6,
        sa_npoints: tuple[int, int, int] = (4096, 1024, 256),
        sa_nsamples: tuple[int, int, int] = (32, 32, 32),
        dropout: float = 0.4,
        ops_backend: str = "pure_torch",
    ) -> None:
        super().__init__()
        if input_channels < 3:
            raise ValueError("input_channels must include xyz and be at least 3")
        extra_channels = input_channels - 3
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.ops_backend = build_pointnet2_ops_backend(ops_backend)
        self.ops_backend_name = self.ops_backend.name

        self.sa1 = SetAbstraction(sa_npoints[0], sa_nsamples[0], 3 + extra_channels, [64, 64, 64], self.ops_backend)
        self.sa2 = SetAbstraction(sa_npoints[1], sa_nsamples[1], 3 + 64, [128, 128, 128], self.ops_backend)
        self.sa3 = SetAbstraction(sa_npoints[2], sa_nsamples[2], 3 + 128, [256, 256, 256], self.ops_backend)

        self.fp3 = FeaturePropagation(128 + 256, [256, 128], self.ops_backend)
        self.fp2 = FeaturePropagation(64 + 128, [128, 64], self.ops_backend)
        self.fp1 = FeaturePropagation(extra_channels + 64, [64, 64], self.ops_backend)

        self.classifier = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(64, num_classes, kernel_size=1),
        )

    def forward(self, point_features: torch.Tensor) -> torch.Tensor:
        """Return logits with shape `(B, N, num_classes)`."""

        if point_features.ndim != 3 or point_features.shape[-1] < 3:
            raise ValueError(f"expected point tensor `(B, N, C>=3)`, got {tuple(point_features.shape)}")
        xyz0 = point_features[:, :, :3].contiguous()
        features0 = point_features[:, :, 3:].transpose(1, 2).contiguous() if point_features.shape[-1] > 3 else None

        xyz1, features1 = self.sa1(xyz0, features0)
        xyz2, features2 = self.sa2(xyz1, features1)
        xyz3, features3 = self.sa3(xyz2, features2)

        features2_up = self.fp3(xyz2, xyz3, features2, features3)
        features1_up = self.fp2(xyz1, xyz2, features1, features2_up)
        features0_up = self.fp1(xyz0, xyz1, features0, features1_up)
        logits = self.classifier(features0_up)
        return logits.transpose(1, 2).contiguous()


def build_pointnet2_semseg_from_config(config: dict) -> PointNet2SemSeg:
    model_config = config.get("model", {})
    input_features = model_config.get("input_features", "xyz_normal")
    input_channels = 6 if input_features == "xyz_normal" else 3
    return PointNet2SemSeg(
        num_classes=int(model_config.get("num_classes", 2)),
        input_channels=input_channels,
        sa_npoints=tuple(model_config.get("sa_npoints", [4096, 1024, 256])),
        sa_nsamples=tuple(model_config.get("sa_nsamples", [32, 32, 32])),
        dropout=float(model_config.get("dropout", 0.4)),
        ops_backend=str(model_config.get("ops_backend", "pure_torch")),
    )
