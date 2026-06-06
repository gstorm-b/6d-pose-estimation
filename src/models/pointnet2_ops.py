"""PointNet++ helper operations with optional optimized backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class PointNet2Ops(Protocol):
    """Backend API used by the PointNet++ modules."""

    name: str

    def farthest_point_sample(self, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """Return farthest-point-sampling indices with shape `(B, npoint)`."""

    def knn_indices(self, nsample: int, xyz: torch.Tensor, query_xyz: torch.Tensor) -> torch.Tensor:
        """Return nearest-neighbor indices with shape `(B, query_count, nsample)`."""

    def three_nn_interpolate(
        self,
        source_xyz: torch.Tensor,
        target_xyz: torch.Tensor,
        source_features: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate sparse source features onto target points."""


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Return squared distances between two point sets.

    Args:
        src: `(B, N, 3)`.
        dst: `(B, M, 3)`.
    """

    return torch.cdist(src, dst, p=2) ** 2


def index_points(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather points with batched indices."""

    batch_size = points.shape[0]
    view_shape = [batch_size] + [1] * (indices.ndim - 1)
    repeat_shape = [1] + list(indices.shape[1:])
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, indices]


def farthest_point_sample_torch(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest point sampling.

    This MVP implementation favors readability over speed. It is acceptable for
    debug runs and can be swapped for optimized kernels later.
    """

    batch_size, point_count, _ = xyz.shape
    npoint = min(npoint, point_count)
    centroids = torch.zeros(batch_size, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.full((batch_size, point_count), 1e10, device=xyz.device)
    farthest = torch.randint(0, point_count, (batch_size,), dtype=torch.long, device=xyz.device)
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=xyz.device)
    for idx in range(npoint):
        centroids[:, idx] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch_size, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


def knn_indices_torch(nsample: int, xyz: torch.Tensor, query_xyz: torch.Tensor) -> torch.Tensor:
    """Return nearest-neighbor indices for each query point."""

    nsample = min(nsample, xyz.shape[1])
    distances = square_distance(query_xyz, xyz)
    return torch.topk(distances, k=nsample, dim=-1, largest=False, sorted=False)[1]


def three_nn_interpolate_torch(
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    source_features: torch.Tensor,
) -> torch.Tensor:
    """Interpolate source features from sparse source points to dense target points.

    Args:
        source_xyz: `(B, S, 3)`.
        target_xyz: `(B, N, 3)`.
        source_features: `(B, C, S)`.

    Returns:
        `(B, C, N)`.
    """

    if source_xyz.shape[1] == 1:
        return source_features.repeat(1, 1, target_xyz.shape[1])
    k = min(3, source_xyz.shape[1])
    distances = torch.cdist(target_xyz, source_xyz, p=2)
    dists, indices = torch.topk(distances, k=k, dim=-1, largest=False, sorted=False)
    inv_dist = 1.0 / torch.clamp(dists, min=1e-10)
    weights = inv_dist / inv_dist.sum(dim=-1, keepdim=True)
    grouped = index_points(source_features.transpose(1, 2), indices)
    return torch.sum(grouped * weights.unsqueeze(-1), dim=2).transpose(1, 2).contiguous()


def farthest_point_sample_pytorch3d(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest point sampling backed by PyTorch3D's CUDA extension."""

    from pytorch3d.ops import sample_farthest_points

    _, indices = sample_farthest_points(xyz, K=min(npoint, xyz.shape[1]), random_start_point=True)
    return indices


def knn_indices_pytorch3d(nsample: int, xyz: torch.Tensor, query_xyz: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbor indices backed by PyTorch3D's CUDA extension."""

    from pytorch3d.ops import knn_points

    return knn_points(query_xyz, xyz, K=min(nsample, xyz.shape[1]), return_nn=False).idx


def three_nn_interpolate_pytorch3d(
    source_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    source_features: torch.Tensor,
) -> torch.Tensor:
    """Three-nearest-neighbor interpolation backed by PyTorch3D KNN/gather."""

    from pytorch3d.ops import knn_gather, knn_points

    if source_xyz.shape[1] == 1:
        return source_features.repeat(1, 1, target_xyz.shape[1])
    k = min(3, source_xyz.shape[1])
    result = knn_points(target_xyz, source_xyz, K=k, return_nn=False)
    distances = torch.sqrt(torch.clamp(result.dists, min=1e-20))
    inv_dist = 1.0 / torch.clamp(distances, min=1e-10)
    weights = inv_dist / inv_dist.sum(dim=-1, keepdim=True)
    grouped = knn_gather(source_features.transpose(1, 2).contiguous(), result.idx)
    return torch.sum(grouped * weights.unsqueeze(-1), dim=2).transpose(1, 2).contiguous()


@dataclass(frozen=True)
class TorchPointNet2Ops:
    """Pure-PyTorch PointNet++ operation backend."""

    name: str = "pure_torch"

    def farthest_point_sample(self, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        return farthest_point_sample_torch(xyz, npoint)

    def knn_indices(self, nsample: int, xyz: torch.Tensor, query_xyz: torch.Tensor) -> torch.Tensor:
        return knn_indices_torch(nsample, xyz, query_xyz)

    def three_nn_interpolate(
        self,
        source_xyz: torch.Tensor,
        target_xyz: torch.Tensor,
        source_features: torch.Tensor,
    ) -> torch.Tensor:
        return three_nn_interpolate_torch(source_xyz, target_xyz, source_features)


@dataclass(frozen=True)
class PyTorch3DPointNet2Ops:
    """PyTorch3D-backed PointNet++ operation backend."""

    name: str = "pytorch3d"

    def farthest_point_sample(self, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        return farthest_point_sample_pytorch3d(xyz, npoint)

    def knn_indices(self, nsample: int, xyz: torch.Tensor, query_xyz: torch.Tensor) -> torch.Tensor:
        return knn_indices_pytorch3d(nsample, xyz, query_xyz)

    def three_nn_interpolate(
        self,
        source_xyz: torch.Tensor,
        target_xyz: torch.Tensor,
        source_features: torch.Tensor,
    ) -> torch.Tensor:
        return three_nn_interpolate_pytorch3d(source_xyz, target_xyz, source_features)


def is_pytorch3d_available() -> bool:
    """Return whether PyTorch3D point-cloud ops can be imported."""

    try:
        from pytorch3d.ops import knn_gather, knn_points, sample_farthest_points  # noqa: F401
    except Exception:  # noqa: BLE001 - import can fail if compiled extension is missing.
        return False
    return True


def build_pointnet2_ops_backend(backend: str = "pure_torch", *, allow_fallback: bool = True) -> PointNet2Ops:
    """Build a PointNet++ ops backend by name."""

    normalized = backend.lower().replace("-", "_")
    if normalized in {"torch", "pure_torch", "pytorch"}:
        return TorchPointNet2Ops()
    if normalized == "pytorch3d":
        if is_pytorch3d_available():
            return PyTorch3DPointNet2Ops()
        if allow_fallback:
            return TorchPointNet2Ops()
        raise ImportError("PyTorch3D ops backend requested, but PyTorch3D could not be imported.")
    raise ValueError(f"Unknown PointNet++ ops backend: {backend}")


# Backward-compatible function names for existing imports and smoke snippets.
farthest_point_sample = farthest_point_sample_torch
knn_indices = knn_indices_torch
three_nn_interpolate = three_nn_interpolate_torch
