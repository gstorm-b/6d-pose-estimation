"""Pure-PyTorch PointNet++ helper operations."""

from __future__ import annotations

import torch


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


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
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


def knn_indices(nsample: int, xyz: torch.Tensor, query_xyz: torch.Tensor) -> torch.Tensor:
    """Return nearest-neighbor indices for each query point."""

    nsample = min(nsample, xyz.shape[1])
    distances = square_distance(query_xyz, xyz)
    return torch.topk(distances, k=nsample, dim=-1, largest=False, sorted=False)[1]


def three_nn_interpolate(
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
