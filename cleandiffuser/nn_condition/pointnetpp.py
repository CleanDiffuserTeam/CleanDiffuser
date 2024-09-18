import numpy as np
import torch


def fps(x: torch.Tensor, M: int) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS) for point cloud data.

    Args:
        x (torch.Tensor):
            Point cloud tensor with shape (B, 3, N).
        M (int):
            Expected number of centroids.

    Returns:
        idx (torch.Tensor):
            The index of the centroids with shape (B, M).
    """
    B, D, N = x.shape

    centroids = torch.zeros(B, M, D, dtype=x.dtype, device=x.device)
    points = x.permute(0, 2, 1)
    idx = torch.zeros(B, M, dtype=torch.long, device=x.device)

    # Randomly initialize the first centroid for each batch
    farthest_idx = torch.randint(0, N, (B,), dtype=torch.long, device=x.device)
    idx[:, 0] = farthest_idx
    centroids[:, 0] = points[torch.arange(B), farthest_idx, :]

    for m in range(1, M):
        dist = ((points.unsqueeze(2) - centroids[:, :m].unsqueeze(1)) ** 2).sum(-1).min(-1)[0]
        farthest_idx = dist.argmax(-1)
        idx[:, m] = farthest_idx
        centroids[:, m] = points[torch.arange(B), farthest_idx, :]

    return idx, centroids.permute(0, 2, 1)


def grouping(x: torch.Tensor, centroids: torch.Tensor, radius: float, K: int) -> torch.Tensor:
    """
    Grouping operation for point cloud data.

    Args:
        x (torch.Tensor):
            Point cloud tensor with shape (B, 3, N).
        centroids (torch.Tensor):
            Centroid tensor with shape (B, 3, M).
        radius (float):
            Radius for grouping.
        K (int):
            Expected number of points in each group.

    Returns:
        group_points (torch.Tensor):
            The grouped points with shape (B, M, K, 3).
    """
    B, D, N = x.shape
    M = centroids.shape[-1]

    dist = ((x.unsqueeze(-1) - centroids.unsqueeze(-2)) ** 2).sum(1).sqrt()  # (B, N, M)
    group_idx = torch.argsort(dist, dim=1)[:, :K, :]
    group_dist = torch.gather(dist, 1, group_idx)
    mask = group_dist > radius
    group_idx[mask] = group_idx[:, :1].expand(B, K, M)[mask]
    group_idx = group_idx.permute(0, 2, 1)
    group_points = torch.gather(x.unsqueeze(1).expand(B, M, D, N), 3, group_idx.unsqueeze(2).expand(B, M, D, K))

    return group_points.permute(0, 1, 3, 2)


if __name__ == "__main__":
    x = torch.cat([torch.randn((2, 3, 1000)) + 3, torch.randn((2, 3, 1000)) - 3], -1)

    idx, y = fps(x, 4)

    z = grouping(x, y, 1.0, 16)
