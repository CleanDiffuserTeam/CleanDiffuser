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
        centroids (torch.Tensor):
            The centroids with shape (B, 3, M).
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
