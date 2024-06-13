import numpy as np
import torch
import torch.nn as nn

from cleandiffuser.nn_condition import MLPCondition


class FourierCondition(MLPCondition):
    """
    Fourier condition is a simple fourier basis to process the input condition.

    Input:
        - condition: (b, 1)
        - mask :     (b, ) or None, None means no mask

    Output:
        - condition: (b, out_dim)
    """

    def __init__(self, out_dim, hidden_dim, scale=16, dropout=0.25):
        super().__init__(hidden_dim, out_dim, hidden_dim, nn.Mish(), dropout)
        self.register_buffer('freqs', torch.randn(hidden_dim // 2) * scale)

    def forward(self, condition: torch.Tensor, mask: torch.Tensor = None):
        emb = condition.squeeze(-1).ger((2 * np.pi * self.freqs).to(condition.dtype))
        emb = torch.cat([emb.cos(), emb.sin()], -1)
        return super().forward(emb, mask)


class PositionalCondition(MLPCondition):
    """
    Positional condition is a simple positional encoding to process the input condition.

    Input:
        - condition: (b, 1)
        - mask :     (b, ) or None, None means no mask

    Output:
        - condition: (b, out_dim)
    """

    def __init__(self, out_dim, hidden_dim, dropout=0.25, max_positions: int = 10000, endpoint: bool = False):
        super().__init__(hidden_dim, out_dim, hidden_dim, nn.Mish(), dropout)
        self.max_positions = max_positions
        self.endpoint = endpoint
        self.dim = out_dim

    def forward(self, condition: torch.Tensor, mask: torch.Tensor = None):
        freqs = torch.arange(start=0, end=self.dim // 2, dtype=torch.float32, device=condition.device)
        freqs = freqs / (self.dim // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = condition.ger(freqs.to(condition.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return super().forward(x, mask)
