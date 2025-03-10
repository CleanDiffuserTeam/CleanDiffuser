from typing import List

import einops
import numpy as np
import torch
import torch.nn as nn


class Mlp(nn.Module):
    """**Multilayer perceptron.** A simple pytorch MLP module.

    Args:
        in_dim: int,
            The dimension of the input tensor.
        hidden_dims: List[int],
            A list of integers, each element is the dimension of the hidden layer.
        out_dim: int,
            The dimension of the output tensor.
        activation: nn.Module,
            The activation function used in the hidden layers.
        out_activation: nn.Module,
            The activation function used in the output layer.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        activation: nn.Module = nn.ReLU(),
        out_activation: nn.Module = nn.Identity(),
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(in_dim if i == 0 else hidden_dims[i - 1], hidden_dims[i]),
                    activation,
                )
                for i in range(len(hidden_dims))
            ],
            nn.Linear(hidden_dims[-1], out_dim),
            out_activation,
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.mlp(x)


class GroupNorm1d(nn.Module):
    def __init__(self, dim, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, dim // min_channels_per_group)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = torch.nn.functional.group_norm(
            x.unsqueeze(2),
            num_groups=self.num_groups,
            weight=self.weight.to(x.dtype),
            bias=self.bias.to(x.dtype),
            eps=self.eps,
        )
        return x.squeeze(2)


class SoftLowerBound(nn.Module):
    """Soft lower bound.

    Args:
        lower_bound: float,
            The lower bound of the output.
    """

    def __init__(self, lower_bound: float):
        super().__init__()
        self.lower_bound = lower_bound

    def forward(self, x):
        return self.lower_bound + torch.nn.functional.softplus(x - self.lower_bound)


class SoftUpperBound(nn.Module):
    """Soft upper bound.

    Args:
        upper_bound: float,
            The upper bound of the output.
    """

    def __init__(self, upper_bound: float):
        super().__init__()
        self.upper_bound = upper_bound

    def forward(self, x):
        return self.upper_bound - torch.nn.functional.softplus(self.upper_bound - x)


class PreNorm(nn.Module):
    """Layer normalization before the function.

    output = fn(LayerNorm(x), **kwargs)
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Residual(nn.Module):
    """Residual connection.

    output = fn(x, **kwargs) + x
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_scale: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(d_model * hidden_scale)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        assert d_model % nhead == 0, "`d_model` must be divisible by `nhead`."

        self.nhead, self.d_k = nhead, d_model // nhead
        self.scale = 1 / np.sqrt(self.d_k)

        self.q_layer = nn.Linear(d_model, d_model, bias=bias)
        self.k_layer = nn.Linear(d_model, d_model, bias=bias)
        self.v_layer = nn.Linear(d_model, d_model, bias=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        if mask is not None:
            if mask.dim() == 2:
                assert mask.shape[0] == q.shape[1] and mask.shape[1] == k.shape[1]
                mask = mask.unsqueeze(0)
            elif mask.dim() == 3:
                assert (
                    mask.shape[0] == q.shape[0]
                    and mask.shape[1] == q.shape[1]
                    and mask.shape[2] == k.shape[1]
                )
            else:
                raise ValueError("`mask` shape should be either (i, j) or (b, i, j)")
            mask = mask.unsqueeze(-1)

        q = einops.rearrange(self.q_layer(q), "b i (h d) -> b i h d", h=self.nhead)
        k = einops.rearrange(self.k_layer(k), "b j (h d) -> b j h d", h=self.nhead)
        v = einops.rearrange(self.v_layer(v), "b j (h d) -> b j h d", h=self.nhead)

        scores = torch.einsum("b i h d, b j h d -> b i j h", q, k) * self.scale
        if mask is not None:
            scores.masked_fill_(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=2)

        attn = self.dropout(attn)

        out = torch.einsum("b i j h, b j h d -> b i h d", attn, v)

        out = einops.rearrange(out, "b i h d -> b i (h d)")

        return out, attn.detach()


def generate_causal_mask(length: int, device: torch.device = "cpu"):
    """Generate a causal mask, where 1 means visible and 0 means invisible."""
    mask = torch.tril(torch.ones(length, length, device=device), diagonal=0)
    return mask


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        hidden_scale: int = 4,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.LayerNorm(d_model),
                        MultiHeadAttention(d_model, nhead, attn_dropout, bias),
                        nn.LayerNorm(d_model),
                        FeedForward(d_model, hidden_scale, ffn_dropout),
                    ]
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask=None):
        attn_maps = []
        for norm1, attn, norm2, ffn in self.layers:
            _x = norm1(x)

            _x, attn_map = attn(_x, _x, _x, mask=mask)

            attn_maps.append(attn_map)

            x = _x + x

            _x = norm2(x)

            x = ffn(_x) + x

        return x, attn_maps
