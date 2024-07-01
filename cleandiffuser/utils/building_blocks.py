import torch
import torch.nn as nn
import numpy as np
import einops


def generate_causal_mask(length: int, device: torch.device = "cpu"):
    mask = torch.tril(torch.ones(length, length, device=device), diagonal=0)
    return mask


class PreNorm(nn.Module):
    """ Layer normalization before the function.

    output = fn(LayerNorm(x), **kwargs)
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Residual(nn.Module):
    """ Residual connection.

    output = fn(x, **kwargs) + x
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_scale: int = 4, dropout: float = 0.):
        super().__init__()
        hidden_dim = int(d_model * hidden_scale)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(
            self, d_model: int, nhead: int, dropout: float = 0.1, bias: bool = False):
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
                assert mask.shape[0] == q.shape[0] and mask.shape[1] == q.shape[1] and mask.shape[2] == k.shape[1]
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


class Transformer(nn.Module):
    def __init__(
            self, d_model: int, nhead: int, num_layers: int, hidden_scale: int = 4,
            attn_dropout: float = 0., ffn_dropout: float = 0.,
            bias: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(d_model),
                MultiHeadAttention(d_model, nhead, attn_dropout, bias),
                nn.LayerNorm(d_model),
                FeedForward(d_model, hidden_scale, ffn_dropout)
            ]) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        attn_maps = []
        for (norm1, attn, norm2, ffn) in self.layers:
            _x = norm1(x)

            _x, attn_map = attn(_x, _x, _x, mask=mask)

            attn_maps.append(attn_map)

            x = _x + x

            _x = norm2(x)

            x = ffn(_x) + x

        return x, attn_maps
