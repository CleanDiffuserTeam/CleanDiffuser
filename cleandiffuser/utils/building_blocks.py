import torch
import torch.nn as nn
import numpy as np
import einops
from typing import List
from .iql import TwinQ, V
from cleandiffuser.utils import SinusoidalEmbedding

IDQLQNet = TwinQ
IDQLVNet = V


class Mlp(nn.Module):
    """ **Multilayer perceptron.** A simple pytorch MLP module.

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
            out_activation
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
    """ Soft lower bound.

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
    """ Soft upper bound.

    Args:
        upper_bound: float,
            The upper bound of the output.
    """

    def __init__(self, upper_bound: float):
        super().__init__()
        self.upper_bound = upper_bound

    def forward(self, x):
        return self.upper_bound - torch.nn.functional.softplus(self.upper_bound - x)


class DQLCritic(nn.Module):
    """ **Deep Q-Learning Critic.** A pytorch critic module for DQL. The module incorporates double Q trick.

    Args:
        obs_dim: int,
            The dimension of the observation space.
        act_dim: int,
            The dimension of the action space.
        hidden_dim: int,
            The dimension of the hidden layers.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q1_model = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, 1))

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1_model(x)

    def q_min(self, obs, act):
        q1, q2 = self.forward(obs, act)
        return torch.min(q1, q2)

class DVTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, dropout: float = 0.0, norm_type="post"):
        super().__init__()
        self.norm_type = norm_type
        
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        def approx_gelu(): return nn.GELU(approximate="tanh")

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4), approx_gelu(), nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size))

    def forward(self, x: torch.Tensor):
        if self.norm_type == "post":
            x = self.norm1(x + self.attn(x, x, x)[0])
            x = self.norm2(x + self.mlp(x))
        elif self.norm_type == "pre":
            x = self.norm1(x)
            x = x + self.attn(x, x, x)[0]
            x = x + self.mlp(self.norm2(x))
        else:
            raise NotImplementedError
        return x
    
class DVHorizonCritic(nn.Module):
    def __init__(
        self,
        in_dim: int,
        emb_dim: int,
        d_model: int = 384,
        n_heads: int = 6,
        depth: int = 12,
        dropout: float = 0.0,
        norm_type: str = "post"
    ):
        super().__init__()
        self.in_dim, self.emb_dim = in_dim, emb_dim
        self.d_model = d_model

        self.x_proj = nn.Linear(in_dim, d_model)

        self.pos_emb = SinusoidalEmbedding(d_model)
        self.pos_emb_cache = None

        self.blocks = nn.ModuleList([DVTransformerBlock(d_model, n_heads, dropout, norm_type) for _ in range(depth)])
        self.final_layer = nn.Linear(d_model, 1)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(self, x: torch.Tensor):
        """
        Input:
            x:          (b, horizon, in_dim)

        Output:
            y:          (b, horizon, in_dim)
        """
        if self.pos_emb_cache is None or self.pos_emb_cache.shape[0] != x.shape[1]:
            self.pos_emb_cache = self.pos_emb(torch.arange(x.shape[1], device=x.device))

        x = self.x_proj(x) + self.pos_emb_cache[None,]

        for block in self.blocks:
            x = block(x)
        x = self.final_layer(x)
        
        x = x[:, 0, :]
        
        return x

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
    def __init__(
            self, d_model: int, nhead: int, dropout: float = 0.1, bias: bool = False
    ):
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
