from typing import Optional

import torch
import torch.nn as nn

from cleandiffuser.utils import SinusoidalEmbedding
from cleandiffuser.nn_diffusion import BaseNNDiffusion


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """ A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning. """

    def __init__(self, hidden_size: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        def approx_gelu(): return nn.GELU(approximate="tanh")

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4), approx_gelu(), nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=1)
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x, x, x)[0]
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer1d(nn.Module):
    def __init__(self, hidden_size: int, out_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiT1d(BaseNNDiffusion):
    def __init__(
        self,
        in_dim: int,
        emb_dim: int,
        d_model: int = 384,
        n_heads: int = 6,
        depth: int = 12,
        dropout: float = 0.0,
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)
        self.in_dim, self.emb_dim = in_dim, emb_dim
        self.d_model = d_model

        self.x_proj = nn.Linear(in_dim, d_model)
        self.map_emb = nn.Sequential(
            nn.Linear(emb_dim, d_model), nn.Mish(), nn.Linear(d_model, d_model), nn.Mish())

        self.pos_emb = SinusoidalEmbedding(d_model)
        self.pos_emb_cache = None

        self.blocks = nn.ModuleList([
            DiTBlock(d_model, n_heads, dropout) for _ in range(depth)])
        self.final_layer = FinalLayer1d(d_model, in_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize time step embedding MLP:
        nn.init.normal_(self.map_emb[0].weight, std=0.02)
        nn.init.normal_(self.map_emb[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self,
                x: torch.Tensor, noise: torch.Tensor,
                condition: Optional[torch.Tensor] = None):
        """
        Input:
            x:          (b, horizon, in_dim)
            noise:      (b, )
            condition:  (b, emb_dim) or None / No condition indicates zeros((b, emb_dim))

        Output:
            y:          (b, horizon, in_dim)
        """
        if self.pos_emb_cache is None or self.pos_emb_cache.shape[0] != x.shape[1]:
            self.pos_emb_cache = self.pos_emb(torch.arange(x.shape[1], device=x.device))

        x = self.x_proj(x) + self.pos_emb_cache[None,]
        emb = self.map_noise(noise)
        if condition is not None:
            emb = emb + condition
        else:
            emb = emb + torch.zeros_like(emb)
        emb = self.map_emb(emb)

        for block in self.blocks:
            x = block(x, emb)
        x = self.final_layer(x, emb)
        return x


class DiT1Ref(DiT1d):
    def __init__(
            self,
            in_dim: int,
            emb_dim: int,
            d_model: int = 384,
            n_heads: int = 6,
            depth: int = 12,
            dropout: float = 0.0,
            timestep_emb_type: str = "positional",
    ):
        super().__init__(in_dim, emb_dim, d_model, n_heads, depth, dropout, timestep_emb_type)
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True) for _ in range(depth)])

    def forward(self,
                x: torch.Tensor, noise: torch.Tensor,
                condition: Optional[torch.Tensor] = None):
        """
        Input:
            x:          (b, horizon, in_dim * 2), where the first half is the reference signal
            noise:      (b, )
            condition:  (b, emb_dim) or None / No condition indicates zeros((b, emb_dim))

        Output:
            y:          (b, horizon, in_dim)
        """
        if self.pos_emb_cache is None or self.pos_emb_cache.shape[0] != x.shape[1]:
            self.pos_emb_cache = self.pos_emb(torch.arange(x.shape[1], device=x.device))

        x_ref, x = torch.chunk(x, 2, -1)
        x_ref_bkp = x_ref.clone()

        x_ref = self.x_proj(x_ref) + self.pos_emb_cache[None, ]
        x = self.x_proj(x) + self.pos_emb_cache[None, ]
        emb = self.map_noise(noise)

        if condition is not None:
            emb = emb + condition
        emb = self.map_emb(emb)

        for cross_attn, block in zip(self.cross_attns, self.blocks):
            x, _ = cross_attn(x, x_ref, x_ref)
            x = block(x, emb)
        x = self.final_layer(x, emb)
        return torch.cat([x_ref_bkp, x], -1)
