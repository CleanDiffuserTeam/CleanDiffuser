from typing import Dict, Optional

import torch
import torch.nn as nn

from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import UntrainablePositionalEmbedding


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(self, hidden_size: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        def approx_gelu():
            return nn.GELU(approximate="tanh")

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            approx_gelu(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            t
        ).chunk(6, dim=1)
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x, x, x)[0]
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer1d(nn.Module):
    def __init__(self, hidden_size: int, out_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiT1d(BaseNNDiffusion):
    """Temporal Diffusion Transformer (DiT) backbone used in AlignDiff.

    DiT for temporal diffusion models. It can be used for variable length sequences.

    Args:
        x_dim (int):
            The dimension of the input. Input tensors are assumed to be in shape of (b, horizon, x_dim),
            where `horizon` can be variable.
        emb_dim (int):
            The dimension of the timestep embedding and condition embedding.
        d_model (int):
            The dimension of the transformer model. Default: 384
        n_heads (int):
            The number of heads in the transformer model. Default: 6
        depth (int):
            The number of transformer layers. Default: 12
        dropout (float):
            The dropout rate. Default: 0.0
        timestep_emb_type (str):
            The type of the timestep embedding. Default: "positional"
        timestep_emb_params (dict):
            The parameters of the timestep embedding. Default: None

    Examples:
        >>> model = DiT1d(x_dim=10, emb_dim=16)
        >>> x = torch.randn((2, 20, 10))
        >>> t = torch.randint(1000, (2,))
        >>> model(x, t).shape
        torch.Size([2, 20, 10])
        >>> x = torch.randn((2, 40, 10))
        >>> condition = torch.randn((2, 16))
        >>> model(x, t, condition).shape
        torch.Size([2, 40, 10])
    """

    def __init__(
        self,
        x_dim: int,
        emb_dim: int,
        d_model: int = 384,
        n_heads: int = 6,
        depth: int = 12,
        dropout: float = 0.0,
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None,
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)
        self.x_proj = nn.Linear(x_dim, d_model)
        self.map_emb = nn.Sequential(
            nn.Linear(emb_dim, d_model), nn.Mish(), nn.Linear(d_model, d_model), nn.Mish()
        )

        self.pos_emb = UntrainablePositionalEmbedding(d_model)
        self.pos_emb_cache = None

        self.blocks = nn.ModuleList([DiTBlock(d_model, n_heads, dropout) for _ in range(depth)])
        self.final_layer = FinalLayer1d(d_model, x_dim)
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

    def forward(
        self, x: torch.Tensor, noise: torch.Tensor, condition: Optional[torch.Tensor] = None
    ):
        if self.pos_emb_cache is None or self.pos_emb_cache.shape[0] != x.shape[1]:
            self.pos_emb_cache = self.pos_emb(torch.arange(x.shape[1], device=x.device))

        x = self.x_proj(x) + self.pos_emb_cache[None,]
        emb = self.map_noise(noise)
        if condition is not None:
            emb = emb + condition
        emb = self.map_emb(emb)

        for block in self.blocks:
            x = block(x, emb)
        x = self.final_layer(x, emb)
        return x


class DiT1dV2(BaseNNDiffusion):
    def __init__(
        self,
        x_dim: int,
        x_seq_len: int,
        emb_dim: int,
        d_model: int = 384,
        n_heads: int = 6,
        depth: int = 12,
        dropout: float = 0.0,
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None,
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)
        self.d_model = d_model
        self.x_proj = nn.Linear(x_dim, d_model)
        self.t_proj = nn.Sequential(
            nn.Linear(emb_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.cond_proj = nn.Sequential(nn.Linear(emb_dim, d_model), nn.LayerNorm(d_model))

        # learnable positional embedding
        self.pos_emb = nn.Parameter(torch.randn(1, x_seq_len, d_model) * 0.02)

        self.blocks = nn.ModuleList([DiTBlock(d_model, n_heads, dropout) for _ in range(depth)])
        self.final_layer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(approximate="tanh"),
            nn.Linear(d_model, x_dim),
        )
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
        nn.init.normal_(self.t_proj[0].weight, std=0.02)
        nn.init.normal_(self.t_proj[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer[3].weight, 0)
        nn.init.constant_(self.final_layer[3].bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: Optional[torch.Tensor] = None):
        t_emb = self.t_proj(self.map_noise(t))
        x_emb = self.x_proj(x) + self.pos_emb

        if condition is None:
            cond_emb = torch.zeros_like(t_emb)
        else:
            cond_emb = self.cond_proj(condition) + t_emb

        for block in self.blocks:
            x_emb = block(x_emb, cond_emb)

        x_emb = self.final_layer(x_emb)

        return x_emb


class DiTBlockWithCrossAttention(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(self, hidden_size: int, n_heads: int, dropout: float = 0.0):
        super().__init__()

        # Self Attention Layer
        self.sa_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.sa_attn = nn.MultiheadAttention(hidden_size, n_heads, dropout, batch_first=True)

        # Cross Attention Layer
        self.ca_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ca_attn = nn.MultiheadAttention(hidden_size, n_heads, dropout, batch_first=True)

        # Feed Forward Layer
        self.ffn_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

        # adaLN
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 9))

    def forward(
        self,
        x: torch.Tensor,
        vec_condition: torch.Tensor,
        seq_condition: torch.Tensor,
        seq_condition_mask: Optional[torch.Tensor] = None,
    ):
        (
            shift_sa,
            scale_sa,
            gate_sa,
            shift_ca,
            scale_ca,
            gate_ca,
            shift_ffn,
            scale_ffn,
            gate_ffn,
        ) = self.adaLN_modulation(vec_condition.unsqueeze(-2)).chunk(9, dim=-1)  # (b, 1, 1)

        h = self.sa_norm(x) * (1 + scale_sa) + shift_sa
        x = x + gate_sa * self.sa_attn(h, h, h)[0]

        h = self.ca_norm(x) * (1 + scale_ca) + shift_ca
        x = (
            x
            + gate_ca
            * self.ca_attn(h, seq_condition, seq_condition, key_padding_mask=seq_condition_mask)[0]
        )

        h = self.ffn_norm(x) * (1 + scale_ffn) + shift_ffn
        x = x + gate_ffn * self.mlp(h)

        return x


class DiT1dWithCrossAttention(BaseNNDiffusion):
    def __init__(
        self,
        x_dim: int,
        x_seq_len: int,
        emb_dim: int,
        d_model: int = 384,
        n_heads: int = 6,
        depth: int = 12,
        dropout: float = 0.0,
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None,
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)
        self.x_proj = nn.Linear(x_dim, d_model)
        self.t_proj = nn.Sequential(
            nn.Linear(emb_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )

        # learnable positional embedding
        self.pos_emb = nn.Parameter(torch.randn(1, x_seq_len, d_model) * 0.02)

        self.blocks = nn.ModuleList(
            [DiTBlockWithCrossAttentionPostNorm(d_model, n_heads, dropout) for _ in range(depth)]
        )
        self.final_layer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(approximate="tanh"),
            nn.Linear(d_model, x_dim),
        )
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
        nn.init.normal_(self.t_proj[0].weight, std=0.02)
        nn.init.normal_(self.t_proj[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer[3].weight, 0)
        nn.init.constant_(self.final_layer[3].bias, 0)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, condition: Optional[Dict[str, torch.Tensor]] = None
    ):
        vec_condition = condition["vec_condition"]
        seq_condition = condition["seq_condition"]
        seq_condition_mask = condition["seq_condition_mask"]

        x = self.x_proj(x) + self.pos_emb
        emb = self.t_proj(self.map_noise(t))

        if condition is not None and vec_condition is not None:
            emb = emb + vec_condition

        for block in self.blocks:
            x = block(x, emb, seq_condition, seq_condition_mask)

        x = self.final_layer(x)

        return x


class DiT1dWithACICrossAttention(BaseNNDiffusion):
    """DiT1d with ACI (Alternating Condition Injection) cross-attention.

    Vanilla DiT uses adaLN to inject vector condition into the transformer layers.
    To inject sequence condition, one can add a cross-attention layer in each block,
    and produce the sequence condition as the key and value.
    For vision-language conditions, given that image tokens are usually much more than text tokens,
    simultaneous injection of both modalities tends to overshadow text-related information,
    thus impairing the capability of the instruction following. To mitigate this issue,
    RDT proposes to strategically alternate between injecting image and text tokens in successive layers'
    cross-attention rather than injecting both in every layer, which is called ACI (Alternating Condition Injection).

    Reference: https://arxiv.org/pdf/2410.07864

    The condition should be a dictionary containing the following keys:
        - vec_condition (torch.Tensor): The vector condition in shape of (b, emb_dim).
        - lang_condition (torch.Tensor): The language embedding in shape of (b, lang_seq_len, emb_dim).
        - lang_condition_mask (torch.Tensor): The language mask in shape of (b, lang_seq_len) or None.
            If it is a boolean tensor, True means masked. If a float tensor, it will be directly added to the corresponding ``key`` value.
        - vis_condition (torch.Tensor): The visual embedding in shape of (b, vis_seq_len, emb_dim).
        - vis_condition_mask (torch.Tensor): The visual mask in shape of (b, vis_seq_len) or None.
            If it is a boolean tensor, True means masked. If a float tensor, it will be directly added to the corresponding ``key`` value.

    Args:
        x_dim (int):
            The dimension of the input. Input tensors are assumed to be in shape of (b, x_seq_len, x_dim),
        x_seq_len (int):
            The length of the input sequence.
        emb_dim (int):
            The dimension of the timestep embedding and condition embedding.
        d_model (int):
            The dimension of the transformer model. Default: 384
        n_heads (int):
            The number of heads in the transformer model. Default: 6
        depth (int):
            The number of transformer layers. Default: 12
        dropout (float):
            Classifier-free guidance condition dropout rate. Default: 0.0
        timestep_emb_type (str):
            The type of the timestep embedding. Default: "positional"
        timestep_emb_params (dict):
            The parameters of the timestep embedding. Default: None
    """

    def __init__(
        self,
        x_dim: int,
        x_seq_len: int,
        emb_dim: int,
        d_model: int = 384,
        n_heads: int = 6,
        depth: int = 12,
        dropout: float = 0.0,
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None,
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)
        self.x_proj = nn.Linear(x_dim, d_model)
        self.t_proj = nn.Sequential(
            nn.Linear(emb_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )

        # learnable positional embedding
        self.pos_emb = nn.Parameter(torch.randn(1, x_seq_len, d_model) * 0.02)

        self.blocks = nn.ModuleList(
            [DiTBlockWithCrossAttentionPostNorm(d_model, n_heads, dropout) for _ in range(depth)]
        )
        self.final_layer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(approximate="tanh"),
            nn.Linear(d_model, x_dim),
        )
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
        nn.init.normal_(self.t_proj[0].weight, std=0.02)
        nn.init.normal_(self.t_proj[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer[3].weight, 0)
        nn.init.constant_(self.final_layer[3].bias, 0)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, condition: Optional[Dict[str, torch.Tensor]] = None
    ):
        vec_condition = condition.get("vec_condition", None)
        lang_condition = condition.get("lang_condition", None)
        lang_condition_mask = condition.get("lang_condition_mask", None)
        vis_condition = condition.get("vis_condition", None)
        vis_condition_mask = condition.get("vis_condition_mask", None)

        x = self.x_proj(x) + self.pos_emb
        emb = self.t_proj(self.map_noise(t))

        if condition is not None and vec_condition is not None:
            emb = emb + vec_condition

        for i, block in enumerate(self.blocks):
            seq_condition = lang_condition if i % 2 == 0 else vis_condition
            seq_condition_mask = lang_condition_mask if i % 2 == 0 else vis_condition_mask
            x = block(x, emb, seq_condition, seq_condition_mask)

        x = self.final_layer(x)

        return x


class DiTBlockWithCrossAttentionPostNorm(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(self, hidden_size: int, n_heads: int, dropout: float = 0.0):
        super().__init__()

        # Self Attention Layer
        self.sa_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.sa_attn = nn.MultiheadAttention(hidden_size, n_heads, dropout, batch_first=True)

        # Cross Attention Layer
        self.ca_norm = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.ca_attn = nn.MultiheadAttention(hidden_size, n_heads, dropout, batch_first=True)

        # Feed Forward Layer
        self.ffn_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

        # adaLN
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6))

    def forward(
        self,
        x: torch.Tensor,
        vec_condition: torch.Tensor,
        seq_condition: torch.Tensor,
        seq_condition_mask: Optional[torch.Tensor] = None,
    ):
        (
            shift_sa,
            scale_sa,
            gate_sa,
            # shift_ca,
            # scale_ca,
            # gate_ca,
            shift_ffn,
            scale_ffn,
            gate_ffn,
        ) = self.adaLN_modulation(vec_condition.unsqueeze(-2)).chunk(6, dim=-1)  # (b, 1, 1)

        x = x + gate_sa * self.sa_attn(x, x, x)[0]
        x = self.sa_norm(x) * (1 + scale_sa) + shift_sa

        x = (
            x
            + self.ca_attn(x, seq_condition, seq_condition, key_padding_mask=seq_condition_mask)[0]
        )
        x = self.ca_norm(x)

        x = x + gate_ffn * self.mlp(x)
        x = self.ffn_norm(x) * (1 + scale_ffn) + shift_ffn

        # h = self.sa_norm(x) * (1 + scale_sa) + shift_sa
        # x = x + gate_sa * self.sa_attn(h, h, h)[0]

        # h = self.ca_norm(x)
        # x = (
        #     x
        #     + self.ca_attn(h, seq_condition, seq_condition, key_padding_mask=seq_condition_mask)[0]
        # )

        # h = self.ffn_norm(x) * (1 + scale_ffn) + shift_ffn
        # x = x + gate_ffn * self.mlp(h)

        return x
