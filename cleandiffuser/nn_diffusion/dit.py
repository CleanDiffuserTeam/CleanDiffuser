from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import UntrainablePositionalEmbedding

__all__ = ["DiT1d", "DiT1dWithACICrossAttention"]


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        use_cross_attn: bool = False,
        adaLN_on_cross_attn: bool = False,
    ):
        super().__init__()
        self._adaLN_on_cross_attn = adaLN_on_cross_attn
        self._use_cross_attn = use_cross_attn

        # self attention
        self.sa_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.sa_attn = nn.MultiheadAttention(hidden_size, n_heads, attn_dropout, batch_first=True)

        # cross attention
        if use_cross_attn:
            self.ca_norm = nn.LayerNorm(
                hidden_size, elementwise_affine=not adaLN_on_cross_attn, eps=1e-6
            )
            self.ca_attn = nn.MultiheadAttention(
                hidden_size, n_heads, attn_dropout, batch_first=True
            )
        else:
            self.ca_norm, self.ca_attn = None, None

        # feed forward
        self.ffn_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.Dropout(ffn_dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(ffn_dropout),
        )

        # adaLN
        n_coeff = 9 if adaLN_on_cross_attn else 6
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, hidden_size * n_coeff)
        )

    def forward(
        self,
        x: torch.Tensor,
        vec_condition: torch.Tensor,
        seq_condition: Optional[torch.Tensor] = None,
        seq_condition_mask: Optional[torch.Tensor] = None,
    ):
        adaLN_coeff = self.adaLN_modulation(vec_condition.unsqueeze(-2))
        if self._adaLN_on_cross_attn:
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
            ) = adaLN_coeff.chunk(9, dim=-1)
        else:
            shift_sa, scale_sa, gate_sa, shift_ffn, scale_ffn, gate_ffn = adaLN_coeff.chunk(
                6, dim=-1
            )

        h = self.sa_norm(x) * (1 + scale_sa) + shift_sa
        x = x + gate_sa * self.sa_attn(h, h, h)[0]

        if self._use_cross_attn:
            if self._adaLN_on_cross_attn:
                h = self.ca_norm(x) * (1 + scale_ca) + shift_ca
            else:
                h = self.ca_norm(x)
                gate_ca = 1.0

            x = (
                x
                + gate_ca
                * self.ca_attn(
                    h, seq_condition, seq_condition, key_padding_mask=seq_condition_mask
                )[0]
            )

        h = self.ffn_norm(x) * (1 + scale_ffn) + shift_ffn
        x = x + gate_ffn * self.mlp(h)
        return x


class FinalLayer1d(nn.Module):
    def __init__(self, hidden_size: int, out_dim: int, head_type: str = "linear"):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(approximate="tanh"),
                nn.Linear(hidden_size, out_dim),
            )
            nn.init.constant_(self.head[-1].weight, 0)
            nn.init.constant_(self.head[-1].bias, 0)
        else:
            self.head = nn.Linear(hidden_size, out_dim)
            nn.init.constant_(self.head.weight, 0)
            nn.init.constant_(self.head.bias, 0)

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        return self.head(x)


class DiT1d(BaseNNDiffusion):
    """Temporal Diffusion Transformer (DiT) backbone used in AlignDiff.

    Args:
        x_dim (int):
            The dimension of the input. Input tensors are assumed to be in shape of (b, x_seq_len, x_dim),
        x_seq_len (int):
            The sequence length of the input.
        emb_dim (int):
            The dimension of the timestep embedding and condition embedding.
        d_model (int):
            The dimension of the model.
        n_heads (int):
            The number of heads in the model.
        depth (int):
            The number of layers in the model.
        attn_dropout (float):
            The dropout rate for the attention layers.
        ffn_dropout (float):
            The dropout rate for the feed forward layers.
        head_type (str):
            The type of the output head. Can be "linear" or "mlp".
        use_trainable_pos_emb (bool):
            Whether to use trainable positional embeddings.
        use_cross_attn (bool):
            Whether to add an extra cross attention layer for sequence conditioning.
        adaLN_on_cross_attn (bool):
            Whether to use adaptive layer normalization on the cross attention layer.
        timestep_emb_type (str):
            The type of the timestep embedding.
        timestep_emb_params (Optional[dict]):
            The parameters of the timestep embedding.

    Examples:
    >>> model = DiT1d(x_dim=5, x_seq_len=10, emb_dim=128)
    >>> xt = torch.randn((2, 10, 5))
    >>> t = torch.randint(1000, (2,))
    >>> condition = {"vec_condition": torch.randn((2, 128))}
    >>> model(xt, t, condition).shape
    torch.Size([2, 10, 5])

    >>> model = DiT1d(x_dim=5, x_seq_len=10, emb_dim=128, use_cross_attn=True)
    >>> condition = {"vec_condition": torch.randn((2, 128)), "seq_condition": torch.randn((2, 8, 128))}
    >>> model(xt, t, condition).shape
    torch.Size([2, 10, 5])
    """

    def __init__(
        self,
        x_dim: int,
        x_seq_len: int,
        emb_dim: int,
        d_model: int = 384,
        n_heads: int = 6,
        depth: int = 12,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        head_type: str = "linear",
        use_trainable_pos_emb: bool = True,
        use_cross_attn: bool = False,
        adaLN_on_cross_attn: bool = False,
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None,
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)
        # input projection
        self.x_proj = nn.Linear(x_dim, d_model)
        self.t_proj = nn.Sequential(
            nn.Linear(emb_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.cond_proj = nn.Sequential(nn.Linear(emb_dim, d_model), nn.LayerNorm(d_model))
        if use_cross_attn:
            self.seq_cond_proj = nn.Sequential(nn.Linear(emb_dim, d_model), nn.LayerNorm(d_model))
        else:
            self.seq_cond_proj = None

        # positional embedding
        pos_emb = UntrainablePositionalEmbedding(d_model)(torch.arange(x_seq_len))[None]
        self.pos_emb = nn.Parameter(pos_emb, requires_grad=use_trainable_pos_emb)

        # main transformer blocks
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    d_model, n_heads, attn_dropout, ffn_dropout, use_cross_attn, adaLN_on_cross_attn
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer1d(d_model, x_dim, head_type)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize time step embedding MLP:
        nn.init.normal_(self.t_proj[0].weight, std=0.02)
        nn.init.normal_(self.t_proj[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
    ):
        if isinstance(condition, dict):
            vec_condition = condition["vec_condition"]
            seq_condition = condition.get("seq_condition", None)
            seq_condition_mask = condition.get("seq_condition_mask", None)
        else:
            vec_condition = condition
            seq_condition = None
            seq_condition_mask = None

        t_emb = self.t_proj(self.map_noise(t))
        x_emb = self.x_proj(x) + self.pos_emb

        cond_emb = t_emb
        if vec_condition is not None:
            cond_emb = cond_emb + self.cond_proj(vec_condition)

        if seq_condition is not None and self.seq_cond_proj is not None:
            seq_condition = self.seq_cond_proj(seq_condition)

        for block in self.blocks:
            x_emb = block(x_emb, cond_emb, seq_condition, seq_condition_mask)

        x_emb = self.final_layer(x_emb, cond_emb)

        return x_emb


class DiT1dWithACICrossAttention(DiT1d):
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

    Args:
        x_dim (int):
            Input dimension. The input should be in shape of (b, x_seq_len, x_dim).
        x_seq_len (int):
            The length of the input sequence.
        emb_dim (int):
            The dimension of the embedding.
        d_model (int):
            The dimension of the model.
        n_heads (int):
            The number of heads in the attention layers.
        depth (int):
            The number of transformer blocks.
        attn_dropout (float):
            The dropout rate of the attention layers.
        ffn_dropout (float):
            The dropout rate of the feed-forward layers.
        head_type (str):
            The type of the head. Can be "linear" or "mlp".
        use_trainable_pos_emb (bool):
            Whether to use trainable positional embedding.
        adaLN_on_cross_attn (bool):
            Whether to use adaLN in the cross-attention layers.
        timestep_emb_type (str):
            The type of the timestep embedding.
        timestep_emb_params (Optional[dict]):
            The parameters of the timestep embedding.

    Examples:
    >>> model = DiT1dWithACICrossAttention(x_dim=5, x_seq_len=10, emb_dim=128)
    >>> xt = torch.randn(2, 10, 5)
    >>> t = torch.randint(0, 1000, (2,))
    >>> condition = {"vec_condition": torch.randn(2, 128), "vis_condition": torch.randn(2, 196, 128), "lang_condition": torch.randn(2, 32, 128)}
    >>> model(xt, t, condition).shape
    torch.Size([2, 10, 5])
    """

    def __init__(
        self,
        x_dim: int,
        x_seq_len: int,
        emb_dim: int,
        d_model: int = 384,
        n_heads: int = 6,
        depth: int = 12,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        head_type: str = "mlp",
        use_trainable_pos_emb: bool = True,
        adaLN_on_cross_attn: bool = False,
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None,
    ):
        super().__init__(
            x_dim,
            x_seq_len,
            emb_dim,
            d_model,
            n_heads,
            depth,
            attn_dropout,
            ffn_dropout,
            head_type,
            use_trainable_pos_emb,
            True,
            adaLN_on_cross_attn,
            timestep_emb_type,
            timestep_emb_params,
        )
        self.seq_cond_proj = None
        self.vis_cond_proj = nn.Sequential(nn.Linear(emb_dim, d_model), nn.LayerNorm(d_model))
        self.lang_cond_proj = nn.Sequential(nn.Linear(emb_dim, d_model), nn.LayerNorm(d_model))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Dict[str, torch.Tensor] = None,
    ):
        vec_condition = condition.get("vec_condition", 0)
        vis_condition = condition.get("vis_condition", None)
        vis_condition_mask = condition.get("vis_condition_mask", None)
        lang_condition = condition.get("lang_condition", None)
        lang_condition_mask = condition.get("lang_condition_mask", None)

        t_emb = self.t_proj(self.map_noise(t))
        x_emb = self.x_proj(x) + self.pos_emb

        cond_emb = t_emb
        if vec_condition is not None:
            cond_emb = cond_emb + self.cond_proj(vec_condition)

        if vis_condition is not None and self.vis_cond_proj is not None:
            vis_condition = self.vis_cond_proj(vis_condition)
        if lang_condition is not None and self.lang_cond_proj is not None:
            lang_condition = self.lang_cond_proj(lang_condition)

        for i, block in enumerate(self.blocks):
            seq_condition = vis_condition if i % 2 == 0 else lang_condition
            seq_condition_mask = vis_condition_mask if i % 2 == 0 else lang_condition_mask
            x_emb = block(x_emb, cond_emb, seq_condition, seq_condition_mask)

        x_emb = self.final_layer(x_emb, cond_emb)

        return x_emb


if __name__ == "__main__":
    x_dim = 10
    x_seq_len = 32
    emb_dim = 16

    x = torch.randn((2, x_seq_len, x_dim))
    t = torch.randint(0, 1000, (2,))
    condition = torch.randn((2, emb_dim))

    model = DiT1d(x_dim, x_seq_len, emb_dim)

    print(model(x, t, condition).shape)

    seq_cond = torch.randn((2, 8, emb_dim))
    model = DiT1d(x_dim, x_seq_len, emb_dim, use_cross_attn=True)
    print(model(x, t, {"vec_condition": condition, "seq_condition": seq_cond}).shape)

    vis_cond = torch.randn((2, 8, emb_dim))
    lang_cond = torch.randn((2, 8, emb_dim))
    lang_cond_mask = torch.ones((2, 8), dtype=torch.bool)
    model = DiT1dWithACICrossAttention(x_dim, x_seq_len, emb_dim)
    print(
        model(
            x,
            t,
            {
                "vec_condition": condition,
                "vis_condition": vis_cond,
                "lang_condition": lang_cond,
                "lang_condition_mask": lang_cond_mask,
            },
        ).shape
    )
