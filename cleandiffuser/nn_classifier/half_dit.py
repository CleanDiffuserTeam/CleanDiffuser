from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from cleandiffuser.nn_diffusion.dit import DiT1d, FinalLayer1d


class HalfDiT1d(DiT1d):
    """Half DiT1d for diffusion classifier.

    Args:
        in_dim: int,
            Dimension of the input sequence.
        out_dim: int,
            Dimension of the output tensor.
        emb_dim: int,
            Dimension of the condition and time embedding.
        d_model: int,
            Dimension of the transformer. Default: 384.
        n_heads: int,
            Number of heads in the transformer. Default: 6.
        depth: int,
            Number of transformer layers. Default: 12.
        dropout: float,
            Dropout rate. Default: 0.0.
        timestep_emb_type: str,
            Type of the timestep embedding. Default: "positional".

    Examples:
        >>> nn_classifier = HalfDiT1d(in_dim=10, out_dim=1, emb_dim=64)
        >>> x = torch.randn(2, 32, 10)
        >>> t = torch.randint(1000, (2,))
        >>> condition = torch.randn(2, 64)
        >>> nn_classifier(x, t).shape
        torch.Size([2, 1])
        >>> nn_classifier(x, t, condition).shape
        torch.Size([2, 1])
    """

    def __init__(
        self,
        x_dim: int,
        out_dim: int,
        x_seq_len: int,
        emb_dim: int,
        d_model: int = 384,
        n_heads: int = 6,
        depth: int = 12,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        use_trainable_pos_emb: bool = True,
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
            "linear",
            use_trainable_pos_emb,
            False,
            False,
            timestep_emb_type,
            timestep_emb_params,
        )
        self.final_layer = FinalLayer1d(d_model, d_model // 2)
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, out_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
    ):
        """
        Input:
            x:          (b, horizon, in_dim)
            noise:      (b, )
            condition:  (b, emb_dim)

        Output:
            logp(x | noise, condition): (b, 1)
        """
        feat = super().forward(x, t, condition).mean(1)
        return self.proj(feat)
