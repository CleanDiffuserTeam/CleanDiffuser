from typing import Optional

import torch
import torch.nn as nn

from cleandiffuser.nn_diffusion.base_nn_diffusion import BaseNNDiffusion

__all__ = ["ChiTransformer"]


class ChiTransformer(BaseNNDiffusion):
    """Transformer diffusion backbone used in Diffusion Policy (DP).

    A Transformer with specially designed attention masking.

    Args:
        x_dim (int):
            The dimension of the input. It is referred to as the dimension of `action` in DP.
            The input should be in shape of (b, x_horizon, x_dim).
        x_horizon (int):
            The horizon of the input.
        condition_dim (int):
            The dimension of the condition embedding. It is referred to as `observation` in DP.
        condition_horizon (int):
            The horizon of the condition embedding.
        d_model (int):
            The dimension of the transformer.
        nhead (int):
            The number of attention heads.
        num_layers (int):
            The number of transformer layers.
        p_drop_emb (float):
            The dropout rate of the embedding layer.
        p_drop_attn (float):
            The dropout rate of the attention layer.
        n_cond_layers (int):
            The number of layers that use the condition embedding.
        timestep_emb_type (str):
            The type of the timestep embedding.
        timestep_emb_params (Optional[dict]):
            The parameters of the timestep embedding.

    Example:
        >>> model = ChiTransformer(x_dim=10, x_horizon=16, condition_dim=5, condition_horizon=2)
        >>> x = torch.randn(4, 16, 10)
        >>> t = torch.randint(1000, (4,))
        >>> condition = torch.randn(4, 2, 5)
        >>> model(x, t, condition).shape
        torch.Size([4, 16, 10])
        >>> model(x, t, None).shape
        torch.Size([4, 16, 10])
    """

    def __init__(
        self,
        x_dim: int,
        x_horizon: int,
        condition_dim: int,
        condition_horizon: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 8,
        p_drop_emb: float = 0.0,
        p_drop_attn: float = 0.3,
        n_cond_layers: int = 0,
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None,
    ):
        super().__init__(d_model, timestep_emb_type, timestep_emb_params)

        T = x_horizon
        T_cond = 1 + condition_horizon

        self.act_emb = nn.Linear(x_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, x_horizon, d_model))

        self.obs_emb = nn.Linear(condition_dim, d_model)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, 1 + condition_horizon, d_model))

        self.drop = nn.Dropout(p_drop_emb)
        self.cond_encoder = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.Mish(), nn.Linear(4 * d_model, d_model)
        )

        # encoder
        if n_cond_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model,
                nhead,
                4 * d_model,
                p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=n_cond_layers
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(d_model, 4 * d_model), nn.Mish(), nn.Linear(4 * d_model, d_model)
            )

        # decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model,
            nhead,
            4 * d_model,
            p_drop_attn,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_layers)

        # attention mask
        mask = (torch.triu(torch.ones(x_horizon, x_horizon)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        self.mask = nn.Parameter(mask, requires_grad=False)

        t, s = torch.meshgrid(
            torch.arange(x_horizon), torch.arange(condition_horizon + 1), indexing="ij"
        )
        mask = t >= (s - 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        self.memory_mask = nn.Parameter(mask, requires_grad=False)

        # decoder head
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, x_dim)

        # constant
        self.T = T
        self.T_cond = T_cond

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor = None):
        t_emb = self.map_noise(t).unsqueeze(1)  # (b, 1, d_model)

        act_emb = self.act_emb(x)
        obs_emb = self.obs_emb(condition)

        cond_emb = torch.cat([t_emb, obs_emb], dim=1)  # (b, 1+To, d_model)
        cond_pos_emb = self.cond_pos_emb[:, : cond_emb.shape[1], :]
        memory = self.drop(cond_emb + cond_pos_emb)
        memory = self.encoder(memory)  # (b, 1+To, d_model)

        act_pos_emb = self.pos_emb[:, : act_emb.shape[1], :]
        x = self.drop(act_emb + act_pos_emb)  # (b, Ta, d_model)
        x = self.decoder(tgt=x, memory=memory, tgt_mask=self.mask, memory_mask=self.memory_mask)

        x = self.ln_f(x)
        x = self.head(x)

        return x


if __name__ == "__main__":
    m = ChiTransformer(x_dim=10, x_horizon=8, condition_dim=5, condition_horizon=4)
    x = torch.randn((2, 8, 10))
    t = torch.randint(0, 1000, (2,))
    condition = torch.randn((2, 4, 5))
    print(m(x, t, condition).shape)
