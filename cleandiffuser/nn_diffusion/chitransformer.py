from typing import Optional

import torch
import torch.nn as nn

from cleandiffuser.utils import (FourierEmbedding, PositionalEmbedding,
                                 SinusoidalEmbedding)

from .base_nn_diffusion import BaseNNDiffusion


def init_weight(module):
    ignore_types = (
        nn.Dropout,
        SinusoidalEmbedding,
        FourierEmbedding,
        PositionalEmbedding,
        nn.TransformerEncoderLayer,
        nn.TransformerDecoderLayer,
        nn.TransformerEncoder,
        nn.TransformerDecoder,
        nn.ModuleList,
        nn.Mish,
        nn.Sequential)

    if isinstance(module, (nn.Linear, nn.Embedding)):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    elif isinstance(module, nn.MultiheadAttention):
        weight_names = [
            'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
        for name in weight_names:
            weight = getattr(module, name)
            if weight is not None:
                torch.nn.init.normal_(weight, mean=0.0, std=0.02)

        bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
        for name in bias_names:
            bias = getattr(module, name)
            if bias is not None:
                torch.nn.init.zeros_(bias)

    elif isinstance(module, nn.LayerNorm):
        torch.nn.init.zeros_(module.bias)
        torch.nn.init.ones_(module.weight)

    elif isinstance(module, ChiTransformer):
        torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
        if module.obs_emb is not None:
            torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)

    elif isinstance(module, ignore_types):
        # no param
        pass
    else:
        raise RuntimeError("Unaccounted module {}".format(module))


class ChiTransformer(BaseNNDiffusion):
    """ condition: (1 + To) | x: (Ta) """

    def __init__(
            self,
            act_dim: int, obs_dim: int, Ta: int, To: int,
            d_model: int = 256, nhead: int = 4, num_layers: int = 8,
            p_drop_emb: float = 0.0, p_drop_attn: float = 0.3,
            n_cond_layers: int = 0,
            timestep_emb_type: str = "positional",
            timestep_emb_params: Optional[dict] = None
    ):
        super().__init__(d_model, timestep_emb_type, timestep_emb_params)

        T = Ta
        T_cond = 1 + To
        self.To = To
        self.obs_dim = obs_dim

        self.act_emb = nn.Linear(act_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, Ta, d_model))

        self.obs_emb = nn.Linear(obs_dim, d_model)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, 1 + To, d_model))

        self.drop = nn.Dropout(p_drop_emb)
        self.cond_encoder = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.Mish(), nn.Linear(4 * d_model, d_model))

        # encoder
        if n_cond_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model, nhead, 4 * d_model, p_drop_attn, activation='gelu', batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=n_cond_layers)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(d_model, 4 * d_model), nn.Mish(), nn.Linear(4 * d_model, d_model))

        # decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, 4 * d_model, p_drop_attn, activation='gelu', batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=num_layers)

        # attention mask
        mask = (torch.triu(torch.ones(Ta, Ta)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        self.mask = nn.Parameter(mask, requires_grad=False)

        t, s = torch.meshgrid(
            torch.arange(Ta), torch.arange(To + 1), indexing='ij')
        mask = (t >= (s - 1))
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        self.memory_mask = nn.Parameter(mask, requires_grad=False)

        # decoder head
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, act_dim)

        # constant
        self.T = T
        self.T_cond = T_cond

        self.apply(init_weight)

    def forward(self,
                x: torch.Tensor, noise: torch.Tensor,
                condition: Optional[torch.Tensor] = None):
        """
        Input:
            x:          (b, Ta, act_dim)
            noise:      (b, )
            condition:  (b, To, obs_dim)

        Output:
            y:          (b, Ta, act_dim)
        """
        if condition is None:
            condition = torch.zeros((x.shape[0], self.To, self.obs_dim)).to(x.device)  # (b, To, obs_dim)
        t_emb = self.map_noise(noise).unsqueeze(1)  # (b, 1, d_model)

        act_emb = self.act_emb(x)
        obs_emb = self.obs_emb(condition)

        cond_emb = torch.cat([t_emb, obs_emb], dim=1)  # (b, 1+To, d_model)
        cond_pos_emb = self.cond_pos_emb[:, :cond_emb.shape[1], :]
        memory = self.drop(cond_emb + cond_pos_emb)
        memory = self.encoder(memory)  # (b, 1+To, d_model)

        act_pos_emb = self.pos_emb[:, :act_emb.shape[1], :]
        x = self.drop(act_emb + act_pos_emb)  # (b, Ta, d_model)
        x = self.decoder(tgt=x, memory=memory, tgt_mask=self.mask, memory_mask=self.memory_mask)

        x = self.ln_f(x)
        x = self.head(x)

        return x
