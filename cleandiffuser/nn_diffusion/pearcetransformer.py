from typing import Optional

import torch
import torch.nn as nn

from cleandiffuser.nn_diffusion import BaseNNDiffusion


class TimeSiren(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(TimeSiren, self).__init__()
        # just a fully connected NN with sin activations
        self.lin1 = nn.Linear(input_dim, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        x = torch.sin(self.lin1(x))
        x = self.lin2(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, trans_emb_dim, transformer_dim, nheads):
        super(TransformerEncoderBlock, self).__init__()
        # mainly going off of https://jalammar.github.io/illustrated-transformer/

        self.trans_emb_dim = trans_emb_dim
        self.transformer_dim = transformer_dim
        self.nheads = nheads

        self.input_to_qkv1 = nn.Linear(self.trans_emb_dim, self.transformer_dim * 3)
        self.multihead_attn1 = nn.MultiheadAttention(self.transformer_dim, num_heads=self.nheads)
        self.attn1_to_fcn = nn.Linear(self.transformer_dim, self.trans_emb_dim)
        self.attn1_fcn = nn.Sequential(
            nn.Linear(self.trans_emb_dim, self.trans_emb_dim * 4),
            nn.GELU(),
            nn.Linear(self.trans_emb_dim * 4, self.trans_emb_dim),
        )
        self.norm1a = nn.BatchNorm1d(self.trans_emb_dim)
        self.norm1b = nn.BatchNorm1d(self.trans_emb_dim)

    def split_qkv(self, qkv):
        assert qkv.shape[-1] == self.transformer_dim * 3
        q = qkv[:, :, : self.transformer_dim]
        k = qkv[:, :, self.transformer_dim : 2 * self.transformer_dim]
        v = qkv[:, :, 2 * self.transformer_dim :]
        return q, k, v

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        qkvs1 = self.input_to_qkv1(inputs)
        # shape out = [3, batchsize, transformer_dim*3]

        qs1, ks1, vs1 = self.split_qkv(qkvs1)
        # shape out = [3, batchsize, transformer_dim]

        attn1_a = self.multihead_attn1(qs1, ks1, vs1, need_weights=False)
        attn1_a = attn1_a[0]
        # shape out = [3, batchsize, transformer_dim = trans_emb_dim x nheads]

        attn1_b = self.attn1_to_fcn(attn1_a)
        attn1_b = attn1_b / 1.414 + inputs / 1.414  # add residual
        # shape out = [3, batchsize, trans_emb_dim]

        # normalise
        attn1_b = self.norm1a(attn1_b.transpose(0, 2).transpose(0, 1))
        attn1_b = attn1_b.transpose(0, 1).transpose(0, 2)
        # batchnorm likes shape = [batchsize, trans_emb_dim, 3]
        # so have to shape like this, then return

        # fully connected layer
        attn1_c = self.attn1_fcn(attn1_b) / 1.414 + attn1_b / 1.414
        # shape out = [3, batchsize, trans_emb_dim]

        # normalise
        # attn1_c = self.norm1b(attn1_c)
        attn1_c = self.norm1b(attn1_c.transpose(0, 2).transpose(0, 1))
        attn1_c = attn1_c.transpose(0, 1).transpose(0, 2)
        return attn1_c


class EmbeddingBlock(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(in_dim, emb_dim), nn.LeakyReLU(), nn.Linear(emb_dim, emb_dim))

    def forward(self, x):
        return self.model(x)


class PearceTransformer(BaseNNDiffusion):
    """Pearce Transformer diffusion model backbone.

    A simple transformer encoder used in diffusion behavior clone (DBC).

    Args:
        x_dim (int):
            The dimension of the input. It is referred to as the dimension of `action` in DBC.
        emb_dim (int):
            The dimension of the timestep embedding and condition embedding.
        condition_horizon (int):
            The horizon of the condition embedding.
            The condition should be of shape (b, condition_horizon, emb_dim) and is referred to as `observation` in DBC.
        d_model (int):
            The dimension of the transformer.
        n_heads (int):
            The number of heads in the transformer.
        timestep_emb_type (str):
            The type of the timestep embedding.
        timestep_emb_params (Optional[dict]):
            The parameters of the timestep embedding. Default: None

    Examples:
        >>> model = PearceTransformer(x_dim=10, emb_dim=16, condition_horizon=2)
        >>> x = torch.randn((2, 10))
        >>> t = torch.randint(1000, (2,))
        >>> condition = torch.randn((2, 2, 16))
        >>> model(x, t, condition).shape
        torch.Size([2, 10])
        >>> model(x, t, None).shape
        torch.Size([2, 10])
    """

    def __init__(
        self,
        x_dim: int,
        emb_dim: int = 128,
        condition_horizon: int = 1,
        d_model: int = 256,
        nhead: int = 8,
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None,
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)
        self.emb_dim = emb_dim
        self.condition_horizon = condition_horizon
        self.act_emb = nn.Sequential(nn.Linear(x_dim, emb_dim), nn.LeakyReLU(), nn.Linear(emb_dim, emb_dim))

        self.act_to_input = nn.Linear(emb_dim, d_model)
        self.t_to_input = nn.Linear(emb_dim, d_model)
        self.cond_to_input = nn.Linear(emb_dim, d_model)

        self.pos_embed = nn.Parameter(torch.randn((1, 2 + condition_horizon, d_model)), requires_grad=True)

        self.transformer_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4 * d_model,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=4,
        )

        self.final = nn.Linear(d_model * (2 + condition_horizon), x_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: Optional[torch.Tensor] = None):
        x_e, t_e = self.act_emb(x), self.map_noise(t)
        x_input, t_input = self.act_to_input(x_e), self.t_to_input(t_e)
        if condition is None:
            condition = torch.zeros((x.shape[0], self.condition_horizon, self.emb_dim), device=x.device)
        c_input = self.cond_to_input(condition)
        tfm_in = torch.cat([x_input.unsqueeze(1), t_input.unsqueeze(1), c_input], dim=1)
        tfm_in += self.pos_embed
        f = self.transformer_blocks(tfm_in).flatten(1)
        out = self.final(f)
        return out
