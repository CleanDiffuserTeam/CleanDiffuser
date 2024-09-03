import torch
import torch.nn as nn

from cleandiffuser.nn_diffusion import BaseNNDiffusion
from typing import Optional


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
        q = qkv[:, :, :self.transformer_dim]
        k = qkv[:, :, self.transformer_dim: 2 * self.transformer_dim]
        v = qkv[:, :, 2 * self.transformer_dim:]
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
        self.model = nn.Sequential(
            nn.Linear(in_dim, emb_dim), nn.LeakyReLU(),
            nn.Linear(emb_dim, emb_dim))

    def forward(self, x):
        return self.model(x)


class PearceTransformer(BaseNNDiffusion):
    def __init__(
            self, act_dim: int, To: int = 1,
            emb_dim: int = 128, trans_emb_dim: int = 64, nhead: int = 16,
            timestep_emb_type: str = "positional",
            timestep_emb_params: Optional[dict] = None
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)

        self.To = To
        self.emb_dim = emb_dim
        self.act_emb = nn.Sequential(
            nn.Linear(act_dim, emb_dim), nn.LeakyReLU(), nn.Linear(emb_dim, emb_dim))

        transformer_dim = trans_emb_dim * nhead

        self.act_to_input = nn.Linear(emb_dim, trans_emb_dim)
        self.t_to_input = nn.Linear(emb_dim, trans_emb_dim)
        self.cond_to_input = nn.Linear(emb_dim, trans_emb_dim)

        self.pos_embed = TimeSiren(1, trans_emb_dim)

        self.transformer_blocks = nn.Sequential(
            TransformerEncoderBlock(trans_emb_dim, transformer_dim, nhead),
            TransformerEncoderBlock(trans_emb_dim, transformer_dim, nhead),
            TransformerEncoderBlock(trans_emb_dim, transformer_dim, nhead),
            TransformerEncoderBlock(trans_emb_dim, transformer_dim, nhead))

        self.final = nn.Linear(trans_emb_dim * (2 + To), act_dim)

    def forward(self,
                x: torch.Tensor, noise: torch.Tensor,
                condition: Optional[torch.Tensor] = None):
        """
        Input:
            x:          (b, act_dim)
            noise:      (b, )
            condition:  (b, To, emb_dim)

        Output:
            y:          (b, act_dim)
        """
        if condition is None:
            condition = torch.zero((x.shape[0], self.To, self.emb_dim)).to(x.device)

        x_e, t_e = self.act_emb(x), self.map_noise(noise)

        x_input, t_input, c_input = self.act_to_input(x_e), self.t_to_input(t_e), self.cond_to_input(condition)

        x_input += self.pos_embed(torch.zeros(1, 1, device=x.device) + 1.0)
        t_input += self.pos_embed(torch.zeros(1, 1, device=x.device) + 2.0)
        c_input += self.pos_embed(
            torch.arange(3, 3 + condition.shape[1], device=x.device, dtype=torch.float32)[None, :, None])

        f = torch.cat([x_input.unsqueeze(1), t_input.unsqueeze(1), c_input], dim=1)
        f = self.transformer_blocks(f.permute(1, 0, 2)).permute(1, 0, 2)

        flat = torch.flatten(f, start_dim=1)

        out = self.final(flat)
        return out
