from typing import List
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from cleandiffuser.utils import at_least_ndim, GroupNorm1d
from cleandiffuser.nn_diffusion import BaseNNDiffusion


def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform':
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform':
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


# ----------------------------------------------------------------------------
# Fully-connected layer.
# Adapted from https://github.com/NVlabs/edm/blob/main/training/networks.py
class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


# ----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.
# Adapted from the 2d version on https://github.com/NVlabs/edm/blob/main/training/networks.py
class Conv1d(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 kernel: int = 3,
                 bias: bool = True,
                 up: bool = False,
                 down: bool = False,
                 resample_filter: List[int] = [1, 1],
                 fused_resample: bool = False,
                 init_mode: str = 'kaiming_normal',
                 init_weight: float = 1.,
                 init_bias: float = 0.
                 ):
        assert not (up and down)
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.up, self.down = up, down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_dim * kernel, fan_out=out_dim * kernel)
        self.weight = torch.nn.Parameter(
            weight_init([out_dim, in_dim, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(
            weight_init([out_dim], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer("resample_filter", f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose1d(x, f.mul(2).tile([self.in_dim, 1, 1]),
                                                     groups=self.in_dim, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv1d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv1d(x, w, padding=w_pad + f_pad)
            x = torch.nn.functional.conv1d(x, f.tile([self.out_dim, 1, 1]), groups=self.out_dim, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose1d(x, f.mul(2).tile([self.in_dim, 1, 1]),
                                                         groups=self.in_dim, stride=2, padding=f_pad)
            if self.down:
                x = torch.nn.functional.conv1d(x, f.tile([self.in_dim, 1, 1]), groups=self.in_dim,
                                               stride=2, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv1d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1))
        return x


# ------------------------------------------------------------------------------
# UNet block with CNN residual and self-attention.
# (b, in_dim, horizon) -> (b, out_dim, horizon)
class UNetBlock1d(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 emb_dim: int,
                 up: bool = False,
                 down: bool = False,
                 resample_filter: List[int] = [1, 1],
                 kernel_size: int = 3,
                 dropout: float = 0.,
                 skip_scale: float = 1.,
                 adaptive_scale: bool = True,
                 activation: nn.Module = nn.SiLU(),
                 attention: bool = False,
                 num_heads: int = 4,
                 norm: str = "group_norm",
                 init: dict = {},
                 ):
        super().__init__()
        self.in_dim, self.out_dim, self.emb_dim = in_dim, out_dim, emb_dim
        self.dropout, self.skip_scale = dropout, skip_scale
        self.adaptive_scale = adaptive_scale
        self.activation = activation

        if norm == "group_norm":
            self.norm0, self.norm1 = GroupNorm1d(in_dim), GroupNorm1d(out_dim)
        else:
            self.norm0, self.norm1 = nn.LayerNorm(in_dim), nn.LayerNorm(out_dim)

        self.conv0, self.conv1 = (
            Conv1d(in_dim, out_dim, 3, True, up, down, resample_filter, **init),
            Conv1d(out_dim, out_dim, 3, init_weight=0.))

        self.affine = Linear(emb_dim, 2 * out_dim if adaptive_scale else out_dim, **init)
        self.dropout = nn.Dropout(dropout)
        self.skip = Conv1d(in_dim, out_dim, 1, True, up, down, resample_filter, **init)
        self.attn = nn.MultiheadAttention(out_dim, num_heads, dropout=dropout, batch_first=True) if attention else None

    def forward(self, x, emb):
        orig = x
        x = self.conv0(self.activation(self.norm0(x)))
        params = at_least_ndim(self.affine(emb), x.dim())
        if self.adaptive_scale:
            scale, shift = params.chunk(2, dim=1)
            x = self.activation(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = self.activation(self.norm1(x.add_(params)))

        x = self.conv1(self.dropout(x))
        x = x.add_(self.skip(orig) if self.skip is not None else orig) * self.skip_scale

        if self.attn is not None:
            x_ = x.permute(0, 2, 1)
            a, _ = self.attn(x_, x_, x_)
            x = (a.permute(0, 2, 1) + x) * self.skip_scale

        return x


# ----------------------------------------------------------------------------
# UNet with residual connections and self-attention.
# Adapted from the 2d version on https://github.com/NVlabs/edm/blob/main/training/networks.py
class UNet1d(BaseNNDiffusion):
    def __init__(self,
                 horizon: int,
                 in_dim: int,
                 model_dim: int,
                 emb_dim: int,
                 dim_mult: List[int] = [1, 2, 2, 2],
                 num_blocks: int = 2,
                 attn_horizon: List[int] = [16],
                 dropout: float = 0.1,
                 timestep_emb_type: str = "positional",
                 norm_type: str = "group_norm",
                 activation: nn.Module = nn.SiLU(),
                 encoder_type: str = "standard",  # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
                 decoder_type: str = "standard",  # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
                 resample_filter: List[int] = [1, 1],  # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
                 ):
        assert norm_type in ['group_norm', 'layer_norm']
        super().__init__(emb_dim, timestep_emb_type)

        init = dict(init_mode="xavier_uniform")
        self.activation = activation

        block_kwargs = dict(
            emb_dim=emb_dim, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5),
            resample_filter=resample_filter, adaptive_scale=False, init=init, kernel_size=3,
            activation=activation, norm=norm_type)

        # Mapping.
        self.map_emb = nn.Sequential(
            Linear(emb_dim, emb_dim, **init), activation, Linear(emb_dim, emb_dim, **init), activation)

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        dout, daux = in_dim, in_dim
        for level, mult in enumerate(dim_mult):
            res = horizon >> level
            if level == 0:
                din = dout
                dout = model_dim
                self.enc[f'{res}x{res}_conv'] = Conv1d(din, dout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock1d(dout, dout, down=True, **block_kwargs)
                if encoder_type == 'skip':
                    self.enc[f'{res}x{res}_aux_down'] = Conv1d(daux, daux, kernel=0, down=True,
                                                               resample_filter=resample_filter)
                    self.enc[f'{res}x{res}_aux_skip'] = Conv1d(daux, dout, kernel=1, **init)
                if encoder_type == 'residual':
                    self.enc[f'{res}x{res}_aux_residual'] = Conv1d(daux, dout, kernel=3,
                                                                   down=True, resample_filter=resample_filter,
                                                                   fused_resample=True, **init)
                    daux = dout
            for idx in range(num_blocks):
                din = dout
                dout = model_dim * mult
                attn = (res in attn_horizon)
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock1d(din, dout, attention=attn, **block_kwargs)
        skips = [block.out_dim for name, block in self.enc.items() if 'aux' not in name]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(dim_mult))):
            res = horizon >> level
            if level == len(dim_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock1d(dout, dout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock1d(dout, dout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock1d(dout, dout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                din = dout + skips.pop()
                dout = model_dim * mult
                attn = (idx == num_blocks and res in attn_horizon)
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock1d(din, dout, attention=attn, **block_kwargs)
            if decoder_type == 'skip' or level == 0:
                if decoder_type == 'skip' and level < len(dim_mult) - 1:
                    self.dec[f'{res}x{res}_aux_up'] = Conv1d(in_dim, in_dim,
                                                             kernel=0, up=True, resample_filter=resample_filter)
                self.dec[f'{res}x{res}_aux_norm'] = GroupNorm1d(dout, eps=1e-6)
                self.dec[f'{res}x{res}_aux_conv'] = Conv1d(dout, in_dim, kernel=3, init_weight=0.)

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
        x = x.permute(0, 2, 1)
        emb = self.map_noise(noise)
        if condition is not None:
            emb = emb + condition
        emb = self.map_emb(emb)

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if 'aux_down' in name:
                aux = block(aux)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux)
            elif 'aux_residual' in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock1d) else block(x)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_up' in name:
                aux = block(aux)
            elif 'aux_norm' in name:
                tmp = block(x)
            elif 'aux_conv' in name:
                tmp = block(self.activation(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_dim:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        return aux.permute(0, 2, 1)
