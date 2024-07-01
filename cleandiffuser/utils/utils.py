import math
import os
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def at_least_ndim(x, ndim):
    if isinstance(x, np.ndarray):
        return np.reshape(x, x.shape + (1,) * (ndim - x.ndim))
    elif isinstance(x, torch.Tensor):
        return torch.reshape(x, x.shape + (1,) * (ndim - x.ndim))
    elif isinstance(x, (int, float)):
        return x
    else:
        raise ValueError(f"Unsupported type {type(x)}")


def to_tensor(x, device="cpu"):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, (np.ndarray, list, tuple, int, float)):
        return torch.tensor(x, device=device)
    else:
        raise ValueError(f"Unsupported type {type(x)}")


def linear_beta_schedule(beta_min: float = 1e-4, beta_max: float = 0.02, T: int = 1000):
    return np.linspace(beta_min, beta_max, T)


def cosine_beta_schedule(s: float = 0.008, T: int = 1000):
    f = np.cos((np.arange(T + 1) / T + s) / (1 + s) * np.pi / 2.) ** 2
    alpha_bar = f / f[0]
    beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return beta.clip(None, 0.999)


# ================ Discretization ==================
def uniform_discretization(T: int = 1000, eps: float = 1e-3):
    return torch.linspace(eps, 1., T)


SUPPORTED_DISCRETIZATIONS = {
    "uniform": uniform_discretization,
}


# ================= Noise schedules =================
def linear_noise_schedule(t_diffusion: torch.Tensor, beta0: float = 0.1, beta1: float = 20.):
    log_alpha = -(beta1 - beta0) / 4. * (t_diffusion ** 2) - beta0 / 2. * t_diffusion
    alpha = log_alpha.exp()
    sigma = (1. - alpha ** 2).sqrt()
    return alpha, sigma


def inverse_linear_noise_schedule(
        alpha: torch.Tensor = None, sigma: torch.Tensor = None, logSNR: torch.Tensor = None,
        beta0: float = 0.1, beta1: float = 20.):
    assert (logSNR is not None) or (alpha is not None and sigma is not None)
    lmbda = (alpha / sigma).log() if logSNR is None else logSNR
    t_diffusion = (2 * (1 + (-2 * lmbda).exp()).log() /
                   (beta0 + (beta0 ** 2 + 2 * (beta1 - beta0) * (1 + (-2 * lmbda).exp()).log())))
    return t_diffusion


def cosine_noise_schedule(t_diffusion: torch.Tensor, s: float = 0.008):
    t_diffusion[-1] = 0.9946
    alpha = (np.pi / 2. * (t_diffusion + s) / (1 + s)).cos() / np.cos(np.pi / 2. * s / (1 + s))
    sigma = (1. - alpha ** 2).sqrt()
    return alpha, sigma


def inverse_cosine_noise_schedule(
        alpha: torch.Tensor = None, sigma: torch.Tensor = None, logSNR: torch.Tensor = None,
        s: float = 0.008):
    assert (logSNR is not None) or (alpha is not None and sigma is not None)
    lmbda = (alpha / sigma).log() if logSNR is None else logSNR
    t_diffusion = (2 * (1 + s) / np.pi *
                   torch.arccos((-0.5 * (1 + (-2 * lmbda).exp()).log() +
                                 np.log(np.cos(np.pi * s / 2 / (s + 1)))).exp()) - s)
    return t_diffusion


SUPPORTED_NOISE_SCHEDULES = {
    "linear": {
        "forward": linear_noise_schedule,
        "reverse": inverse_linear_noise_schedule},
    "cosine": {
        "forward": cosine_noise_schedule,
        "reverse": inverse_cosine_noise_schedule},
}


# ================= Sampling step schedule ===============
def uniform_sampling_step_schedule(T: int = 1000, sampling_steps: int = 10):
    return torch.linspace(0, T - 1, sampling_steps + 1, dtype=torch.long)


def uniform_sampling_step_schedule_continuous(trange=None, sampling_steps: int = 10):
    if trange is None:
        trange = [1e-3, 1.]
    return torch.linspace(trange[0], trange[1], sampling_steps + 1, dtype=torch.float32)


SUPPORTED_SAMPLING_STEP_SCHEDULE = {
    "uniform": uniform_sampling_step_schedule,
    "uniform_continuous": uniform_sampling_step_schedule_continuous,
}


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ema_update(model: nn.Module, model_ema: nn.Module, ema_rate: float):
    for param, param_ema in zip(model.parameters(), model_ema.parameters()):
        param_ema.data.mul_(ema_rate).add_(param.data, alpha=1 - ema_rate)


"""
Basic NNs
"""


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
            num_groups=self.num_groups, weight=self.weight.to(x.dtype),
            bias=self.bias.to(x.dtype), eps=self.eps)
        return x.squeeze(2)


class Mlp(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dims: List[int],
                 out_dim: int,
                 activation: nn.Module = nn.ReLU(),
                 out_activation: nn.Module = nn.Identity()):

        super().__init__()
        self.mlp = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(in_dim if i == 0 else hidden_dims[i - 1], hidden_dims[i]),
                activation)
            for i in range(len(hidden_dims))
        ], nn.Linear(hidden_dims[-1], out_dim), out_activation)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.mlp(x)


"""
Timestep embeddings. 
These are used to embed the timestep (b, ), either discrete or continuous, into a vector (b, dim).
"""


# -----------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures, from https://github.com/NVlabs/edm/blob/main/training/networks.py#L269
class PositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_positions: int = 10000, endpoint: bool = False):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.dim // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.dim // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class UntrainablePositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_positions: int = 10000, endpoint: bool = False):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.dim // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.dim // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# -----------------------------------------------------------
# Timestep embedding used in Transformer
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# -----------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures
class FourierEmbedding(nn.Module):
    def __init__(self, dim: int, scale=16):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(dim // 8) * scale, requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim // 4, dim), nn.Mish(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor):
        emb = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        emb = torch.cat([emb.cos(), emb.sin()], -1)
        return self.mlp(emb)


class UntrainableFourierEmbedding(nn.Module):
    def __init__(self, dim: int, scale=16):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(dim // 2) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor):
        emb = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        emb = torch.cat([emb.cos(), emb.sin()], -1)
        return emb


SUPPORTED_TIMESTEP_EMBEDDING = {
    "positional": PositionalEmbedding,
    "fourier": FourierEmbedding,
    "untrainable_fourier": UntrainableFourierEmbedding,
    "untrainable_positional": UntrainablePositionalEmbedding,
}


# -----------------------------------------------------------
# Beautiful model size visualization from https://github.com/jannerm/diffuser/tree/main

def _to_str(num):
    if num >= 1e6:
        return f'{(num / 1e6):.2f} M'
    else:
        return f'{(num / 1e3):.2f} k'


def param_to_module(param):
    module_name = param[::-1].split('.', maxsplit=1)[-1][::-1]
    return module_name


def report_parameters(model, topk=10):
    counts = {k: p.numel() for k, p in model.named_parameters() if p.requires_grad}
    n_parameters = sum(counts.values())
    print(f'Total parameters: {_to_str(n_parameters)}')

    modules = dict(model.named_modules())
    sorted_keys = sorted(counts, key=lambda x: -counts[x])
    max_length = max([len(k) for k in sorted_keys])
    for i in range(topk):
        key = sorted_keys[i]
        count = counts[key]
        module = param_to_module(key)
        print(' ' * 8, f'{key:10}: {_to_str(count)} | {modules[module]}')

    remaining_parameters = sum([counts[k] for k in sorted_keys[topk:]])
    print(' ' * 8, f'... and {len(counts) - topk} others accounting for {_to_str(remaining_parameters)} parameters')
    return n_parameters


# ----------------------- DD return scale

# discount = 0.997
DD_RETURN_SCALE = {

    "halfcheetah-medium-expert-v2": 3600,
    "halfcheetah-medium-replay-v2": 1600,
    "halfcheetah-medium-v2": 1700,

    "hopper-medium-expert-v2": 1200,
    "hopper-medium-replay-v2": 1000,
    "hopper-medium-v2": 1000,

    "walker2d-medium-expert-v2": 1600,
    "walker2d-medium-replay-v2": 1300,
    "walker2d-medium-v2": 1300,

    "kitchen-partial-v0": 470,
    "kitchen-mixed-v0": 400,

    "antmaze-medium-play-v2": 100,
    "antmaze-medium-diverse-v2": 100,
    "antmaze-large-play-v2": 100,
    "antmaze-large-diverse-v2": 100,
}


# ------------------------ DQL Qnetworks

class SoftLowerBound(nn.Module):
    def __init__(self, lower_bound):
        super().__init__()
        self.lower_bound = lower_bound

    def forward(self, x):
        return self.lower_bound + torch.nn.functional.softplus(x - self.lower_bound)


class SoftUpperBound(nn.Module):
    def __init__(self, upper_bound):
        super().__init__()
        self.upper_bound = upper_bound

    def forward(self, x):
        return self.upper_bound - torch.nn.functional.softplus(self.upper_bound - x)


class DQLCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
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


class NegDQLCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.q1_model = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, 1), SoftUpperBound(0))

        self.q2_model = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, 1), SoftUpperBound(0))

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1_model(x)

    def q_min(self, obs, act):
        q1, q2 = self.forward(obs, act)
        return torch.min(q1, q2)


# ---------------------- IDQL QNetworks

class IDQLQNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.Q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, 1))
        self.Q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, 1))

    def both(self, obs, act):
        q1, q2 = self.Q1(torch.cat([obs, act], -1)), self.Q2(torch.cat([obs, act], -1))
        return q1, q2

    def forward(self, obs, act):
        return torch.min(*self.both(obs, act))


class IDQLVNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256):
        super().__init__()
        self.V = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, 1))

    def forward(self, obs):
        v = self.V(obs)
        return v


# ---------------------- Freeze and Unfreeze

class FreezeModules:
    def __init__(self, modules):
        self.modules = modules
        self.original_grad_status = {}

    def __enter__(self):
        for module in self.modules:
            for param in module.parameters():
                self.original_grad_status[id(param)] = param.requires_grad
                param.requires_grad = False

    def __exit__(self, type, value, traceback):
        for module in self.modules:
            for param in module.parameters():
                param.requires_grad = self.original_grad_status[id(param)]


class UnfreezeModules:
    def __init__(self, modules):
        self.modules = modules
        self.original_grad_status = {}

    def __enter__(self):
        for module in self.modules:
            for param in module.parameters():
                self.original_grad_status[id(param)] = param.requires_grad
                param.requires_grad = True

    def __exit__(self, type, value, traceback):
        for module in self.modules:
            for param in module.parameters():
                param.requires_grad = self.original_grad_status[id(param)]


class EvalModules:
    def __init__(self, modules):
        self.modules = modules
        self.original_status = {}

    def __enter__(self):
        for module in self.modules:
            self.original_status[id(module)] = module.training
            module.eval()

    def __exit__(self, type, value, traceback):
        for module in self.modules:
            module.train(self.original_status[id(module)])


class TrainModules:
    def __init__(self, modules):
        self.modules = modules
        self.original_status = {}

    def __enter__(self):
        for module in self.modules:
            self.original_status[id(module)] = module.training
            module.train()

    def __exit__(self, type, value, traceback):
        for module in self.modules:
            module.train(self.original_status[id(module)])
