import random
from typing import Callable, Dict, List, Union

import numba
import numpy as np
import torch
import torch.nn as nn

from cleandiffuser.utils.typings import TensorDict


def at_least_ndim(x: Union[np.ndarray, torch.Tensor, int, float], ndim: int, pad: int = 0):
    """Add dimensions to the input tensor to make it at least ndim-dimensional.

    Args:
        x: Union[np.ndarray, torch.Tensor, int, float], input tensor
        ndim: int, minimum number of dimensions
        pad: int, padding direction. `0`: pad in the last dimension, `1`: pad in the first dimension

    Returns:
        Any of these 2 options

        - np.ndarray or torch.Tensor: reshaped tensor
        - int or float: input value

    Examples:
        >>> x = np.random.rand(3, 4)
        >>> at_least_ndim(x, 3, 0).shape
        (3, 4, 1)
        >>> x = torch.randn(3, 4)
        >>> at_least_ndim(x, 4, 1).shape
        (1, 1, 3, 4)
        >>> x = 1
        >>> at_least_ndim(x, 3)
        1
    """
    if isinstance(x, np.ndarray):
        if ndim > x.ndim:
            if pad == 0:
                return np.reshape(x, x.shape + (1,) * (ndim - x.ndim))
            else:
                return np.reshape(x, (1,) * (ndim - x.ndim) + x.shape)
        else:
            return x
    elif isinstance(x, torch.Tensor):
        if ndim > x.ndim:
            if pad == 0:
                return torch.reshape(x, x.shape + (1,) * (ndim - x.ndim))
            else:
                return torch.reshape(x, (1,) * (ndim - x.ndim) + x.shape)
        else:
            return x
    elif isinstance(x, (int, float)):
        return x
    else:
        raise ValueError(f"Unsupported type {type(x)}")


def to_tensor(x, device=None):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, (np.ndarray, list, tuple, int, float)):
        return torch.tensor(x, device=device)
    else:
        raise ValueError(f"Unsupported type {type(x)}")


def concat_zeros(x: torch.Tensor, dim: int = 0):
    """Concatenate zeros to the tensor.

    Args:
        x (torch.Tensor): input tensor
        dim (int): concatenate dimension. Default: 0

    Returns:
        torch.Tensor: output tensor
    """
    return torch.cat([x, torch.zeros_like(x)], dim=dim)


@numba.jit(nopython=True)
def create_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
) -> np.ndarray:
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)

    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0  # episode start index
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]  # episode end index
        episode_length = end_idx - start_idx  # episode length

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset

            assert start_offset >= 0
            assert end_offset >= 0
            assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)

            indices.append(
                [
                    buffer_start_idx,
                    buffer_end_idx,
                    sample_start_idx,
                    sample_end_idx,
                    end_idx,
                ]
            )
    indices = np.array(indices)
    return indices


def get_mask(
    x: Union[torch.Tensor, TensorDict],
    prob: float = 0.0,
    dims: Union[int, List[int]] = 0,
):
    if prob <= 0.0:
        return 1.0
    if isinstance(dims, int):
        dims = [dims]
    if isinstance(x, torch.Tensor):
        mask_shape = tuple(x.shape[i] if i in dims else 1 for i in range(x.dim()))
        mask = (torch.rand(mask_shape, device=x.device) > prob).to(x.dtype)
        return mask
    else:
        return dict_apply(x, get_mask, prob=prob, dims=dims)


def linear_beta_schedule(beta_min: float = 1e-4, beta_max: float = 0.02, T: int = 1000):
    return np.linspace(beta_min, beta_max, T)


def cosine_beta_schedule(s: float = 0.008, T: int = 1000):
    f = np.cos((np.arange(T + 1) / T + s) / (1 + s) * np.pi / 2.0) ** 2
    alpha_bar = f / f[0]
    beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return beta.clip(None, 0.999)


# ================ Discretization ==================
def uniform_discretization(T: int = 1000, eps: float = 1e-3):
    return torch.linspace(eps, 1.0, T)


def quad_discretization(T: int = 1000, eps: float = 1e-3, n: float = 1.5):
    return (1 - eps) * torch.linspace(0, 1, T) ** n + eps


SUPPORTED_DISCRETIZATIONS = {"uniform": uniform_discretization, "quad": quad_discretization}


# ================= Noise schedules =================
def linear_noise_schedule(t_diffusion: torch.Tensor, beta0: float = 0.1, beta1: float = 20.0):
    log_alpha = -(beta1 - beta0) / 4.0 * (t_diffusion**2) - beta0 / 2.0 * t_diffusion
    alpha = log_alpha.exp()
    sigma = (1.0 - alpha**2).sqrt()
    return alpha, sigma


def inverse_linear_noise_schedule(
    alpha: torch.Tensor = None,
    sigma: torch.Tensor = None,
    logSNR: torch.Tensor = None,
    beta0: float = 0.1,
    beta1: float = 20.0,
):
    assert (logSNR is not None) or (alpha is not None and sigma is not None)
    lmbda = (alpha / sigma).log() if logSNR is None else logSNR
    t_diffusion = (
        2
        * (1 + (-2 * lmbda).exp()).log()
        / (beta0 + (beta0**2 + 2 * (beta1 - beta0) * (1 + (-2 * lmbda).exp()).log()))
    )
    return t_diffusion


def cosine_noise_schedule(t_diffusion: torch.Tensor, s: float = 0.008):
    eps = t_diffusion[0]
    t_diffusion = t_diffusion - eps
    t_diffusion = (t_diffusion * 0.9946) + eps
    alpha = (np.pi / 2.0 * (t_diffusion + s) / (1 + s)).cos() / np.cos(np.pi / 2.0 * s / (1 + s))
    sigma = (1.0 - alpha**2).sqrt()
    return alpha, sigma


def inverse_cosine_noise_schedule(
    alpha: torch.Tensor = None,
    sigma: torch.Tensor = None,
    logSNR: torch.Tensor = None,
    s: float = 0.008,
):
    assert (logSNR is not None) or (alpha is not None and sigma is not None)
    lmbda = (alpha / sigma).log() if logSNR is None else logSNR
    t_diffusion = (
        2
        * (1 + s)
        / np.pi
        * torch.arccos(
            (-0.5 * (1 + (-2 * lmbda).exp()).log() + np.log(np.cos(np.pi * s / 2 / (s + 1)))).exp()
        )
        - s
    )
    return t_diffusion


SUPPORTED_NOISE_SCHEDULES = {
    "linear": {
        "forward": linear_noise_schedule,
        "reverse": inverse_linear_noise_schedule,
    },
    "cosine": {
        "forward": cosine_noise_schedule,
        "reverse": inverse_cosine_noise_schedule,
    },
}


# ================= Sampling step schedule ===============
def uniform_sampling_step_schedule(T: int = 1000, sampling_steps: int = 10):
    return torch.linspace(0, T - 1, sampling_steps + 1, dtype=torch.long)


def uniform_sampling_step_schedule_continuous(trange=None, sampling_steps: int = 10):
    if trange is None:
        trange = [1e-3, 1.0]
    return torch.linspace(trange[0], trange[1], sampling_steps + 1, dtype=torch.float32)


def quad_sampling_step_schedule(T: int = 1000, sampling_steps: int = 10, n: int = 1.5):
    schedule = (T - 1) * (torch.linspace(0, 1, sampling_steps + 1, dtype=torch.float32) ** n)
    return schedule.to(torch.long)


def quad_sampling_step_schedule_continuous(trange=None, sampling_steps: int = 10, n: int = 1.5):
    if trange is None:
        trange = [1e-3, 1.0]
    schedule = (trange[1] - trange[0]) * (
        torch.linspace(0, 1, sampling_steps + 1, dtype=torch.float32) ** n
    ) + trange[0]
    return schedule


def cat_cos_sampling_step_schedule(T: int = 1000, sampling_steps: int = 10, n: int = 2.0):
    idx = torch.linspace(0, 1, sampling_steps + 1, dtype=torch.float32)
    idx = 0.5 * (2 * (idx > 0.5) - 1) * torch.sin(np.pi * torch.abs(idx - 0.5)) ** (1 / n) + 0.5
    schedule = (T - 1) * idx
    return schedule.to(torch.long)


def cat_cos_sampling_step_schedule_continuous(trange=None, sampling_steps: int = 10, n: int = 2.0):
    if trange is None:
        trange = [1e-3, 1.0]
    idx = torch.linspace(0, 1, sampling_steps + 1, dtype=torch.float32)
    idx = 0.5 * (2 * (idx > 0.5) - 1) * torch.sin(np.pi * torch.abs(idx - 0.5)) ** (1 / n) + 0.5
    schedule = (trange[1] - trange[0]) * idx + trange[0]
    return schedule


def quad_cos_sampling_step_schedule(T: int = 1000, sampling_steps: int = 10, n: int = 2.0):
    idx = torch.linspace(0, 1, sampling_steps + 1, dtype=torch.float32)
    idx = ((torch.sin(np.pi * (idx - 0.5)) + 1) / 2) ** n
    schedule = (T - 1) * idx
    return schedule.to(torch.long)


def quad_cos_sampling_step_schedule_continuous(trange=None, sampling_steps: int = 10, n: int = 2.0):
    if trange is None:
        trange = [1e-3, 1.0]
    idx = torch.linspace(0, 1, sampling_steps + 1, dtype=torch.float32)
    idx = ((torch.sin(np.pi * (idx - 0.5)) + 1) / 2) ** n
    schedule = (trange[1] - trange[0]) * idx + trange[0]
    return schedule


SUPPORTED_SAMPLING_STEP_SCHEDULE = {
    "uniform": uniform_sampling_step_schedule,
    "uniform_continuous": uniform_sampling_step_schedule_continuous,
    "quad": quad_sampling_step_schedule,
    "quad_continuous": quad_sampling_step_schedule_continuous,
    "cat_cos": cat_cos_sampling_step_schedule,
    "cat_cos_continuous": cat_cos_sampling_step_schedule_continuous,
    "quad_cos": quad_cos_sampling_step_schedule,
    "quad_cos_continuous": quad_cos_sampling_step_schedule_continuous,
}


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ema_update(model: nn.Module, model_ema: nn.Module, ema_rate: float):
    for param, param_ema in zip(model.parameters(), model_ema.parameters()):
        param_ema.data.mul_(ema_rate).add_(param.data, alpha=1 - ema_rate)


# -----------------------------------------------------------
class PositionalEmbedding(nn.Module):
    """Positional Embedding.

    This module uses positional features to embed (..., ) -> (..., dim // 4),
    and then uses a MLP to proj (..., dim // 4) -> (..., dim).

    Args:
        dim (int): Embedding dimension. It should be divisible by 8.
        max_positions (int): Maximum number of positions
        endpoint (bool): Whether to include the endpoint. Default: False

    Examples:
        >>> embedding = PositionalEmbedding(128)
        >>> x = torch.randint(1000, (32,))
        >>> y = embedding(x)
        >>> y.shape
        torch.Size([32, 128])
    """

    def __init__(self, dim: int, max_positions: int = 10000, endpoint: bool = False):
        super().__init__()
        freqs = torch.arange(start=0, end=dim // 8, dtype=torch.float32)
        freqs = freqs / (dim // 8 - (1 if endpoint else 0))
        freqs = (1 / max_positions) ** freqs
        self.freqs = nn.Parameter(freqs, requires_grad=False)
        self.mlp = nn.Sequential(nn.Linear(dim // 4, dim), nn.Mish(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(-1)
        x = x * at_least_ndim(self.freqs, x.dim(), 1)
        x = torch.cat([x.cos(), x.sin()], dim=-1)
        return self.mlp(x)


class UntrainablePositionalEmbedding(nn.Module):
    """Untrainable Positional Embedding.

    This module uses positional features to embed (..., ) -> (..., dim).
    It has no trainable parameters.

    Args:
        dim (int): Embedding dimension. It should be divisible by 2.
        max_positions (int): Maximum number of positions
        endpoint (bool): Whether to include the endpoint. Default: False

    Examples:
        >>> embedding = UntrainablePositionalEmbedding(128)
        >>> x = torch.randint(1000, (32,))
        >>> y = embedding(x)
        >>> y.shape
        torch.Size([32, 128])
    """

    def __init__(self, dim: int, max_positions: int = 10000, endpoint: bool = False):
        super().__init__()
        freqs = torch.arange(start=0, end=dim // 2, dtype=torch.float32)
        freqs = freqs / (dim // 2 - (1 if endpoint else 0))
        freqs = (1 / max_positions) ** freqs
        self.freqs = nn.Parameter(freqs, requires_grad=False)

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(-1)
        x = x * at_least_ndim(self.freqs, x.dim(), 1)
        return torch.cat([x.cos(), x.sin()], dim=-1)


class FourierEmbedding(nn.Module):
    """Fourier Embedding.

    This module uses fourier features to embed (..., ) -> (..., dim // 4),
    and then uses a MLP to proj (..., dim // 4) -> (..., dim).

    Args:
        dim (int): Embedding dimension. It should be divisible by 8.
        scale (float): Scale factor. Default: 16.0

    Examples:
        >>> embedding = FourierEmbedding(128)
        >>> x = torch.rand(1000, (32,))
        >>> y = embedding(x)
        >>> y.shape
        torch.Size([32, 128])
    """

    def __init__(self, dim: int, scale: float = 16.0):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(dim // 8) * scale * 2 * np.pi, requires_grad=False)
        self.mlp = nn.Sequential(nn.Linear(dim // 4, dim), nn.Mish(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(-1)
        x = x * at_least_ndim(self.freqs, x.dim(), 1)
        x = torch.cat([x.cos(), x.sin()], -1)
        return self.mlp(x)


class UntrainableFourierEmbedding(nn.Module):
    """Untrainable Fourier Embedding.

    This module uses fourier features to embed (..., ) -> (..., dim).
    It has no trainable parameters.

    Args:
        dim (int): Embedding dimension. It should be divisible by 2.
        scale (float): Scale factor. Default: 16.0

    Examples:
        >>> embedding = UntrainableFourierEmbedding(128)
        >>> x = torch.rand(1000, (32,))
        >>> y = embedding(x)
        >>> y.shape
        torch.Size([32, 128])
    """

    def __init__(self, dim: int, scale: float = 16.0):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(dim // 2) * scale * 2 * np.pi, requires_grad=False)

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(-1)
        x = x * at_least_ndim(self.freqs, x.dim(), 1)
        return torch.cat([x.cos(), x.sin()], -1)


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
        return f"{(num / 1e6):.2f} M"
    else:
        return f"{(num / 1e3):.2f} k"


def param_to_module(param):
    module_name = param[::-1].split(".", maxsplit=1)[-1][::-1]
    return module_name


def report_parameters(model, topk=10):
    counts = {k: p.numel() for k, p in model.named_parameters() if p.requires_grad}
    n_parameters = sum(counts.values())
    print(f"Total parameters: {_to_str(n_parameters)}")

    modules = dict(model.named_modules())
    sorted_keys = sorted(counts, key=lambda x: -counts[x])
    for i in range(topk):
        key = sorted_keys[i]
        count = counts[key]
        module = param_to_module(key)
        print(" " * 8, f"{key:10}: {_to_str(count)} | {modules[module]}")

    remaining_parameters = sum([counts[k] for k in sorted_keys[topk:]])
    print(
        " " * 8,
        f"... and {len(counts) - topk} others accounting for {_to_str(remaining_parameters)} parameters",
    )
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


def dict_apply(x: Dict[str, torch.Tensor], func: Callable[[torch.Tensor], torch.Tensor], **kwargs):
    if isinstance(x, torch.Tensor):
        return func(x, **kwargs)
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func, **kwargs)
        elif value is None:
            result[key] = None
        else:
            result[key] = func(value, **kwargs)
    return result


def dict_operation(
    x: Union[torch.Tensor, TensorDict],
    y: Union[torch.Tensor, TensorDict],
    func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    **kwargs,
):
    if not (isinstance(x, dict) and isinstance(y, dict)):
        return func(x, y, **kwargs)
    result = {}
    for key in x.keys():
        result[key] = dict_operation(x[key], y[key], func, **kwargs)
    return result


def loop_dataloader(dl):
    while True:
        for b in dl:
            yield b


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
