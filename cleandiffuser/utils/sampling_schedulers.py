from typing import List, Union

import torch

from cleandiffuser.utils.noise_schedulers import NoiseScheduler


class SamplingScheduler:
    def __init__(self):
        self.cache = {}

    def record(self, key, value):
        self.cache[key] = value

    def schedule(self, sampling_steps: int, device: Union[str, torch.device] = "cpu", **kwargs):
        return None

    def __call__(self, sampling_steps: int, device: Union[str, torch.device] = "cpu", **kwargs):
        if sampling_steps in self.cache.keys():
            return self.cache[sampling_steps]
        else:
            schedule = self.schedule(sampling_steps, device, **kwargs)
            self.record(sampling_steps, schedule)
            return schedule


class PowerSamplingScheduler(SamplingScheduler):
    def __init__(self, T: int = 0, t_min: float = 1e-3, t_max: float = 1.0, power: float = 2.0, **kwargs):
        super().__init__()
        self.t_min, self.t_max = t_min, t_max
        self.T = T
        self.power = power

    def schedule(self, sampling_steps: int, device: Union[str, torch.device] = "cpu", **kwargs):
        base_seq = torch.linspace(0, 1, sampling_steps + 1, dtype=torch.float32, device=device) ** self.power
        t = base_seq * (self.t_max - self.t_min) + self.t_min
        # Continuous
        if self.T <= 0:
            return t
        # Discrete
        else:
            assert (
                sampling_steps <= self.T
            ), f"Sampling steps {sampling_steps} must be less than or equal to Diffusion steps {self.T}."
            t = torch.round(t * self.T)
            return t


class QuadSamplingScheduler(PowerSamplingScheduler):
    def __init__(self, T: int = 0, t_min: float = 1e-3, t_max: float = 1.0, **kwargs):
        super().__init__(T, t_min, t_max, power=2.0, **kwargs)


class LinearSamplingScheduler(PowerSamplingScheduler):
    def __init__(self, T: int = 0, t_min: float = 1e-3, t_max: float = 1.0, **kwargs):
        super().__init__(T, t_min, t_max, power=1.0, **kwargs)


class ManualSamplingScheduler(SamplingScheduler):
    def __init__(self, schedule: List[Union[int, float]], **kwargs):
        self.schedule = schedule

    def schedule(self, sampling_steps: int, device: Union[str, torch.device] = "cpu", **kwargs):
        return torch.tensor(self.schedule, dtype=torch.float32, device=device)


class UniformLogSNRSamplingScheduler(SamplingScheduler):
    def __init__(self, noise_scheduler: NoiseScheduler, T: int = 0, t_min: float = 1e-3, t_max: float = 1.0, **kwargs):
        super().__init__()
        self.t_min, self.t_max = t_min, t_max
        self.noise_scheduler = noise_scheduler
        self.T = T

    def _t_to_snr(self, t: torch.Tensor):
        alpha, sigma = self.noise_scheduler.t_to_schedule(t)
        return (alpha / sigma).log()

    def schedule(self, sampling_steps: int, device: Union[str, torch.device] = "cpu", max_iter: int = 500, **kwargs):
        t_range = torch.tensor([self.t_min, self.t_max])
        snrs = torch.linspace(*self._t_to_snr(t_range), sampling_steps + 1, device=device)

        ts = torch.ones((sampling_steps + 1,), device=device) * (self.t_max + self.t_min) / 2
        ts[0] = t_range[0]
        ts[-1] = t_range[1]
        ts_max = torch.ones_like(ts) * t_range[1]
        ts_min = torch.ones_like(ts) * t_range[0]

        for _ in range(max_iter):
            snr = self._t_to_snr(ts)
            delta = (snr - snrs).abs()[1:-1]
            below = snr < snrs
            above = snr > snrs
            ts_max[torch.where(below)] = ts[torch.where(below)]
            ts_min[torch.where(above)] = ts[torch.where(above)]
            ts[torch.where(below)] = (ts[torch.where(below)] + ts_min[torch.where(below)]) / 2
            ts[torch.where(above)] = (ts[torch.where(above)] + ts_max[torch.where(above)]) / 2
            if delta.max() < 1e-4:
                break

        # Continuous
        if self.T <= 0:
            return ts
        # Discrete
        else:
            assert (
                sampling_steps <= self.T
            ), f"Sampling steps {sampling_steps} must be less than or equal to Diffusion steps {self.T}."

            return torch.round(ts * self.T)


def get_sampling_scheduler(sampling_scheduler: str, **kwargs) -> SamplingScheduler:
    if sampling_scheduler == "linear":
        return LinearSamplingScheduler(**kwargs)
    elif sampling_scheduler == "quad":
        return QuadSamplingScheduler(**kwargs)
    elif sampling_scheduler == "power":
        return PowerSamplingScheduler(**kwargs)
    elif sampling_scheduler == "manual":
        return ManualSamplingScheduler(**kwargs)
    elif sampling_scheduler == "uniform_logsnr":
        return UniformLogSNRSamplingScheduler(**kwargs)
    else:
        raise NotImplementedError
