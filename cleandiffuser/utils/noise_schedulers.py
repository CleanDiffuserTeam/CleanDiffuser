import numpy as np
import torch


class NoiseScheduler:
    """Noise scheduler maps the diffusion time-steps to the noise schedule (alpha & sigma)."""

    def t_to_schedule(self, t: torch.Tensor, **kwargs):
        """From diffusion time-steps to the noise schedule.

        Args:
            t (torch.Tensor):
                Diffusion time-steps. shape: (batch_size, ...).

        Returns:
            alpha, sigma (torch.Tensor):
                The noise schedule. shape: (batch_size, ...).
        """
        raise NotImplementedError

    def schedule_to_t(self, alpha: torch.Tensor, sigma: torch.Tensor, **kwargs):
        """From the noise schedule to the diffusion time-steps.

        Args:
            alpha (torch.Tensor):
                Alpha of the noise schedule. shape: (batch_size, ...).
            sigma (torch.Tensor):
                Sigma of the noise schedule. shape: (batch_size, ...).

        Returns:
            t (torch.Tensor):
                The diffusion time-steps. shape: (batch_size, ...).
        """
        raise NotImplementedError


class LinearNoiseScheduler(NoiseScheduler):
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min, self.beta_max = beta_min, beta_max

    def t_to_schedule(self, t: torch.Tensor):
        alpha = torch.exp(-(self.beta_max - self.beta_min) * t.pow(2) / 4.0 - self.beta_min * t / 2)
        sigma = (1.0 - alpha**2).sqrt()
        return alpha, sigma

    def schedule_to_t(self, alpha: torch.Tensor, sigma: torch.Tensor):
        lmbda = (alpha / sigma).log()
        t = (
            2
            * (1 + (-2 * lmbda).exp()).log()
            / (
                self.beta_min
                + (self.beta_min**2 + 2 * (self.beta_max - self.beta_min) * (1 + (-2 * lmbda).exp()).log())
            )
        )
        return t


class CosineNoiseScheduler(NoiseScheduler):
    def __init__(self, s: float = 0.008, t_max: float = 0.9946):
        self.s, self.t_max = s, t_max
        self.div = np.cos(s / (s + 1) * np.pi / 2.0)

    def t_to_schedule(self, t: torch.Tensor):
        scaled_t = t * self.t_max
        alpha = torch.cos((self.s + scaled_t) / (self.s + 1) * np.pi / 2.0) / self.div
        sigma = (1.0 - alpha**2).sqrt()
        return alpha, sigma

    def schedule_to_t(self, alpha: torch.Tensor, sigma: torch.Tensor):
        lmbda = (alpha / sigma).log()
        t = (
            2
            * (1 + self.s)
            / np.pi
            * torch.arccos(
                (-0.5 * (1 + (-2 * lmbda).exp()).log() + np.log(np.cos(np.pi * self.s / 2 / (self.s + 1)))).exp()
            )
            - self.s
        )
        return t / self.t_max


class SigmaExponentialNoiseScheduler(NoiseScheduler):
    def __init__(self, sigma_min: float = 0.02, sigma_max: float = 100.0):
        self.sigma_min, self.sigma_max = sigma_min, sigma_max

    def t_to_schedule(self, t: torch.Tensor):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        alpha = torch.ones_like(sigma)
        return alpha, sigma

    def schedule_to_t(self, alpha: torch.Tensor, sigma: torch.Tensor):
        t = torch.log(sigma / self.sigma_min) / np.log(self.sigma_max / self.sigma_min)
        return t


class SigmaLinearNoiseScheduler(NoiseScheduler):
    def __init__(self, sigma_min: float = 0.002, sigma_max: float = 80.0):
        self.sigma_min, self.sigma_max = sigma_min, sigma_max

    def t_to_schedule(self, t: torch.Tensor):
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * t
        alpha = torch.ones_like(sigma)
        return alpha, sigma

    def schedule_to_t(self, alpha: torch.Tensor, sigma: torch.Tensor):
        t = (sigma - self.sigma_min) / (self.sigma_max - self.sigma_min)
        return t


def get_noise_scheduler(noise_scheduler: str, **kwargs) -> NoiseScheduler:
    if noise_scheduler == "linear":
        return LinearNoiseScheduler(**kwargs)
    elif noise_scheduler == "cosine":
        return CosineNoiseScheduler(**kwargs)
    elif noise_scheduler == "sigma_exponential":
        return SigmaExponentialNoiseScheduler(**kwargs)
    elif noise_scheduler == "sigma_linear":
        return SigmaLinearNoiseScheduler(**kwargs)
    else:
        raise NotImplementedError


if __name__ == "__main__":

    def t_to_snr(t):
        alpha, sigma = scheduler.t_to_schedule(t)
        return (alpha / sigma).log()

    t_range = torch.tensor([1e-3, 1.0])
    scheduler = LinearNoiseScheduler()
    alpha, sigma = scheduler.t_to_schedule(t_range)
    SNR_range = (alpha / sigma).log()
    snrs = torch.linspace(SNR_range[0], SNR_range[1], 11)

    ts = torch.ones((11,)) * 0.5
    ts_max = torch.ones_like(ts) * t_range[1]
    ts_min = torch.ones_like(ts) * t_range[0]

    for _ in range(500):
        snr = t_to_snr(ts)
        delta = (snr - snrs).abs()[1:-1]
        below = snr < snrs
        above = snr > snrs
        ts_max[torch.where(below)[0]] = ts[torch.where(below)]
        ts_min[torch.where(above)[0]] = ts[torch.where(above)]
        ts[torch.where(below)] = (ts[torch.where(below)] + ts_min[torch.where(below)]) / 2
        ts[torch.where(above)] = (ts[torch.where(above)] + ts_max[torch.where(above)]) / 2
        if delta.max() < 1e-4:
            break
