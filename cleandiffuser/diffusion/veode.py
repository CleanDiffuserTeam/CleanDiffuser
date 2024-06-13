from typing import Optional
from typing import Union

import numpy as np
import torch

from cleandiffuser.classifier import BaseClassifier
from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from .edm import EDMArchetecture


class VEODE(EDMArchetecture):
    def __init__(
            self,

            # ----------------- Neural Networks ----------------- #
            nn_diffusion: BaseNNDiffusion,
            nn_condition: Optional[BaseNNCondition] = None,

            # ----------------- Masks ----------------- #
            # Fix some portion of the input data, and only allow the diffusion model to complete the rest part.
            fix_mask: Union[list, np.ndarray, torch.Tensor] = None,  # be in the shape of `x_shape`
            # Add loss weight
            loss_weight: Union[list, np.ndarray, torch.Tensor] = None,  # be in the shape of `x_shape`

            # ------------------ Plugs ---------------- #
            # Add a classifier to enable classifier-guidance
            classifier: Optional[BaseClassifier] = None,

            # ------------------ Params ---------------- #
            grad_clip_norm: Optional[float] = None,
            diffusion_steps: int = 1000,
            ema_rate: float = 0.995,
            optim_params: Optional[dict] = None,

            # ------------------- EDM Params ------------------- #
            sigma_min: float = 0.02,
            sigma_max: float = 100.,

            device: Union[torch.device, str] = "cpu"
    ):
        super().__init__(
            nn_diffusion, nn_condition, fix_mask, loss_weight, classifier, grad_clip_norm,
            diffusion_steps, ema_rate, optim_params, device)

        self.sigma_min, self.sigma_max = sigma_min, sigma_max

    def set_sample_steps(self, N: int):
        self.sample_steps = N
        self.sigma_s = self.sigma_max * (self.sigma_min / self.sigma_max) ** (
                    torch.arange(N, device=self.device) / (N - 1))
        self.t_s = self.sigma_s ** 2
        self.scale_s = torch.ones_like(self.sigma_s) * 1.0
        self.dot_sigma_s = 1 / (2 * self.sigma_s)
        self.dot_scale_s = torch.zeros_like(self.sigma_s)
        self.x_weight_s = (self.dot_sigma_s / self.sigma_s + self.dot_scale_s / self.scale_s)
        self.D_weight_s = self.dot_sigma_s / self.sigma_s * self.scale_s

    def c_skip(self, sigma): return torch.ones_like(sigma)

    def c_out(self, sigma): return sigma

    def c_in(self, sigma): return torch.ones_like(sigma)

    def c_noise(self, sigma): return (0.5 * sigma).log()

    def loss_weighting(self, sigma): return 1 / (sigma ** 2)

    def sample_noise_distribution(self, N):
        log_sigma = torch.rand((N, 1), device=self.device) * np.log(self.sigma_max / self.sigma_min) + np.log(
            self.sigma_min)
        return log_sigma.exp()
