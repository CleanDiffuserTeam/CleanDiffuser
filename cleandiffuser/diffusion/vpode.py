from typing import Optional
from typing import Union

import numpy as np
import torch

from cleandiffuser.classifier import BaseClassifier
from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from .edm import EDMArchetecture


class VPODE(EDMArchetecture):
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
            beta_min: float = 0.1,
            beta_max: float = 20.,
            eps_s: float = 1e-3,
            eps_t: float = 1e-5,

            device: Union[torch.device, str] = "cpu"
    ):
        super().__init__(
            nn_diffusion, nn_condition, fix_mask, loss_weight, classifier, grad_clip_norm,
            diffusion_steps, ema_rate, optim_params, device)

        self.beta_min, self.beta_max = beta_min, beta_max
        self.eps_s, self.eps_t = eps_s, eps_t
        self.beta_d = beta_max - beta_min

    def set_sample_steps(self, N: int):
        self.sample_steps = N
        self.t_s = torch.arange(N, device=self.device) / (N - 1) * (1e-3 - 1) + 1
        self.sigma_s = ((0.5 * self.beta_d * (self.t_s ** 2) + self.beta_min * self.t_s).exp() - 1.).sqrt()
        self.scale_s = 1 / (1 + self.sigma_s ** 2).sqrt()
        self.dot_sigma_s = (0.5 * (self.sigma_s ** 2 + 1) * (self.beta_d * self.t_s + self.beta_min) / self.sigma_s)
        self.dot_scale_s = -self.sigma_s / (1 + self.sigma_s ** 2) ** 1.5 * self.dot_sigma_s
        self.x_weight_s = (self.dot_sigma_s / self.sigma_s + self.dot_scale_s / self.scale_s)
        self.D_weight_s = self.dot_sigma_s / self.sigma_s * self.scale_s

    def c_skip(self, sigma): return torch.ones_like(sigma)

    def c_out(self, sigma): return -sigma

    def c_in(self, sigma): return 1 / (1 + sigma ** 2).sqrt()

    def c_noise(self, sigma):
        scale = 1 / (1 + sigma ** 2).sqrt()
        t = ((self.beta_min ** 2 - 4 * self.beta_d * scale.log()).sqrt() - self.beta_min) / self.beta_d
        return ((self.diffusion_steps - 1) * t).long()

    def loss_weighting(self, sigma): return 1 / (sigma ** 2)

    def sample_noise_distribution(self, N):
        t = torch.rand((N, 1), device=self.device) * (1 - self.eps_t) + self.eps_t
        return ((0.5 * self.beta_d * t ** 2 + self.beta_min * t).exp() - 1).sqrt()
