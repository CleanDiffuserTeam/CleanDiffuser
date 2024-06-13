from typing import Union, Optional

import numpy as np
import torch

from cleandiffuser.classifier import BaseClassifier
from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from .edm import EDMArchetecture


class EDMDDIM(EDMArchetecture):
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
            C1: float = 0.001,
            C2: float = 0.008,
            j0: float = 8,

            device: Union[torch.device, str] = "cpu"
    ):
        super().__init__(
            nn_diffusion, nn_condition, fix_mask, loss_weight, classifier, grad_clip_norm,
            diffusion_steps, ema_rate, optim_params, device)

        self.C1, self.C2, self.j0 = C1, C2, j0
        self.u = None

    def set_sample_steps(self, N: int):
        self.sample_steps = N
        bar_alpha = torch.sin(
            torch.arange(self.diffusion_steps + 1, device=self.device) /
            (self.diffusion_steps * (self.C2 + 1)) * np.pi / 2) ** 2
        tmp = torch.max(bar_alpha[:-1] / bar_alpha[1:], torch.tensor(self.C1, device=self.device))
        self.u = torch.empty_like(bar_alpha[:-1])
        for i in range(self.diffusion_steps):
            if i == 0:
                self.u[-1 - i] = (1 / tmp[-1 - i] - 1).sqrt()
            else:
                self.u[-1 - i] = ((self.u[-i] ** 2 + 1) / tmp[-1 - i] - 1).sqrt()
        idx = torch.arange(N, device=self.device)
        self.t_s = self.u[torch.floor(self.j0 + (self.diffusion_steps - 1 - self.j0) /
                                      (N - 1) * idx + 0.5).long()]
        self.sigma_s = self.t_s
        self.scale_s = torch.ones_like(self.sigma_s) * 1.0
        self.dot_sigma_s = torch.ones_like(self.sigma_s) * 1.0
        self.dot_scale_s = torch.zeros_like(self.sigma_s)
        self.x_weight_s = (self.dot_sigma_s / self.sigma_s + self.dot_scale_s / self.scale_s)
        self.D_weight_s = self.dot_sigma_s / self.sigma_s * self.scale_s

    def c_skip(self, sigma): return torch.ones_like(sigma)

    def c_out(self, sigma): return -sigma

    def c_in(self, sigma): return 1 / (1 + sigma ** 2).sqrt()

    def c_noise(self, sigma): return sigma

    def loss_weighting(self, sigma): return 1 / (sigma ** 2)

    def sample_noise_distribution(self, N):
        j = torch.randint(0, self.diffusion_steps, (N, 1), device=self.device)
        return self.u[j]
