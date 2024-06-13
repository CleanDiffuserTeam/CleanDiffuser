from typing import Optional
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from cleandiffuser.classifier import BaseClassifier
from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import at_least_ndim
from .basic import DiffusionModel


class DDIM(DiffusionModel):

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

            # ------------------- DPM Params ------------------- #
            noise_schedule: str = "linear",  # or cosine
            t_eps: float = 1e-3,

            device: Union[torch.device, str] = "cpu"
    ):
        super().__init__(
            nn_diffusion, nn_condition, fix_mask, loss_weight, classifier, grad_clip_norm,
            diffusion_steps, ema_rate, optim_params, device)

        self.noise_schedule = noise_schedule
        self.t_eps = t_eps

    @property
    def t_range(self):
        if self.noise_schedule == "linear":
            return self.t_eps, 1.
        elif self.noise_schedule == "cosine":
            return self.t_eps, 0.9946
        else:
            raise ValueError(f"noise_schedule should be 'linear' or 'cosine', but got {self.noise_schedule}.")

    def alpha_schedule(self, t):
        if self.noise_schedule == "linear":
            beta0, beta1 = 0.1, 20
            return (-(beta1-beta0)/4*(t**2) - beta0/2*t).exp()
        elif self.noise_schedule == "cosine":
            s = 0.008
            return ((torch.cos(np.pi/2*(t+s)/(1+s))).log() - np.log(np.cos(np.pi/2*s/(1+s)))).exp()
        else:
            raise ValueError(f"noise_schedule should be 'linear' or 'cosine', but got {self.noise_schedule}.")

    # ---------------------------------------------------------------------------
    # Training

    def add_noise(self, x0, t=None, eps=None):
        if t is None:
            t = torch.rand((x0.shape[0], ), device=self.device)
            t = self.t_range[0] + t * (self.t_range[1] - self.t_range[0])
        eps = torch.randn_like(x0) if eps is None else eps
        alpha = self.alpha_schedule(at_least_ndim(t, x0.dim()))
        sigma = (1 - alpha ** 2).sqrt()
        xt = x0 * alpha + sigma * eps
        xt = xt * (1. - self.fix_mask) + x0 * self.fix_mask
        return xt, t, eps

    def loss(self, x0, condition=None):
        xt, t, eps = self.add_noise(x0)
        condition = self.model["condition"](condition) if condition is not None else None
        loss = (self.model["diffusion"](xt, t, condition) - eps) ** 2
        return (loss * self.loss_weight).mean()

    def update(self, x0, condition=None, **kwargs):
        self.optimizer.zero_grad()
        loss = self.loss(x0, condition)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm) \
            if self.grad_clip_norm else None
        self.optimizer.step()
        self.ema_update()
        log = {"loss": loss.item(), "grad_norm": grad_norm}
        return log

    def update_classifier(self, x0, condition):
        xt, t, eps = self.add_noise(x0)
        log = self.classifier.update(xt, t, condition)
        return log

    # ---------------------------------------------------------------------------
    # Inference

    def sample(
            self,
            # ---------- the known fixed portion ---------- #
            prior: Optional[torch.Tensor] = None,
            # ----------------- sampling ----------------- #
            n_samples: int = 1,
            sample_steps: int = None,
            use_ema: bool = True,
            solver: str = "euler",
            # ------------------ guidance ------------------ #
            condition_cfg=None,
            mask_cfg=None,
            w_cfg: float = 0.0,
            condition_cg=None,
            w_cg: float = 0.0,

            preserve_history: bool = False,
    ):
        log, x_history = {}, None
        model = self.model_ema if use_ema else self.model

        t = torch.linspace(self.t_range[1], self.t_range[0], sample_steps + 1, device=self.device)
        alphas = self.alpha_schedule(t)
        sigmas = (1 - alphas ** 2).sqrt()
        logSNRs = (alphas / sigmas).log()

        if prior is None:
            xt = torch.randn((n_samples, *self.default_x_shape), device=self.device)
        else:
            xt = torch.randn_like(prior, device=self.device) * 0.5
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask

        if preserve_history:
            x_history = np.empty((n_samples, sample_steps + 1, *xt.shape))
            x_history[:, 0] = xt.cpu().numpy()

        with torch.no_grad():
            condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
            condition_vec_cg = condition_cg

        for i in range(sample_steps):

            h = logSNRs[i + 1] - logSNRs[i]

            t_batch = t[i].repeat(n_samples)

            # ----------------- CFG ----------------- #
            with torch.no_grad():
                if w_cfg != 0.0 and w_cfg != 1.0:
                    condition_vec_cfg = torch.cat([condition_vec_cfg, torch.zeros_like(condition_vec_cfg)], 0)
                    eps_theta = model["diffusion"](
                        torch.repeat_interleave(xt, 2, dim=0),
                        torch.repeat_interleave(t_batch, 2, dim=0),
                        condition_vec_cfg)
                    eps_theta = w_cfg * eps_theta[:n_samples] + (1. - w_cfg) * eps_theta[n_samples:]
                else:
                    eps_theta = model["diffusion"](xt, t_batch, condition_vec_cfg)
            # ----------------- CG ----------------- #
            if self.classifier is not None and w_cg != 0.0:
                log_p, grad = self.classifier.gradients(xt.clone(), t_batch, condition_vec_cg)
                eps_theta = eps_theta - w_cg * sigmas[i] * grad
            else:
                log_p = None

            # do not change the fixed portion
            xt = alphas[i + 1] / alphas[i] * xt - sigmas[i + 1] * torch.expm1(h) * eps_theta
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask

            if preserve_history:
                x_history[:, t] = xt.cpu().numpy()
            log["log_p"] = log_p

        log["sample_history"] = x_history
        if log["log_p"] is None and self.classifier is not None and condition_cg is not None:
            with torch.no_grad():
                logp = self.classifier.logp(xt, t_batch, condition_vec_cg)
            log["log_p"] = logp

        return xt, log

