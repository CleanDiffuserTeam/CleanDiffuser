from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.diffusion import DiffusionModel
from typing import Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cleandiffuser.utils import at_least_ndim


class ConsistencyModel(DiffusionModel):

    def __init__(
            self,

            # ----------------- Neural Networks ----------------- #
            nn_diffusion: BaseNNDiffusion,
            nn_condition=None,

            # ----------------- Masks ----------------- #
            # Fix some portion of the input data, and only allow the diffusion model to complete the rest part.
            fix_mask: Union[list, np.ndarray, torch.Tensor] = None,  # be in the shape of `x_shape`
            # Add loss weight
            loss_weight: Union[list, np.ndarray, torch.Tensor] = None,  # be in the shape of `x_shape`

            # ------------------ Plugs ---------------- #
            # Add a classifier to enable classifier-guidance
            classifier=None,

            # ------------------ Params ---------------- #
            grad_clip_norm: Optional[float] = None,
            diffusion_steps: int = 1000,
            ema_rate: float = 0.95,
            optim_params: Optional[dict] = None,
            x_max: Optional[torch.Tensor] = None,
            x_min: Optional[torch.Tensor] = None,

            # ------------------- DPM Params ------------------- #
            total_gradient_steps: int = 1000000,
            eps: float = 0.002,
            T: float = 80,
            rho: float = 7,
            s0: int = 2,
            s1: int = 150,
            sigma_data: float = 0.5,
            device: Union[torch.device, str] = "cpu"
    ):
        super().__init__(
            nn_diffusion, nn_condition, fix_mask, loss_weight, classifier, grad_clip_norm,
            diffusion_steps, ema_rate, optim_params, device)

        self.sigma_data = sigma_data
        self.x_max, self.x_min = x_max, x_min
        self.eps, self.T = eps, T
        self.s0, self.s1, self.K = s0, s1, total_gradient_steps
        self.t = (eps ** (1 / rho) + torch.arange(s1 + 1, device=device)
                  / s1 * (T ** (1 / rho) - eps ** (1 / rho))) ** rho
        self.mu_0 = ema_rate

    @property
    def clip_pred(self):
        return (self.x_max is not None) or (self.x_min is not None)

    def f(self, xt, t, condition, use_ema=False):
        model = self.model_ema if use_ema else self.model
        coeff1 = self.sigma_data ** 2 / ((t - self.eps) ** 2 + self.sigma_data ** 2)
        coeff2 = self.sigma_data * (t - self.eps) / (self.sigma_data ** 2 + t ** 2).sqrt()

        coeff1 = at_least_ndim(coeff1, xt.dim())
        coeff2 = at_least_ndim(coeff2, xt.dim())

        f_theta = coeff1 * xt + coeff2 * model["diffusion"](xt, torch.log(t) / 4., condition)
        if self.clip_pred:
            if self.x_max is not None:
                f_theta = self.x_max - F.softplus(self.x_max - f_theta)
            if self.x_min is not None:
                f_theta = self.x_min + F.softplus(f_theta - self.x_min)
        return f_theta

    def N_schedual(self, k):
        return np.ceil(np.sqrt(k / self.K * ((self.s1 + 1) ** 2 - self.s0 ** 2) + self.s0 ** 2) - 1) + 1

    def ema_rate_schedule(self, N_k):
        return np.exp(self.s0 * np.log(self.mu_0) / N_k)

    # ---------------------------------------------------------------------------
    # Training

    def loss(self, x0, condition=None, N_k=0):

        n = np.random.randint(0, N_k - 1, x0.shape[0])
        t1, t2 = self.t[n + 1], self.t[n]

        condition_vec = self.model["condition"](condition) if condition is not None else None
        condition_vec_ema = self.model_ema["condition"](condition) if condition is not None else None

        x1 = self.f(x0 + at_least_ndim(t1, x0.dim()) * torch.randn_like(x0), t1, condition_vec, use_ema=False)
        x2 = self.f(x0 + at_least_ndim(t2, x0.dim()) * torch.randn_like(x0), t2, condition_vec_ema, use_ema=True)

        loss = (x1 - x2) ** 2

        return (loss * self.loss_weight * (1 - self.fix_mask)).mean()

    def update(self, x0, condition=None, update_ema=True, gradient_step=0, **kwargs):
        N_k = self.N_schedual(gradient_step)
        print(N_k)
        loss = self.loss(x0, condition, N_k)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm) \
            if self.grad_clip_norm else None
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.ema_rate = self.ema_rate_schedule(N_k)
        if update_ema: self.ema_update()
        log = {"loss": loss.item(), "grad_norm": grad_norm}
        return log

    # ---------------------------------------------------------------------------
    # Inference

    def sample(
            self,
            # ---------- the known fixed portion ---------- #
            prior: Optional[torch.Tensor] = None,
            # ----------------- sampling ----------------- #
            n_samples: int = 1,
            sample_steps: int = 1,
            use_ema: bool = True,
            temperature: float = 1.0,
            # ------------------ guidance ------------------ #
            condition_cfg=None,
            mask_cfg=None,
            w_cfg: float = 0.0,
            condition_cg=None,
            w_cg: float = 0.0,
            # ------------------ others ------------------ #
            requires_grad: bool = False,
            preserve_history: bool = False,
            **kwargs,
    ):
        # initialize logger
        log = {
            "sample_history": np.empty((n_samples, sample_steps + 1, *prior.shape)) if preserve_history else None, }

        # choose the model
        model = self.model_ema if use_ema else self.model

        # initialize the samples
        xt = torch.randn_like(prior, device=self.device) * temperature * self.T
        xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
        if preserve_history: log["sample_history"][:, 0] = xt.cpu().numpy()

        n_s = np.linspace(0, self.s1, sample_steps + 1, dtype=np.int64)[1:]
        t_s = self.t[n_s].cpu().numpy()

        # preprocess the conditions
        with torch.set_grad_enabled(requires_grad):
            condition_vec = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None

        for t in reversed(t_s):

            t_batch = torch.tensor(t, device=self.device, dtype=torch.long).repeat(n_samples)
            xt = self.f(xt + np.sqrt(t ** 2 - self.eps ** 2) * torch.randn_like(xt),
                        t_batch, condition_vec, use_ema=use_ema)

            # Fix the known portion
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
            if preserve_history: log["sample_history"][:, 1] = xt.cpu().numpy()

        return xt, log
