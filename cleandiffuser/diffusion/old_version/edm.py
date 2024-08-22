from typing import Optional
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from cleandiffuser.utils import at_least_ndim
from cleandiffuser.classifier import BaseClassifier
from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from .basic import DiffusionModel


class EDMArchetecture(DiffusionModel):

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

            device: Union[torch.device, str] = "cpu"
    ):
        super().__init__(
            nn_diffusion, nn_condition, fix_mask, loss_weight, classifier, grad_clip_norm,
            diffusion_steps, ema_rate, optim_params, device)

        self.dot_scale_s = None
        self.dot_sigma_s = None
        self.scale_s = None
        self.t_s = None
        self.sigma_s = None
        self.x_weight_s, self.D_weight_s = None, None
        self.sample_steps = None

    def set_sample_steps(self, N: int):
        raise NotImplementedError

    def c_skip(self, sigma):
        raise NotImplementedError

    def c_out(self, sigma):
        raise NotImplementedError

    def c_in(self, sigma):
        raise NotImplementedError

    def c_noise(self, sigma):
        raise NotImplementedError

    def loss_weighting(self, sigma):
        raise NotImplementedError

    def sample_noise_distribution(self, N):
        raise NotImplementedError

    def sample_scale_distribution(self, N):
        raise NotImplementedError

    def D(self, x, sigma, condition=None, use_ema=False):
        """ Prepositioning in EDM """
        c_skip, c_out, c_in, c_noise = self.c_skip(sigma), self.c_out(sigma), self.c_in(sigma), self.c_noise(sigma)
        F = self.model_ema["diffusion"] if use_ema else self.model["diffusion"]
        c_noise = at_least_ndim(c_noise.squeeze(), 1)
        return c_skip * x + c_out * F(c_in * x, c_noise, condition)

    # ---------------------------------------------------------------------------
    # Training

    def loss(self, x0, condition=None):
        sigma = self.sample_noise_distribution(x0.shape[0])
        sigma = at_least_ndim(sigma, x0.dim())
        eps = torch.randn_like(x0) * sigma * (1. - self.fix_mask)
        condition = self.model["condition"](condition) if condition is not None else None
        loss = (self.loss_weighting(sigma) * (self.D(x0 + eps, sigma, condition) - x0) ** 2)
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
        sigma = self.sample_noise_distribution(x0.shape[0])
        sigma = at_least_ndim(sigma, x0.dim())
        noise = self.c_noise(sigma)
        eps = torch.randn_like(x0) * sigma * (1. - self.fix_mask)
        log = self.classifier.update(x0 + eps, at_least_ndim(noise.squeeze(), 1), condition)
        return log

    # ---------------------------------------------------------------------------
    # Inference

    def dot_x(
            self, x, i, use_ema=False,
            # ----------------- CFG ----------------- #
            condition_vec_cfg=None,
            w_cfg: float = 0.0,
            # ----------------- CG ----------------- #
            condition_vec_cg=None,
            w_cg: float = 1.0,
    ):
        b = x.shape[0]
        sigma = at_least_ndim(self.sigma_s[i].repeat(b), x.dim())
        noise = self.c_noise(sigma)
        unscale = 1. / self.scale_s[i] * (1. - self.fix_mask) + self.fix_mask
        # ----------------- CFG ----------------- #
        with torch.no_grad():
            if w_cfg != 0.0 and w_cfg != 1.0:
                repeat_dim = [2 if i == 0 else 1 for i in range(x.dim())]
                condition_vec_cfg = torch.cat([condition_vec_cfg, torch.zeros_like(condition_vec_cfg)], 0)
                D = self.D(
                    (x * unscale).repeat(*repeat_dim),
                    sigma.repeat(*repeat_dim),
                    condition_vec_cfg, use_ema)
                D = w_cfg * D[:b] + (1. - w_cfg) * D[b:]
            elif w_cfg == 0.0:
                D = self.D(x * unscale, sigma, None, use_ema)
            else:
                D = self.D(x * unscale, sigma, condition_vec_cfg, use_ema)
        # ----------------- CG ----------------- #
        if self.classifier is not None and w_cg != 0.0 and condition_vec_cg is not None:
            log_p, grad = self.classifier.gradients(x * unscale,
                                                    at_least_ndim(noise.squeeze(), 1), condition_vec_cg)
            D = D + w_cg * self.scale_s[i] * (sigma ** 2) * grad
        else:
            log_p = None

        # do not change the fixed portion
        dot_x = self.x_weight_s[i] * x - self.D_weight_s[i] * D
        dot_x = dot_x * (1. - self.fix_mask)

        return dot_x, {"log_p": log_p}

    def sample(
            self,
            # ---------- the known fixed portion ---------- #
            prior: Optional[torch.Tensor] = None,
            # ----------------- sampling ----------------- #
            n_samples: int = 1,
            sample_steps: int = 5,
            use_ema: bool = True,
            solver: str = "euler",
            # ------------------ guidance ------------------ #
            condition_cfg=None,
            mask_cfg=None,
            w_cfg: float = 0.0,
            condition_cg=None,
            w_cg: float = 0.0,

            preserve_history: bool = False,
            **kwargs
    ):
        """
        Sample from the diffusion model.
        ---
        Input:
            - prior: Optional[torch.Tensor] = None
                The known fixed portion of the input data. Should be in the shape of `(n_samples, *x_shape)`.
                Leave the unknown part as `0`. If `None`, which means `fix_mask` is `None`, the model takes no prior.

            - sample_steps: int = 5
                Number of sampling steps.

            - use_ema: bool = True
                Whether to use the EMA model. If `False`, you should `eval` the model.

            - solver: str = "euler"
                The solver to use. Can be either "euler" or "heun".

            - condition_cfg: Optional[torch.Tensor] = None
                The condition for the CFG. Should be in the shape of `(n_samples, *shape_of_nn_condition_input)`.
                If `None`, the model takes no condition.

            - mask_cfg: Optional[torch.Tensor] = None
                The mask for the CFG. Should be in the shape of `(n_samples, *shape_of_nn_condition_mask)`.
                Model will ignore the `mask_cfg==0` parts in `condition_cfg`. If `None`, the model takes no mask.

            - w_cfg: float = 0.0
                The weight for the CFG. If `0.0`, the model takes no CFG.

            - condition_cg: Optional[torch.Tensor] = None
                The condition for the CG. Should be in the shape of `(n_samples, 1)`.
                If `None`, the model takes no condition.

            - w_cg: float = 0.0
                The weight for the CG. If `0.0`, the model takes no CG.

            - preserve_history: bool = False
                Whether to preserve the history of the sampling process. If `True`, the model will return the history.

        Output:
            - xt: torch.Tensor
                The sampled data. Should be in the shape of `(n_samples, *x_shape)`.

            - log: dict
                The log of the sampling process. Contains the following keys:
                    - "sample_history": np.ndarray
                        The history of the sampling process. Should be in the shape of `(n_samples, N + 1, *x_shape)`.
                        If `preserve_history` is `False`, this key will not exist.
                    - "log_p": torch.Tensor
                        The log probability of the sampled data estimated by CG.
                        Should be in the shape of `(n_samples,)`.
        """
        if sample_steps != self.sample_steps:
            self.set_sample_steps(sample_steps)

        N = self.sample_steps
        model = self.model_ema if use_ema else self.model
        log, x_history = {}, None

        condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None

        if prior is None:
            xt = torch.randn((n_samples, *self.default_x_shape),
                             device=self.device) * self.sigma_s[0] * self.scale_s[0]
        else:
            xt = torch.randn_like(prior, device=self.device) * self.sigma_s[0] * self.scale_s[0]
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask

        if preserve_history:
            x_history = np.empty((n_samples, N + 1, *xt.shape))
            x_history[:, 0] = xt.cpu().numpy()

        for i in range(N):
            # tmp_w_cg = w_cg if i < 1 else 0.0
            delta_x, log = self.dot_x(xt, i, use_ema, condition_vec_cfg, w_cfg, condition_cg, w_cg)
            delta_t = self.t_s[i] - self.t_s[i + 1]
            x_tp1 = xt - delta_x * delta_t
            if prior is not None:
                x_tp1 = x_tp1 * (1. - self.fix_mask) + prior * self.fix_mask
            if solver == "heun":
                if i != N - 1 and self.sigma_s[i + 1] > 0.005:
                    delta_x_2, log = self.dot_x(x_tp1, i + 1, use_ema, condition_vec_cfg, w_cfg, condition_cg, w_cg)
                    x_tp1 = xt - (delta_x + delta_x_2) / 2. * delta_t
                    if prior is not None:
                        x_tp1 = x_tp1 * (1. - self.fix_mask) + prior * self.fix_mask

            xt = x_tp1

            if preserve_history:
                x_history[:, i + 1] = xt.cpu().numpy()

        log["sample_history"] = x_history
        if log["log_p"] is None and self.classifier is not None and condition_cg is not None:
            with torch.no_grad():
                logp = self.classifier.logp(
                    xt, at_least_ndim(self.c_noise(self.sigma_s[-1]).squeeze(), 1), condition_cg)
            log["log_p"] = logp

        return xt, log

    def sample_x(
            self,
            # ---------- the known fixed portion ---------- #
            prior: Optional[torch.Tensor] = None,
            # ----------------- sampling ----------------- #
            n_samples: int = 1,
            sample_steps: int = 5,
            extra_sample_steps: int = 8,
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
        if sample_steps != self.sample_steps:
            self.set_sample_steps(sample_steps)

        N = self.sample_steps
        model = self.model_ema if use_ema else self.model
        log, x_history = {}, None

        condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None

        if prior is None:
            xt = torch.randn((n_samples, *self.default_x_shape),
                             device=self.device) * self.sigma_s[0] * self.scale_s[0]
        else:
            xt = torch.randn_like(prior, device=self.device) * self.sigma_s[0] * self.scale_s[0]
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask

        if preserve_history:
            x_history = np.empty((n_samples, N + 1, *xt.shape))
            x_history[:, 0] = xt.cpu().numpy()

        for i in range(N):
            # tmp_w_cg = w_cg if i < 1 else 0.0
            delta_x, log = self.dot_x(xt, i, use_ema, condition_vec_cfg, w_cfg, condition_cg, w_cg)
            delta_t = self.t_s[i] - self.t_s[i + 1]
            x_tp1 = xt - delta_x * delta_t
            if prior is not None:
                x_tp1 = x_tp1 * (1. - self.fix_mask) + prior * self.fix_mask
            if solver == "heun":
                if i != N - 1 and self.sigma_s[i + 1] > 0.005:
                    delta_x_2, log = self.dot_x(x_tp1, i + 1, use_ema, condition_vec_cfg, w_cfg, condition_cg, w_cg)
                    x_tp1 = xt - (delta_x + delta_x_2) / 2. * delta_t
                    if prior is not None:
                        x_tp1 = x_tp1 * (1. - self.fix_mask) + prior * self.fix_mask

            xt = x_tp1

            if preserve_history:
                x_history[:, i + 1] = xt.cpu().numpy()

        if extra_sample_steps > 0:

            delta_t = self.t_s[N - 1] - self.t_s[N]

            for _ in range(extra_sample_steps):

                delta_x, log = self.dot_x(xt, N - 1, use_ema, condition_vec_cfg, w_cfg, condition_cg, w_cg)
                x_tp1 = xt - delta_x * delta_t
                if prior is not None:
                    x_tp1 = x_tp1 * (1. - self.fix_mask) + prior * self.fix_mask

                xt = x_tp1

        log["sample_history"] = x_history
        if log["log_p"] is None and self.classifier is not None and condition_cg is not None:
            with torch.no_grad():
                logp = self.classifier.logp(
                    xt, at_least_ndim(self.c_noise(self.sigma_s[-1]).squeeze(), 1), condition_cg)
            log["log_p"] = logp

        return xt, log


class EDM(EDMArchetecture):
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
            sigma_data: float = 0.5,
            sigma_min: float = 0.002,
            sigma_max: float = 80.,
            rho: float = 7.,
            P_mean: float = -1.2,
            P_std: float = 1.2,

            device: Union[torch.device, str] = "cpu"
    ):
        super().__init__(
            nn_diffusion, nn_condition, fix_mask, loss_weight, classifier, grad_clip_norm,
            diffusion_steps, ema_rate, optim_params, device)

        self.sigma_data = sigma_data
        self.sigma_min, self.sigma_max, self.rho = sigma_min, sigma_max, rho
        self.P_mean, self.P_std = P_mean, P_std

    def set_sample_steps(self, N: int):
        self.sample_steps = N
        self.sigma_s = (self.sigma_max ** (1 / self.rho) + torch.arange(N + 1, device=self.device) / N *
                        (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
        self.t_s = self.sigma_s
        self.scale_s = torch.ones_like(self.sigma_s)
        self.dot_sigma_s = torch.ones_like(self.sigma_s)
        self.dot_scale_s = torch.zeros_like(self.sigma_s)
        self.x_weight_s = (self.dot_sigma_s / self.sigma_s + self.dot_scale_s / self.scale_s)
        self.D_weight_s = self.dot_sigma_s / self.sigma_s * self.scale_s

    def c_skip(self, sigma): return self.sigma_data ** 2 / (self.sigma_data ** 2 + sigma ** 2)

    def c_out(self, sigma): return sigma * self.sigma_data / (self.sigma_data ** 2 + sigma ** 2).sqrt()

    def c_in(self, sigma): return 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()

    def c_noise(self, sigma): return 0.25 * sigma.log()

    def loss_weighting(self, sigma): return (self.sigma_data ** 2 + sigma ** 2) / ((sigma * self.sigma_data) ** 2)

    def sample_noise_distribution(self, N):
        log_sigma = torch.randn(N, device=self.device) * self.P_std + self.P_mean
        return log_sigma.exp()

    def sample_scale_distribution(self, N):
        return torch.ones(N, device=self.device)
