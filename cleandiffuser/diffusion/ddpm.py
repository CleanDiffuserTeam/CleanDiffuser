from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from cleandiffuser.classifier import BaseClassifier
from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import (
    at_least_ndim,
    cosine_beta_schedule,
    linear_beta_schedule)
from .basic import DiffusionModel


class DDPM(DiffusionModel):

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
            x_max: Optional[torch.Tensor] = None,
            x_min: Optional[torch.Tensor] = None,

            # ------------------- DPM Params ------------------- #
            predict_noise: bool = True,
            beta_schedule: str = "cosine",  # or cosine
            beta_schedule_params: Optional[dict] = None,

            device: Union[torch.device, str] = "cpu"
    ):
        super().__init__(
            nn_diffusion, nn_condition, fix_mask, loss_weight, classifier, grad_clip_norm,
            diffusion_steps, ema_rate, optim_params, device)

        self.predict_noise = predict_noise

        if beta_schedule_params is None:
            beta_schedule_params = {}
        beta_schedule_params["T"] = self.diffusion_steps

        if beta_schedule == "linear":
            beta = linear_beta_schedule(**beta_schedule_params)
        elif beta_schedule == "cosine":
            beta = cosine_beta_schedule(**beta_schedule_params)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.beta = torch.tensor(beta, device=self.device, dtype=torch.float32)
        self.alpha = 1 - self.beta
        self.bar_alpha = torch.cumprod(self.alpha.clone(), 0)
        self.x_max, self.x_min = x_max, x_min

    @property
    def clip_pred(self):
        return (self.x_max is not None) or (self.x_min is not None)

    # ---------------------------------------------------------------------------
    # Training

    def add_noise(self, x0, t=None, eps=None):
        t = torch.randint(self.diffusion_steps, (x0.shape[0],), device=self.device) if t is None else t
        eps = torch.randn_like(x0) if eps is None else eps
        bar_alpha = at_least_ndim(self.bar_alpha[t], x0.dim())
        xt = x0 * bar_alpha.sqrt() + eps * (1 - bar_alpha).sqrt()
        xt = xt * (1. - self.fix_mask) + x0 * self.fix_mask
        return xt, t, eps

    def loss(self, x0, condition=None):
        xt, t, eps = self.add_noise(x0)
        condition = self.model["condition"](condition) if condition is not None else None
        if self.predict_noise:
            loss = (self.model["diffusion"](xt, t, condition) - eps) ** 2
        else:
            loss = (self.model["diffusion"](xt, t, condition) - x0) ** 2

        return (loss * self.loss_weight * (1 - self.fix_mask)).mean()

    def update(self, x0, condition=None, update_ema=True, **kwargs):
        loss = self.loss(x0, condition)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm) \
            if self.grad_clip_norm else None
        self.optimizer.step()
        self.optimizer.zero_grad()
        if update_ema: self.ema_update()
        log = {"loss": loss.item(), "grad_norm": grad_norm}
        return log

    def update_classifier(self, x0, condition):
        xt, t, eps = self.add_noise(x0)
        log = self.classifier.update(xt, t, condition)
        return log

    # ---------------------------------------------------------------------------
    # Inference

    def predict_function(
            self, x, t, bar_alpha,
            use_ema=False, requires_grad=False,
            # ----------------- CFG ----------------- #
            condition_vec_cfg=None,
            w_cfg: float = 0.0,
            # ----------------- CG ----------------- #
            condition_vec_cg=None,
            w_cg: float = 1.0,
    ):
        b = x.shape[0]
        model = self.model_ema if use_ema else self.model

        # ----------------- CFG ----------------- #
        with torch.set_grad_enabled(requires_grad):
            if w_cfg != 0.0 and w_cfg != 1.0:
                repeat_dim = [2 if i == 0 else 1 for i in range(x.dim())]
                condition_vec_cfg = torch.cat([condition_vec_cfg, torch.zeros_like(condition_vec_cfg)], 0)
                pred = model["diffusion"](
                    x.repeat(*repeat_dim), t.repeat(2), condition_vec_cfg)
                pred = w_cfg * pred[:b] + (1. - w_cfg) * pred[b:]
            elif w_cfg == 0.0:
                pred = model["diffusion"](x, t, None)
            else:
                pred = model["diffusion"](x, t, condition_vec_cfg)

        # ----------------- CG ----------------- #
        if self.classifier is not None and w_cg != 0.0 and condition_vec_cg is not None:
            log_p, grad = self.classifier.gradients(x.clone(), t, condition_vec_cg)
            if self.predict_noise:
                pred = pred - w_cg * (1 - bar_alpha).sqrt() * grad
            else:
                pred = pred + w_cg * (1 - bar_alpha) / bar_alpha.sqrt() * grad
        else:
            log_p = None

        if self.predict_noise:
            if self.clip_pred:
                upper_bound = (x - bar_alpha.sqrt() * self.x_min) / (1 - bar_alpha).sqrt() \
                    if self.x_min is not None else None
                lower_bound = (x - bar_alpha.sqrt() * self.x_max) / (1 - bar_alpha).sqrt() \
                    if self.x_max is not None else None
                pred = pred.clip(lower_bound, upper_bound)
            pred = pred * (1 - self.fix_mask)
        else:
            if self.clip_pred:
                pred = pred.clip(self.x_min, self.x_max)
            pred = pred * (1 - self.fix_mask) + x * self.fix_mask

        return pred, {"log_p": log_p}

    def sample(
            self,
            # ---------- the known fixed portion ---------- #
            prior: Optional[torch.Tensor] = None,
            # ----------------- sampling ----------------- #
            n_samples: int = 1,
            sample_steps: int = None,
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

        # check `sample_steps`
        if sample_steps != self.diffusion_steps:
            import warnings
            warnings.warn(f"sample_steps != diffusion_steps, sample_steps will be set to diffusion_steps.")
            sample_steps = self.diffusion_steps

        # initialize the samples
        xt = torch.randn_like(prior, device=self.device) * temperature
        xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
        if preserve_history: log["sample_history"][:, 0] = xt.cpu().numpy()

        # preprocess the conditions
        with torch.set_grad_enabled(requires_grad):
            condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
            condition_vec_cg = condition_cg

        # enter the sampling loop
        for t in range(self.diffusion_steps - 1, -1, -1):

            t_batch = torch.tensor(t, device=self.device, dtype=torch.long).repeat(n_samples)
            bar_alpha = self.bar_alpha[t]
            bar_alpha_prev = self.bar_alpha[t - 1] if t > 0 else torch.tensor(1.0, device=self.device)
            alpha = self.alpha[t]
            beta = self.beta[t]

            # predict eps_theta or x_theta with CG/CFG
            pred_theta, log = self.predict_function(
                xt, t_batch, bar_alpha,
                use_ema=use_ema,
                requires_grad=requires_grad,
                condition_vec_cfg=condition_vec_cfg,
                condition_vec_cg=condition_vec_cg,
                w_cfg=w_cfg, w_cg=w_cg)

            # one step denoise
            if self.predict_noise:

                xt = 1 / alpha.sqrt() * (xt - beta / (1 - bar_alpha).sqrt() * pred_theta)

            else:

                xt = 1 / (1 - bar_alpha) * (
                    alpha.sqrt() * (1 - bar_alpha_prev) * xt +
                    beta * bar_alpha_prev.sqrt() * pred_theta)

            if t != 0:
                xt = xt + (beta * (1 - bar_alpha_prev) / (1 - bar_alpha)).sqrt() * torch.randn_like(xt)

            # Fix the known portion
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
            if preserve_history: log["sample_history"][:, 1] = xt.cpu().numpy()

        # calculate the final log_p
        if log["log_p"] is None and self.classifier is not None and condition_cg is not None:
            with torch.no_grad():
                logp = self.classifier.logp(xt, t[-1].repeat(n_samples), condition_vec_cg)
            log["log_p"] = logp

        return xt, log


    def sample_x(
            self,
            # ---------- the known fixed portion ---------- #
            prior: Optional[torch.Tensor] = None,
            # ----------------- sampling ----------------- #
            n_samples: int = 1,
            sample_steps: int = None,
            extra_sample_steps: int = 8,
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

        # check `sample_steps`
        if sample_steps != self.diffusion_steps:
            import warnings
            warnings.warn(f"sample_steps != diffusion_steps, sample_steps will be set to diffusion_steps.")
            sample_steps = self.diffusion_steps

        # initialize the samples
        xt = torch.randn_like(prior, device=self.device) * temperature
        xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
        if preserve_history: log["sample_history"][:, 0] = xt.cpu().numpy()

        # preprocess the conditions
        with torch.set_grad_enabled(requires_grad):
            condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
            condition_vec_cg = condition_cg

        # enter the sampling loop
        for t in range(self.diffusion_steps - 1, -1, -1):

            t_batch = torch.tensor(t, device=self.device, dtype=torch.long).repeat(n_samples)
            bar_alpha = self.bar_alpha[t]
            bar_alpha_prev = self.bar_alpha[t - 1] if t > 0 else torch.tensor(1.0, device=self.device)
            alpha = self.alpha[t]
            beta = self.beta[t]

            # predict eps_theta or x_theta with CG/CFG
            pred_theta, log = self.predict_function(
                xt, t_batch, bar_alpha,
                use_ema=use_ema,
                requires_grad=requires_grad,
                condition_vec_cfg=condition_vec_cfg,
                condition_vec_cg=condition_vec_cg,
                w_cfg=w_cfg, w_cg=w_cg)

            # one step denoise
            if self.predict_noise:

                xt = 1 / alpha.sqrt() * (xt - beta / (1 - bar_alpha).sqrt() * pred_theta)

            else:

                xt = 1 / (1 - bar_alpha) * (
                    alpha.sqrt() * (1 - bar_alpha_prev) * xt +
                    beta * bar_alpha_prev.sqrt() * pred_theta)

            if t != 0:
                xt = xt + (beta * (1 - bar_alpha_prev) / (1 - bar_alpha)).sqrt() * torch.randn_like(xt)

            # Fix the known portion
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
            if preserve_history: log["sample_history"][:, 1] = xt.cpu().numpy()

        if extra_sample_steps > 0:

            t_batch = torch.tensor(0, device=self.device, dtype=torch.long).repeat(n_samples)
            bar_alpha = self.bar_alpha[0]
            bar_alpha_prev = torch.tensor(1.0, device=self.device)
            alpha = self.alpha[0]
            beta = self.beta[0]

            for _ in range(extra_sample_steps):

                # predict eps_theta or x_theta with CG/CFG
                pred_theta, log = self.predict_function(
                    xt, t_batch, bar_alpha,
                    use_ema=use_ema,
                    requires_grad=requires_grad,
                    condition_vec_cfg=condition_vec_cfg,
                    condition_vec_cg=condition_vec_cg,
                    w_cfg=w_cfg, w_cg=w_cg)

                # one step denoise
                if self.predict_noise:

                    xt = 1 / alpha.sqrt() * (xt - beta / (1 - bar_alpha).sqrt() * pred_theta)

                else:

                    xt = 1 / (1 - bar_alpha) * (
                            alpha.sqrt() * (1 - bar_alpha_prev) * xt +
                            beta * bar_alpha_prev.sqrt() * pred_theta)

                # Fix the known portion
                xt = xt * (1. - self.fix_mask) + prior * self.fix_mask

        # calculate the final log_p
        if log["log_p"] is None and self.classifier is not None and condition_cg is not None:
            with torch.no_grad():
                logp = self.classifier.logp(xt, t[-1].repeat(n_samples), condition_vec_cg)
            log["log_p"] = logp

        return xt, log
