from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from cleandiffuser.classifier import BaseClassifier
from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import at_least_ndim
from .basic import DiffusionModel

SAMPLER_CONFIG = {

    "ode_dpm_1": {
        "predict_noise": True,
        "order": 1, },

    "ddim": {
        "predict_noise": True,
        "order": 1, },

    "sde_dpm_1": {
        "predict_noise": True,
        "order": 1, },

    "ode_dpmpp_1": {
        "predict_noise": False,
        "order": 1, },

    "sde_dpmpp_1": {
        "predict_noise": False,
        "order": 1, },

    "ode_dpm_2": {
        "predict_noise": True,
        "order": 2, },

    "sde_dpm_2": {
        "predict_noise": True,
        "order": 2, },

    "ode_dpmpp_2": {
        "predict_noise": False,
        "order": 2, },

    "sde_dpmpp_2": {
        "predict_noise": False,
        "order": 2, }, }


def epstheta_to_xtheta(x, alpha, sigma, eps_theta):
    """
    x_theta = (x - sigma * eps_theta) / alpha
    """
    return (x - sigma * eps_theta) / alpha


def xtheta_to_epstheta(x, alpha, sigma, x_theta):
    """
    eps_theta = (x - alpha * x_theta) / sigma
    """
    return (x - alpha * x_theta) / sigma


def onestep_estimate(xt, pred_theta, i, alphas, sigmas, h, sampler):

    if sampler[:-2] == "ode_dpm" or sampler == "ddim":
        xt_next = (
                alphas[i] / alphas[i - 1] * xt -
                sigmas[i] * torch.expm1(h[i]) * pred_theta)
    elif sampler[:-2] == "sde_dpm":
        xt_next = (
                alphas[i] / alphas[i - 1] * xt -
                2. * sigmas[i] * torch.expm1(h[i]) * pred_theta +
                sigmas[i] * torch.expm1(2. * h[i]).sqrt() * torch.randn_like(xt))
    elif sampler[:-2] == "ode_dpmpp":
        xt_next = (
                sigmas[i] / sigmas[i - 1] * xt -
                alphas[i] * torch.expm1(-h[i]) * pred_theta)
    elif sampler[:-2] == "sde_dpmpp":
        xt_next = (
                sigmas[i] / sigmas[i - 1] * (-h[i]).exp() * xt -
                alphas[i] * torch.expm1(-2. * h[i]) * pred_theta +
                sigmas[i] * (-1. * torch.expm1(-2. * h[i])).sqrt() * torch.randn_like(xt))
    else:
        raise ValueError(f'Unknown sampler: {sampler}.')

    return xt_next


class DPMSolver(DiffusionModel):

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
            predict_noise: bool = False,
            noise_schedule: str = "linear",  # or cosine
            t_eps: float = 1e-3,

            device: Union[torch.device, str] = "cpu"
    ):
        super().__init__(
            nn_diffusion, nn_condition, fix_mask, loss_weight, classifier, grad_clip_norm,
            diffusion_steps, ema_rate, optim_params, device)

        self.predict_noise = predict_noise
        self.noise_schedule = noise_schedule
        self.t_eps = t_eps
        self.x_max, self.x_min = x_max, x_min

    @property
    def clip_pred(self):
        return (self.x_max is not None) or (self.x_min is not None)

    @property
    def supported_samplers(self):
        return list(SAMPLER_CONFIG.keys())

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
            return (-(beta1 - beta0) / 4 * (t ** 2) - beta0 / 2 * t).exp()
        elif self.noise_schedule == "cosine":
            s = 0.008
            return ((torch.cos(np.pi / 2 * (t + s) / (1 + s))).log() - np.log(np.cos(np.pi / 2 * s / (1 + s)))).exp()
        else:
            raise ValueError(f"noise_schedule should be 'linear' or 'cosine', but got {self.noise_schedule}.")

    # ---------------------------------------------------------------------------
    # Training

    def add_noise(self, x0, t=None, eps=None):
        if t is None:
            t = torch.rand((x0.shape[0],), device=self.device)
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

        if self.predict_noise:
            loss = (self.model["diffusion"](xt, t, condition) - eps) ** 2
        else:
            loss = (self.model["diffusion"](xt, t, condition) - x0) ** 2

        return (loss * self.loss_weight * (1. - self.fix_mask)).mean()

    def update(self, x0, condition=None, update_ema=True, **kwargs):
        self.optimizer.zero_grad()
        loss = self.loss(x0, condition)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm) \
            if self.grad_clip_norm else None
        self.optimizer.step()
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
            self, x, t, alpha, sigma,
            use_ema=False, requires_grad=False, predict_noise=False,
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

        if self.predict_noise and predict_noise:
            # The network predicts noise, and we need to predict noise here
            pass
        elif self.predict_noise and not predict_noise:
            # The network predicts noise, and we need to predict x0 here
            pred = epstheta_to_xtheta(x, alpha, sigma, pred)
        elif not self.predict_noise and predict_noise:
            # The network predicts x0, and we need to predict noise here
            pred = xtheta_to_epstheta(x, alpha, sigma, pred)
        else:
            # The network predicts x0, and we need to predict x0 here
            pass

        # ----------------- CG ----------------- #
        if self.classifier is not None and w_cg != 0.0 and condition_vec_cg is not None:
            log_p, grad = self.classifier.gradients(x.clone(), t, condition_vec_cg)
            if predict_noise:
                pred = pred - w_cg * sigma * grad
            else:
                pred = pred + w_cg * ((sigma ** 2) / alpha) * grad
        else:
            log_p = None

        # If the bounds are not None, clip `pred`
        if predict_noise:
            if self.clip_pred:
                upper_bound = (x - alpha * self.x_min) / sigma if self.x_min is not None else None
                lower_bound = (x - alpha * self.x_max) / sigma if self.x_max is not None else None
                pred = pred.clip(lower_bound, upper_bound)
            pred = pred * (1 - self.fix_mask)
        else:
            if self.clip_pred:
                pred = pred.clip(self.x_min, self.x_max)
            pred = pred * (1 - self.fix_mask) + x * self.fix_mask

        return pred, {"log_p": log_p}

    # def first_order_sample(
    #         self,
    #         sampler: str = "ode_dpm_1",
    #         # ---------- the known fixed portion ---------- #
    #         prior: Optional[torch.Tensor] = None,
    #         # ----------------- sampling ----------------- #
    #         n_samples: int = 1,
    #         sample_steps: int = None,
    #         use_ema: bool = True,
    #         kappa: float = 1.0,
    #         temperature: float = 1.0,
    #         # ------------------ guidance ------------------ #
    #         condition_cfg=None,
    #         mask_cfg=None,
    #         w_cfg: float = 0.0,
    #         condition_cg=None,
    #         w_cg: float = 0.0,
    #
    #         requires_grad: bool = False,
    #         preserve_history: bool = False,
    # ):
    #     log = {
    #         "sample_history": np.empty((n_samples, sample_steps + 1, *prior.shape)) if preserve_history else None, }
    #
    #     model = self.model_ema if use_ema else self.model
    #
    #     idx = torch.arange(sample_steps + 1)
    #     t = (((sample_steps - idx) / sample_steps * self.t_range[1] ** (1 / kappa) +
    #           idx / sample_steps * self.t_range[0] ** (1 / kappa)) ** kappa).to(self.device)
    #
    #     alphas = self.alpha_schedule(t)
    #     sigmas = (1 - alphas ** 2).sqrt()
    #     logSNRs = (alphas / sigmas).log()
    #     h = torch.zeros_like(logSNRs)
    #     h[1:] = logSNRs[1:] - logSNRs[:-1]
    #
    #     xt = torch.randn_like(prior, device=self.device) * temperature
    #     xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
    #     if preserve_history: log["sample_history"][:, 0] = xt.cpu().numpy()
    #
    #     with torch.set_grad_enabled(requires_grad):
    #         condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
    #         condition_vec_cg = condition_cg
    #
    #     for i in range(1, sample_steps + 1):
    #
    #         pred_theta, log = self.predict_function(
    #             xt, t[i - 1].repeat(n_samples), alphas[i - 1], sigmas[i - 1],
    #             use_ema=use_ema,
    #             requires_grad=requires_grad,
    #             predict_noise=SAMPLER_CONFIG[sampler]["predict_noise"],
    #             condition_vec_cfg=condition_vec_cfg,
    #             condition_vec_cg=condition_vec_cg,
    #             w_cfg=w_cfg, w_cg=w_cg)
    #
    #         xt = onestep_estimate(xt, pred_theta, i, alphas, sigmas, h, sampler)
    #         xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
    #         if preserve_history: log["sample_history"][:, 0] = xt.cpu().numpy()
    #
    #     if log["log_p"] is None and self.classifier is not None and condition_cg is not None:
    #         with torch.no_grad():
    #             logp = self.classifier.logp(xt, t[-1].repeat(n_samples), condition_vec_cg)
    #         log["log_p"] = logp
    #
    #     return xt, log
    #
    # def second_order_sample_multistep(
    #         self,
    #         sampler: str = "ode_dpmpp_2",
    #         # ---------- the known fixed portion ---------- #
    #         prior: Optional[torch.Tensor] = None,
    #         # ----------------- sampling ----------------- #
    #         n_samples: int = 1,
    #         sample_steps: int = None,
    #         use_ema: bool = True,
    #         kappa: float = 1.0,
    #         temperature: float = 1.0,
    #         # ------------------ guidance ------------------ #
    #         condition_cfg=None,
    #         mask_cfg=None,
    #         w_cfg: float = 0.0,
    #         condition_cg=None,
    #         w_cg: float = 0.0,
    #
    #         requires_grad: bool = False,
    #         preserve_history: bool = False,
    # ):
    #     log = {
    #         "sample_history": np.empty((n_samples, sample_steps + 1, *prior.shape)) if preserve_history else None, }
    #
    #     model = self.model_ema if use_ema else self.model
    #
    #     idx = torch.arange(sample_steps + 1)
    #     t = (((sample_steps - idx) / sample_steps * self.t_range[1] ** (1 / kappa) +
    #           idx / sample_steps * self.t_range[0] ** (1 / kappa)) ** kappa).to(self.device)
    #
    #     alphas = self.alpha_schedule(t)
    #     sigmas = (1 - alphas ** 2).sqrt()
    #     logSNRs = (alphas / sigmas).log()
    #     h = torch.zeros_like(logSNRs)
    #     h[1:] = logSNRs[1:] - logSNRs[:-1]
    #
    #     buffer = []
    #
    #     xt = torch.randn_like(prior, device=self.device) * temperature
    #     xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
    #     if preserve_history: log["sample_history"][:, 0] = xt.cpu().numpy()
    #
    #     with torch.set_grad_enabled(requires_grad):
    #         condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
    #         condition_vec_cg = condition_cg
    #
    #     # pre-update
    #
    #     pred_theta, log = self.predict_function(
    #         xt, t[0].repeat(n_samples), alphas[0], sigmas[0],
    #         use_ema=use_ema,
    #         requires_grad=requires_grad,
    #         predict_noise=SAMPLER_CONFIG[sampler]["predict_noise"],
    #         condition_vec_cfg=condition_vec_cfg,
    #         condition_vec_cg=condition_vec_cg,
    #         w_cfg=w_cfg, w_cg=w_cg)
    #     buffer.append(pred_theta.clone())
    #
    #     xt = onestep_estimate(xt, pred_theta, 1, alphas, sigmas, h, sampler)
    #     xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
    #     if preserve_history: log["sample_history"][:, 0] = xt.cpu().numpy()
    #
    #     pred_theta, log = self.predict_function(
    #         xt, t[1].repeat(n_samples), alphas[1], sigmas[1],
    #         use_ema=use_ema,
    #         requires_grad=requires_grad,
    #         predict_noise=SAMPLER_CONFIG[sampler]["predict_noise"],
    #         condition_vec_cfg=condition_vec_cfg,
    #         condition_vec_cg=condition_vec_cg,
    #         w_cfg=w_cfg, w_cg=w_cg)
    #     buffer.append(pred_theta.clone())
    #
    #     for i in range(2, sample_steps + 1):
    #
    #         r = h[i - 1] / h[i]
    #
    #         D = (1 + 0.5 / r) * buffer[-1] - 0.5 / r * buffer[-2]
    #
    #         xt = onestep_estimate(xt, D, i, alphas, sigmas, h, sampler)
    #         xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
    #         if preserve_history: log["sample_history"][:, 1] = xt.cpu().numpy()
    #
    #         if i < sample_steps + 1:
    #             pred_theta, log = self.predict_function(
    #                 xt, t[i].repeat(n_samples), alphas[i], sigmas[i],
    #                 use_ema=use_ema,
    #                 requires_grad=requires_grad,
    #                 predict_noise=SAMPLER_CONFIG[sampler]["predict_noise"],
    #                 condition_vec_cfg=condition_vec_cfg,
    #                 condition_vec_cg=condition_vec_cg,
    #                 w_cfg=w_cfg, w_cg=w_cg)
    #             buffer.append(pred_theta.clone())
    #
    #     if log["log_p"] is None and self.classifier is not None and condition_cg is not None:
    #         with torch.no_grad():
    #             logp = self.classifier.logp(xt, t[-1].repeat(n_samples), condition_vec_cg)
    #         log["log_p"] = logp
    #
    #     return xt, log

    def sample(
            self,
            # ---------- the known fixed portion ---------- #
            prior: Optional[torch.Tensor] = None,
            # ----------------- sampling ----------------- #
            n_samples: int = 1,
            sample_steps: int = 5,
            use_ema: bool = True,
            temperature: float = 1.0,
            kappa: float = 1.0,
            sampler: str = "ddim",
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
        assert sampler in self.supported_samplers, f"Sampler '{sampler}' is not supported."

        # initialize logger
        log = {
            "sample_history": np.empty((n_samples, sample_steps + 1, *prior.shape)) if preserve_history else None, }

        # choose the model
        model = self.model_ema if use_ema else self.model

        # schedule the sampling steps
        idx = torch.arange(sample_steps + 1)
        t = (((sample_steps - idx) / sample_steps * self.t_range[1] ** (1 / kappa) +
              idx / sample_steps * self.t_range[0] ** (1 / kappa)) ** kappa).to(self.device)

        alphas = self.alpha_schedule(t)
        sigmas = (1 - alphas ** 2).sqrt()
        logSNRs = (alphas / sigmas).log()
        h = torch.zeros_like(logSNRs)
        h[1:] = logSNRs[1:] - logSNRs[:-1]

        buffer = []  # only for 2nd order samplers

        # initialize the samples
        xt = torch.randn_like(prior, device=self.device) * temperature
        xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
        if preserve_history: log["sample_history"][:, 0] = xt.cpu().numpy()

        # preprocess the conditions
        with torch.set_grad_enabled(requires_grad):
            condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
            condition_vec_cg = condition_cg

        # enter the sampling loop
        for i in range(1, sample_steps + 1):

            # predict eps_theta or x_theta with CG/CFG
            pred_theta, log = self.predict_function(
                xt, t[i - 1].repeat(n_samples), alphas[i - 1], sigmas[i - 1],
                use_ema=use_ema,
                requires_grad=requires_grad,
                predict_noise=SAMPLER_CONFIG[sampler]["predict_noise"],
                condition_vec_cfg=condition_vec_cfg,
                condition_vec_cg=condition_vec_cg,
                w_cfg=w_cfg, w_cg=w_cg)
            if SAMPLER_CONFIG[sampler]["order"] == 2: buffer.append(pred_theta.clone())

            # one step denoise
            if SAMPLER_CONFIG[sampler]["order"] == 2 and i > 1:

                r = h[i - 1] / h[i]
                D = (1 + 0.5 / r) * buffer[-1] - 0.5 / r * buffer[-2]

                xt = onestep_estimate(xt, D, i, alphas, sigmas, h, sampler)

            else:

                xt = onestep_estimate(xt, pred_theta, i, alphas, sigmas, h, sampler)

            # fix the known portion, and preserve the sampling history
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
            if preserve_history: log["sample_history"][:, 1] = xt.cpu().numpy()

        # calculate the final log_p
        if log["log_p"] is None and self.classifier is not None and condition_cg is not None:
            with torch.no_grad():
                logp = self.classifier.logp(xt, t[-1].repeat(n_samples), condition_vec_cg)
            log["log_p"] = logp

        # clip xt if bound is not None
        if self.clip_pred:
            xt = xt.clip(self.x_min, self.x_max)

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
            temperature: float = 1.0,
            kappa: float = 1.0,
            sampler: str = "ddim",
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
        assert sampler in self.supported_samplers, f"Sampler '{sampler}' is not supported."

        # initialize logger
        log = {
            "sample_history": np.empty((n_samples, sample_steps + 1, *prior.shape)) if preserve_history else None, }

        # choose the model
        model = self.model_ema if use_ema else self.model

        # schedule the sampling steps
        idx = torch.arange(sample_steps + 1)
        t = (((sample_steps - idx) / sample_steps * self.t_range[1] ** (1 / kappa) +
              idx / sample_steps * self.t_range[0] ** (1 / kappa)) ** kappa).to(self.device)

        alphas = self.alpha_schedule(t)
        sigmas = (1 - alphas ** 2).sqrt()
        logSNRs = (alphas / sigmas).log()
        h = torch.zeros_like(logSNRs)
        h[1:] = logSNRs[1:] - logSNRs[:-1]

        buffer = []  # only for 2nd order samplers

        # initialize the samples
        xt = torch.randn_like(prior, device=self.device) * temperature
        xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
        if preserve_history: log["sample_history"][:, 0] = xt.cpu().numpy()

        # preprocess the conditions
        with torch.set_grad_enabled(requires_grad):
            condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
            condition_vec_cg = condition_cg

        # enter the sampling loop
        for i in range(1, sample_steps + 1):

            # predict eps_theta or x_theta with CG/CFG
            pred_theta, log = self.predict_function(
                xt, t[i - 1].repeat(n_samples), alphas[i - 1], sigmas[i - 1],
                use_ema=use_ema,
                requires_grad=requires_grad,
                predict_noise=SAMPLER_CONFIG[sampler]["predict_noise"],
                condition_vec_cfg=condition_vec_cfg,
                condition_vec_cg=condition_vec_cg,
                w_cfg=w_cfg, w_cg=w_cg)
            if SAMPLER_CONFIG[sampler]["order"] == 2: buffer.append(pred_theta.clone())

            # one step denoise
            if SAMPLER_CONFIG[sampler]["order"] == 2 and i > 1:

                r = h[i - 1] / h[i]
                D = (1 + 0.5 / r) * buffer[-1] - 0.5 / r * buffer[-2]

                xt = onestep_estimate(xt, D, i, alphas, sigmas, h, sampler)

            else:

                xt = onestep_estimate(xt, pred_theta, i, alphas, sigmas, h, sampler)

            # fix the known portion, and preserve the sampling history
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
            if preserve_history: log["sample_history"][:, 1] = xt.cpu().numpy()  # TODO: BUG logging

        # diffusion-X extra sampling steps, only for 1st order solvers
        if SAMPLER_CONFIG[sampler]["order"] == 1 and extra_sample_steps > 0:
            for i in range(extra_sample_steps):

                # predict eps_theta or x_theta with CG/CFG
                pred_theta, log = self.predict_function(
                    xt, t[sample_steps - 1].repeat(n_samples), alphas[sample_steps - 1], sigmas[sample_steps - 1],
                    use_ema=use_ema,
                    requires_grad=requires_grad,
                    predict_noise=SAMPLER_CONFIG[sampler]["predict_noise"],
                    condition_vec_cfg=condition_vec_cfg,
                    condition_vec_cg=condition_vec_cg,
                    w_cfg=w_cfg, w_cg=w_cg)

                xt = onestep_estimate(xt, pred_theta, sample_steps, alphas, sigmas, h, sampler)

                # fix the known portion, and preserve the sampling history
                xt = xt * (1. - self.fix_mask) + prior * self.fix_mask

        # calculate the final log_p
        if log["log_p"] is None and self.classifier is not None and condition_cg is not None:
            with torch.no_grad():
                logp = self.classifier.logp(xt, t[-1].repeat(n_samples), condition_vec_cg)
            log["log_p"] = logp

        # clip xt if bound is not None
        if self.clip_pred:
            xt = xt.clip(self.x_min, self.x_max)

        return xt, log



    # def sample_with_ode_dpm_1(
    #         self,
    #         # ---------- the known fixed portion ---------- #
    #         prior: Optional[torch.Tensor] = None,
    #         # ----------------- sampling ----------------- #
    #         n_samples: int = 1,
    #         sample_steps: int = None,
    #         use_ema: bool = True,
    #         kappa: float = 1.0,
    #         temperature: float = 1.0,
    #         # ------------------ guidance ------------------ #
    #         condition_cfg=None,
    #         mask_cfg=None,
    #         w_cfg: float = 0.0,
    #         condition_cg=None,
    #         w_cg: float = 0.0,
    #
    #         preserve_history: bool = False,
    # ):
    #     log = {
    #         "sample_history": np.empty((n_samples, sample_steps + 1, *prior.shape)) if preserve_history else None, }
    #
    #     model = self.model_ema if use_ema else self.model
    #
    #     idx = torch.arange(sample_steps + 1)
    #     t = (((sample_steps - idx) / sample_steps * self.t_range[1] ** (1 / kappa) +
    #           idx / sample_steps * self.t_range[0] ** (1 / kappa)) ** kappa).to(self.device)
    #
    #     alphas = self.alpha_schedule(t)
    #     sigmas = (1 - alphas ** 2).sqrt()
    #     logSNRs = (alphas / sigmas).log()
    #     h = torch.zeros_like(logSNRs)
    #     h[1:] = logSNRs[1:] - logSNRs[:-1]
    #
    #     xt = torch.randn_like(prior, device=self.device) * temperature
    #     xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
    #     if preserve_history: log["sample_history"][:, 0] = xt.cpu().numpy()
    #
    #     with torch.no_grad():
    #         condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
    #         condition_vec_cg = condition_cg
    #
    #     for i in range(1, sample_steps + 1):
    #
    #         eps_theta, log = self.predict_eps_theta(
    #             xt, t[i - 1].repeat(n_samples), alphas[i - 1], sigmas[i - 1],
    #             use_ema, condition_vec_cfg, w_cfg, condition_vec_cg, w_cg)
    #
    #         xt = alphas[i] / alphas[i - 1] * xt - sigmas[i] * torch.expm1(h[i]) * eps_theta
    #         xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
    #
    #         if preserve_history:
    #             x_history[:, i] = xt.cpu().numpy()
    #
    #     log["sample_history"] = x_history
    #     if log["log_p"] is None and self.classifier is not None and condition_cg is not None:
    #         with torch.no_grad():
    #             logp = self.classifier.logp(xt, t[-1].repeat(n_samples), condition_vec_cg)
    #         log["log_p"] = logp
    #
    #     return xt, log
    #
    # def sample_with_sde_dpm_1(
    #         self,
    #         # ---------- the known fixed portion ---------- #
    #         prior: Optional[torch.Tensor] = None,
    #         # ----------------- sampling ----------------- #
    #         n_samples: int = 1,
    #         sample_steps: int = None,
    #         use_ema: bool = True,
    #         kappa: float = 1.0,
    #         temperature: float = 1.0,
    #         # ------------------ guidance ------------------ #
    #         condition_cfg=None,
    #         mask_cfg=None,
    #         w_cfg: float = 0.0,
    #         condition_cg=None,
    #         w_cg: float = 0.0,
    #
    #         preserve_history: bool = False,
    # ):
    #     log, x_history = {}, None
    #     model = self.model_ema if use_ema else self.model
    #
    #     idx = torch.arange(sample_steps + 1)
    #     t = (((sample_steps - idx) / sample_steps * self.t_range[1] ** (1 / kappa) +
    #           idx / sample_steps * self.t_range[0] ** (1 / kappa)) ** kappa).to(self.device)
    #
    #     alphas = self.alpha_schedule(t)
    #     sigmas = (1 - alphas ** 2).sqrt()
    #     logSNRs = (alphas / sigmas).log()
    #     h = torch.zeros_like(logSNRs)
    #     h[1:] = logSNRs[1:] - logSNRs[:-1]
    #
    #     if prior is None:
    #         xt = torch.randn((n_samples, *self.default_x_shape), device=self.device) * temperature
    #     else:
    #         xt = torch.randn_like(prior, device=self.device) * temperature
    #         xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
    #
    #     if preserve_history:
    #         x_history = np.empty((n_samples, sample_steps + 1, *xt.shape))
    #         x_history[:, 0] = xt.cpu().numpy()
    #
    #     with torch.no_grad():
    #         condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
    #         condition_vec_cg = condition_cg
    #
    #     for i in range(1, sample_steps + 1):
    #
    #         eps_theta, log = self.predict_eps_theta(
    #             xt, t[i - 1].repeat(n_samples), alphas[i - 1], sigmas[i - 1],
    #             use_ema, condition_vec_cfg, w_cfg, condition_vec_cg, w_cg)
    #
    #         xt = (
    #                 alphas[i] / alphas[i - 1] * xt -
    #                 2. * sigmas[i] * torch.expm1(h[i]) * eps_theta +
    #                 sigmas[i] * torch.expm1(2. * h[i]).sqrt() * torch.randn_like(xt))
    #         xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
    #
    #         if preserve_history:
    #             x_history[:, i] = xt.cpu().numpy()
    #
    #     log["sample_history"] = x_history
    #     if log["log_p"] is None and self.classifier is not None and condition_cg is not None:
    #         with torch.no_grad():
    #             logp = self.classifier.logp(xt, t[-1].repeat(n_samples), condition_vec_cg)
    #         log["log_p"] = logp
    #
    #     return xt, log
    #
    # def sample_with_sde_dpmpp_1(
    #         self,
    #         # ---------- the known fixed portion ---------- #
    #         prior: Optional[torch.Tensor] = None,
    #         # ----------------- sampling ----------------- #
    #         n_samples: int = 1,
    #         sample_steps: int = None,
    #         use_ema: bool = True,
    #         kappa: float = 1.0,
    #         temperature: float = 1.0,
    #         # ------------------ guidance ------------------ #
    #         condition_cfg=None,
    #         mask_cfg=None,
    #         w_cfg: float = 0.0,
    #         condition_cg=None,
    #         w_cg: float = 0.0,
    #
    #         preserve_history: bool = False,
    # ):
    #     log, x_history = {}, None
    #     model = self.model_ema if use_ema else self.model
    #
    #     idx = torch.arange(sample_steps + 1)
    #     t = (((sample_steps - idx) / sample_steps * self.t_range[1] ** (1 / kappa) +
    #           idx / sample_steps * self.t_range[0] ** (1 / kappa)) ** kappa).to(self.device)
    #
    #     alphas = self.alpha_schedule(t)
    #     sigmas = (1 - alphas ** 2).sqrt()
    #     logSNRs = (alphas / sigmas).log()
    #     h = torch.zeros_like(logSNRs)
    #     h[1:] = logSNRs[1:] - logSNRs[:-1]
    #
    #     if prior is None:
    #         xt = torch.randn((n_samples, *self.default_x_shape), device=self.device) * temperature
    #     else:
    #         xt = torch.randn_like(prior, device=self.device) * temperature
    #         xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
    #
    #     if preserve_history:
    #         x_history = np.empty((n_samples, sample_steps + 1, *xt.shape))
    #         x_history[:, 0] = xt.cpu().numpy()
    #
    #     with torch.no_grad():
    #         condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
    #         condition_vec_cg = condition_cg
    #
    #     for i in range(1, sample_steps + 1):
    #
    #         x_theta, log = self.predict_x_theta(
    #             xt, t[i - 1].repeat(n_samples), alphas[i - 1], sigmas[i - 1],
    #             use_ema, condition_vec_cfg, w_cfg, condition_vec_cg, w_cg)
    #
    #         xt = (
    #                 sigmas[i] / sigmas[i - 1] * (-h[i]).exp() * xt +
    #                 alphas[i] * (1. - (-2. * h[i]).exp()) * x_theta +
    #                 sigmas[i] * (1. - (-2. * h[i]).exp()).sqrt() * torch.randn_like(xt))
    #         xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
    #
    #         if preserve_history:
    #             x_history[:, i] = xt.cpu().numpy()
    #
    #     log["sample_history"] = x_history
    #     if log["log_p"] is None and self.classifier is not None and condition_cg is not None:
    #         with torch.no_grad():
    #             logp = self.classifier.logp(xt, t[-1].repeat(n_samples), condition_vec_cg)
    #         log["log_p"] = logp
    #
    #     return xt, log

    # def sample_with_ode_dpm_2(
    #         self,
    #         # ---------- the known fixed portion ---------- #
    #         prior: Optional[torch.Tensor] = None,
    #         # ----------------- sampling ----------------- #
    #         n_samples: int = 1,
    #         sample_steps: int = None,
    #         use_ema: bool = True,
    #         kappa: float = 1.0,
    #         temperature: float = 1.0,
    #         # ------------------ guidance ------------------ #
    #         condition_cfg=None,
    #         mask_cfg=None,
    #         w_cfg: float = 0.0,
    #         condition_cg=None,
    #         w_cg: float = 0.0,
    #
    #         preserve_history: bool = False,
    # ):
    #     raise NotImplementedError

    # def sample_with_ode_dpmpp_1(
    #         self,
    #         # ---------- the known fixed portion ---------- #
    #         prior: Optional[torch.Tensor] = None,
    #         # ----------------- sampling ----------------- #
    #         n_samples: int = 1,
    #         sample_steps: int = 5,
    #         use_ema: bool = True,
    #         kappa: float = 1.0,
    #         temperature: float = 1.0,
    #         # ------------------ guidance ------------------ #
    #         condition_cfg=None,
    #         mask_cfg=None,
    #         w_cfg: float = 0.0,
    #         condition_cg=None,
    #         w_cg: float = 0.0,
    #
    #         preserve_history: bool = False,
    # ):
    #     raise NotImplementedError
