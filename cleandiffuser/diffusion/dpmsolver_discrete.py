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


class DPMSolverDiscrete(DiffusionModel):

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

        if self.noise_schedule == "linear":
            beta0, beta1 = 0.1, 20
            self.t_cont = torch.linspace(t_eps, 1., diffusion_steps, device=device)
            self.alphas = (-(beta1 - beta0) / 4 * (self.t_cont ** 2) - beta0 / 2 * self.t_cont).exp()
        elif self.noise_schedule == "cosine":
            s = 0.008
            self.t_cont = torch.linspace(t_eps, 0.9946, diffusion_steps, device=device)
            self.alphas = ((torch.cos(np.pi / 2 * (self.t_cont + s) / (1 + s))).log()
                           - np.log(np.cos(np.pi / 2 * s / (1 + s)))).exp()
        else:
            raise ValueError(f"noise_schedule should be 'linear' or 'cosine', but got {self.noise_schedule}.")

        self.sigmas = (1 - self.alphas ** 2).sqrt()
        self.logSNRs = (self.alphas / self.sigmas).log()

    # ---------------------------------------------------------------------------
    # Training

    def add_noise(self, x0, idx=None, eps=None):
        if idx is None:
            idx = torch.randint(self.diffusion_steps, (x0.shape[0],), device=self.device)
        eps = torch.randn_like(x0) if eps is None else eps
        alpha = at_least_ndim(self.alphas[idx], x0.dim())
        sigma = at_least_ndim(self.sigmas[idx], x0.dim())
        xt = x0 * alpha + sigma * eps
        xt = xt * (1. - self.fix_mask) + x0 * self.fix_mask
        return xt, idx, eps

    def loss(self, x0, condition=None):
        xt, idx, eps = self.add_noise(x0)
        condition = self.model["condition"](condition) if condition is not None else None

        if self.predict_noise:
            loss = (self.model["diffusion"](xt, idx, condition) - eps) ** 2
        else:
            loss = (self.model["diffusion"](xt, idx, condition) - x0) ** 2

        return (loss * self.loss_weight * (1. - self.fix_mask)).mean()

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
        xt, idx, eps = self.add_noise(x0)
        log = self.classifier.update(xt, idx, condition)
        return log

    # ---------------------------------------------------------------------------
    # Inference

    def predict_eps_theta(
            self, x, idx, alpha, sigma, use_ema=False,
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
        with torch.no_grad():
            if w_cfg != 0.0 and w_cfg != 1.0:
                repeat_dim = [2 if i == 0 else 1 for i in range(x.dim())]
                condition_vec_cfg = torch.cat([condition_vec_cfg, torch.zeros_like(condition_vec_cfg)], 0)
                pred = model["diffusion"](
                    x.repeat(*repeat_dim), idx.repeat(2), condition_vec_cfg)
                pred = w_cfg * pred[:b] + (1. - w_cfg) * pred[b:]
            else:
                pred = model["diffusion"](x, idx, condition_vec_cfg)

        if self.predict_noise:
            eps_theta = pred
        else:
            eps_theta = xtheta_to_epstheta(x, alpha, sigma, pred)

        # ----------------- CG ----------------- #
        if self.classifier is not None and w_cg != 0.0 and condition_vec_cg is not None:
            log_p, grad = self.classifier.gradients(x.clone(), idx, condition_vec_cg)
            eps_theta = eps_theta - w_cg * sigma * grad
        else:
            log_p = None

        eps_theta = eps_theta * (1 - self.fix_mask)

        return eps_theta, {"log_p": log_p}

    def predict_x_theta(
            self, x, idx, alpha, sigma, use_ema=False,
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
        with torch.no_grad():
            if w_cfg != 0.0 and w_cfg != 1.0:
                repeat_dim = [2 if i == 0 else 1 for i in range(x.dim())]
                condition_vec_cfg = torch.cat([condition_vec_cfg, torch.zeros_like(condition_vec_cfg)], 0)
                pred = model["diffusion"](
                    x.repeat(*repeat_dim), idx.repeat(2), condition_vec_cfg)
                pred = w_cfg * pred[:b] + (1. - w_cfg) * pred[b:]
            else:
                pred = model["diffusion"](x, idx, condition_vec_cfg)

        if self.predict_noise:
            x_theta = epstheta_to_xtheta(x, alpha, sigma, pred)
        else:
            x_theta = pred

        # ----------------- CG ----------------- #
        if self.classifier is not None and w_cg != 0.0 and condition_vec_cg is not None:
            log_p, grad = self.classifier.gradients(x.clone(), idx, condition_vec_cg)
            x_theta = x_theta + w_cg * ((sigma ** 2) / alpha) * grad
        else:
            log_p = None

        x_theta = x_theta * (1 - self.fix_mask) + x * self.fix_mask

        return x_theta, {"log_p": log_p}

    def sample_with_ode_dpm_1(
            self,
            # ---------- the known fixed portion ---------- #
            prior: Optional[torch.Tensor] = None,
            # ----------------- sampling ----------------- #
            n_samples: int = 1,
            sample_steps: int = None,
            use_ema: bool = True,
            kappa: float = 1.0,
            temperature: float = 1.0,
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

        idx = torch.arange(sample_steps + 1)
        t_cond = (((sample_steps - idx) / sample_steps * 1. ** (1 / kappa) +
                   idx / sample_steps * self.t_eps ** (1 / kappa)) ** kappa).to(self.device)
        t_sample = (torch.max(t_cond-1/self.diffusion_steps, torch.zeros_like(t_cond))*self.diffusion_steps).long()

        alphas = self.alphas[t_sample]
        sigmas = self.sigmas[t_sample]
        logSNRs = self.logSNRs[t_sample]
        h = torch.zeros_like(logSNRs)
        h[1:] = logSNRs[1:] - logSNRs[:-1]

        if prior is None:
            xt = torch.randn((n_samples, *self.default_x_shape), device=self.device) * temperature
        else:
            xt = torch.randn_like(prior, device=self.device) * temperature
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask

        if preserve_history:
            x_history = np.empty((n_samples, sample_steps + 1, *xt.shape))
            x_history[:, 0] = xt.cpu().numpy()

        with torch.no_grad():
            condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
            condition_vec_cg = condition_cg

        for i in range(1, sample_steps + 1):

            eps_theta, log = self.predict_eps_theta(
                xt, t_sample[i - 1].repeat(n_samples), alphas[i - 1], sigmas[i - 1],
                use_ema, condition_vec_cfg, w_cfg, condition_vec_cg, w_cg)

            xt = alphas[i] / alphas[i - 1] * xt - sigmas[i] * torch.expm1(h[i]) * eps_theta
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask

            if preserve_history:
                x_history[:, i] = xt.cpu().numpy()

        log["sample_history"] = x_history
        if log["log_p"] is None and self.classifier is not None and condition_cg is not None:
            with torch.no_grad():
                logp = self.classifier.logp(xt, t_sample[-1].repeat(n_samples), condition_vec_cg)
            log["log_p"] = logp

        return xt, log

    def sample_with_sde_dpm_1(
            self,
            # ---------- the known fixed portion ---------- #
            prior: Optional[torch.Tensor] = None,
            # ----------------- sampling ----------------- #
            n_samples: int = 1,
            sample_steps: int = None,
            use_ema: bool = True,
            kappa: float = 1.0,
            temperature: float = 1.0,
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

        idx = torch.arange(sample_steps + 1)
        t_cond = (((sample_steps - idx) / sample_steps * 1. ** (1 / kappa) +
                   idx / sample_steps * self.t_eps ** (1 / kappa)) ** kappa).to(self.device)
        t_sample = (torch.max(t_cond - 1 / self.diffusion_steps,
                              torch.zeros_like(t_cond)) * self.diffusion_steps).long()

        alphas = self.alphas[t_sample]
        sigmas = self.sigmas[t_sample]
        logSNRs = self.logSNRs[t_sample]
        h = torch.zeros_like(logSNRs)
        h[1:] = logSNRs[1:] - logSNRs[:-1]

        if prior is None:
            xt = torch.randn((n_samples, *self.default_x_shape), device=self.device) * temperature
        else:
            xt = torch.randn_like(prior, device=self.device) * temperature
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask

        if preserve_history:
            x_history = np.empty((n_samples, sample_steps + 1, *xt.shape))
            x_history[:, 0] = xt.cpu().numpy()

        with torch.no_grad():
            condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
            condition_vec_cg = condition_cg

        for i in range(1, sample_steps + 1):

            eps_theta, log = self.predict_eps_theta(
                xt, t_sample[i - 1].repeat(n_samples), alphas[i - 1], sigmas[i - 1],
                use_ema, condition_vec_cfg, w_cfg, condition_vec_cg, w_cg)

            xt = (
                    alphas[i] / alphas[i - 1] * xt -
                    2. * sigmas[i] * torch.expm1(h[i]) * eps_theta +
                    sigmas[i] * torch.expm1(2. * h[i]).sqrt() * torch.randn_like(xt))
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask

            if preserve_history:
                x_history[:, i] = xt.cpu().numpy()

        log["sample_history"] = x_history
        if log["log_p"] is None and self.classifier is not None and condition_cg is not None:
            with torch.no_grad():
                logp = self.classifier.logp(xt, t_sample[-1].repeat(n_samples), condition_vec_cg)
            log["log_p"] = logp

        return xt, log

    def sample_with_sde_dpmpp_1(
            self,
            # ---------- the known fixed portion ---------- #
            prior: Optional[torch.Tensor] = None,
            # ----------------- sampling ----------------- #
            n_samples: int = 1,
            sample_steps: int = None,
            use_ema: bool = True,
            kappa: float = 1.0,
            temperature: float = 1.0,
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

        idx = torch.arange(sample_steps + 1)
        t_cond = (((sample_steps - idx) / sample_steps * 1. ** (1 / kappa) +
                   idx / sample_steps * self.t_eps ** (1 / kappa)) ** kappa).to(self.device)
        t_sample = (torch.max(t_cond - 1 / self.diffusion_steps,
                              torch.zeros_like(t_cond)) * self.diffusion_steps).long()

        alphas = self.alphas[t_sample]
        sigmas = self.sigmas[t_sample]
        logSNRs = self.logSNRs[t_sample]
        h = torch.zeros_like(logSNRs)
        h[1:] = logSNRs[1:] - logSNRs[:-1]

        if prior is None:
            xt = torch.randn((n_samples, *self.default_x_shape), device=self.device) * temperature
        else:
            xt = torch.randn_like(prior, device=self.device) * temperature
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask

        if preserve_history:
            x_history = np.empty((n_samples, sample_steps + 1, *xt.shape))
            x_history[:, 0] = xt.cpu().numpy()

        with torch.no_grad():
            condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
            condition_vec_cg = condition_cg

        for i in range(1, sample_steps + 1):

            x_theta, log = self.predict_x_theta(
                xt, t_sample[i - 1].repeat(n_samples), alphas[i - 1], sigmas[i - 1],
                use_ema, condition_vec_cfg, w_cfg, condition_vec_cg, w_cg)

            xt = (
                    sigmas[i] / sigmas[i - 1] * (-h[i]).exp() * xt +
                    alphas[i] * (1. - (-2. * h[i]).exp()) * x_theta +
                    sigmas[i] * (1. - (-2. * h[i]).exp()).sqrt() * torch.randn_like(xt))
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask

            if preserve_history:
                x_history[:, i] = xt.cpu().numpy()

        log["sample_history"] = x_history
        if log["log_p"] is None and self.classifier is not None and condition_cg is not None:
            with torch.no_grad():
                logp = self.classifier.logp(xt, t_sample[-1].repeat(n_samples), condition_vec_cg)
            log["log_p"] = logp

        return xt, log

    def sample_with_ode_dpm_2(
            self,
            # ---------- the known fixed portion ---------- #
            prior: Optional[torch.Tensor] = None,
            # ----------------- sampling ----------------- #
            n_samples: int = 1,
            sample_steps: int = None,
            use_ema: bool = True,
            kappa: float = 1.0,
            temperature: float = 1.0,
            # ------------------ guidance ------------------ #
            condition_cfg=None,
            mask_cfg=None,
            w_cfg: float = 0.0,
            condition_cg=None,
            w_cg: float = 0.0,

            preserve_history: bool = False,
    ):
        raise NotImplementedError

    def sample_with_ode_dpmpp_1(
            self,
            # ---------- the known fixed portion ---------- #
            prior: Optional[torch.Tensor] = None,
            # ----------------- sampling ----------------- #
            n_samples: int = 1,
            sample_steps: int = 5,
            use_ema: bool = True,
            kappa: float = 1.0,
            temperature: float = 1.0,
            # ------------------ guidance ------------------ #
            condition_cfg=None,
            mask_cfg=None,
            w_cfg: float = 0.0,
            condition_cg=None,
            w_cg: float = 0.0,

            preserve_history: bool = False,
    ):
        raise NotImplementedError

    def sample_with_ode_dpmpp_2(
            self,
            # ---------- the known fixed portion ---------- #
            prior: Optional[torch.Tensor] = None,
            # ----------------- sampling ----------------- #
            n_samples: int = 1,
            sample_steps: int = 5,
            use_ema: bool = True,
            kappa: float = 1.0,
            temperature: float = 1.0,
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

        idx = torch.arange(sample_steps + 1)
        t_cond = (((sample_steps - idx) / sample_steps * 1. ** (1 / kappa) +
                   idx / sample_steps * self.t_eps ** (1 / kappa)) ** kappa).to(self.device)
        t_sample = (torch.max(t_cond - 1 / self.diffusion_steps,
                              torch.zeros_like(t_cond)) * self.diffusion_steps).long()

        alphas = self.alphas[t_sample]
        sigmas = self.sigmas[t_sample]
        logSNRs = self.logSNRs[t_sample]
        h = torch.zeros_like(logSNRs)
        h[1:] = logSNRs[1:] - logSNRs[:-1]
        buffer = []

        if prior is None:
            xt = torch.randn((n_samples, *self.default_x_shape), device=self.device) * temperature
        else:
            xt = torch.randn_like(prior, device=self.device) * temperature
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask

        if preserve_history:
            x_history = np.empty((n_samples, sample_steps + 1, *xt.shape))
            x_history[:, 0] = xt.cpu().numpy()

        with torch.no_grad():
            condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
            condition_vec_cg = condition_cg

        x_theta_t0, log = self.predict_x_theta(
            xt, t_sample[0].repeat(n_samples), alphas[0], sigmas[0], use_ema, condition_vec_cfg, w_cfg, condition_vec_cg, w_cg)
        x_theta_t0 = x_theta_t0 * (1. - self.fix_mask) + prior * self.fix_mask
        buffer.append(x_theta_t0)

        xt = sigmas[1] / sigmas[0] * xt - alphas[1] * torch.expm1(-h[1]) * buffer[-1]
        xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
        if preserve_history:
            x_history[:, 1] = xt.cpu().numpy()

        x_theta_t1, log = self.predict_x_theta(
            xt, t_sample[1].repeat(n_samples), alphas[1], sigmas[1], use_ema, condition_vec_cfg, w_cfg, condition_vec_cg, w_cg)
        x_theta_t1 = x_theta_t1 * (1. - self.fix_mask) + prior * self.fix_mask
        buffer.append(x_theta_t1)

        for i in range(2, sample_steps + 1):

            r = h[i - 1] / h[i]

            D = (1 + 0.5 / r) * buffer[-1] - 0.5 / r * buffer[-2]

            xt = sigmas[i] / sigmas[i - 1] * xt - alphas[i] * torch.expm1(-h[i]) * D
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask

            if i < sample_steps + 1:
                x_theta_ti, log = self.predict_x_theta(
                    xt, t_sample[i].repeat(n_samples), alphas[i], sigmas[i],
                    use_ema, condition_vec_cfg, w_cfg, condition_vec_cg, w_cg)
                x_theta_ti = x_theta_ti * (1. - self.fix_mask) + prior * self.fix_mask
                buffer.append(x_theta_ti.clone())

            if preserve_history:
                x_history[:, i] = xt.cpu().numpy()

        log["sample_history"] = x_history
        if log["log_p"] is None and self.classifier is not None and condition_cg is not None:
            with torch.no_grad():
                logp = self.classifier.logp(xt, t_sample[-1].repeat(n_samples), condition_vec_cg)
            log["log_p"] = logp

        return xt, log

    def sample(
            self,
            # ---------- the known fixed portion ---------- #
            prior: Optional[torch.Tensor] = None,
            # ----------------- sampling ----------------- #
            n_samples: int = 1,
            sample_steps: int = 5,
            use_ema: bool = True,
            kappa: float = 1.0,
            temperature: float = 1.0,
            solver: str = "dpmsolver++",  # or ddim
            # ------------------ guidance ------------------ #
            condition_cfg=None,
            mask_cfg=None,
            w_cfg: float = 0.0,
            condition_cg=None,
            w_cg: float = 0.0,

            preserve_history: bool = False,
    ):
        if solver == "ode_dpm_1" or solver == "ddim":
            return self.sample_with_ode_dpm_1(
                prior, n_samples, sample_steps, use_ema, kappa, temperature, condition_cfg, mask_cfg, w_cfg,
                condition_cg, w_cg,
                preserve_history)
        elif solver == "sde_dpm_1":
            return self.sample_with_sde_dpm_1(
                prior, n_samples, sample_steps, use_ema, kappa, temperature, condition_cfg, mask_cfg, w_cfg,
                condition_cg, w_cg,
                preserve_history)
        elif solver == "sde_dpmpp_1":
            return self.sample_with_sde_dpmpp_1(
                prior, n_samples, sample_steps, use_ema, kappa, temperature, condition_cfg, mask_cfg, w_cfg,
                condition_cg, w_cg,
                preserve_history)
        elif solver == "ode_dpm_2":
            return self.sample_with_ode_dpm_2(
                prior, n_samples, sample_steps, use_ema, kappa, temperature, condition_cfg, mask_cfg, w_cfg,
                condition_cg, w_cg,
                preserve_history)
        elif solver == "ode_dpmpp_1":
            return self.sample_with_ode_dpmpp_1(
                prior, n_samples, sample_steps, use_ema, kappa, temperature, condition_cfg, mask_cfg, w_cfg,
                condition_cg, w_cg,
                preserve_history)
        elif solver == "ode_dpmpp_2":
            return self.sample_with_ode_dpmpp_2(
                prior, n_samples, sample_steps, use_ema, kappa, temperature, condition_cfg, mask_cfg, w_cfg,
                condition_cg, w_cg,
                preserve_history)
        else:
            raise ValueError(f"Solver: {solver} has not been implemented yet.")
