from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from cleandiffuser.classifier import BaseClassifier
from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import (
    at_least_ndim)
from .basic import DiffusionModel


class ContinuousEDM(DiffusionModel):
    """Continuous-time EDM
    
    EDM posits that the concepts of `t` in the diffusion process and the noise schedule are equivalent.
    Previous noise schedules can be interpreted as perturbing the data with Gaussian noise 
    followed by scaling. EDM sets the standard deviation of noise as `t`, the scale as 1, 
    and devises a series of preconditioning steps to aid in model learning.
    
    The current implementation of EDM only supports continuous-time ODEs.
    The sampling steps are required to be greater than 1.

    Args:
    - nn_diffusion: BaseNNDiffusion
        The neural network backbone for the Diffusion model.
    - nn_condition: Optional[BaseNNCondition]
        The neural network backbone for the condition embedding.
        
    - fix_mask: Union[list, np.ndarray, torch.Tensor]
        Fix some portion of the input data, and only allow the diffusion model to complete the rest part.
        The mask should be in the shape of `x_shape`.
    - loss_weight: Union[list, np.ndarray, torch.Tensor]
        Add loss weight. The weight should be in the shape of `x_shape`.
        
    - classifier: Optional[BaseClassifier]
        Add a classifier to enable classifier-guidance.
        
    - grad_clip_norm: Optional[float]
        Gradient clipping norm.
    - ema_rate: float
        Exponential moving average rate.
    - optim_params: Optional[dict]
        Optimizer parameters.
        
    - sigma_data: float
        The standard deviation of the data. Default: 0.5.
    - sigma_min: float
        The minimum standard deviation of the noise. Default: 0.002.
    - sigma_max: float
        The maximum standard deviation of the noise. Default: 80.
    - rho: float
        The power of the noise schedule. Default: 7.
    - P_mean: float
        Hyperparameter for noise sampling during training. Default: -1.2.
    - P_std: float
        Hyperparameter for noise sampling during training. Default: 1.2.
        
    - x_max: Optional[torch.Tensor]
        The maximum value for the input data. `None` indicates no constraint.
    - x_min: Optional[torch.Tensor]
        The minimum value for the input data. `None` indicates no constraint.
        
    - device: Union[torch.device, str]
        The device to run the model.
    """
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

            # ------------------ Plugins ---------------- #
            # Add a classifier to enable classifier-guidance
            classifier: Optional[BaseClassifier] = None,

            # ------------------ Training Params ---------------- #
            grad_clip_norm: Optional[float] = None,
            ema_rate: float = 0.995,
            optim_params: Optional[dict] = None,

            # ------------------- Diffusion Params ------------------- #
            sigma_data: float = 0.5,
            sigma_min: float = 0.002,
            sigma_max: float = 80.,
            rho: float = 7.,
            P_mean: float = -1.2,
            P_std: float = 1.2,

            x_max: Optional[torch.Tensor] = None,
            x_min: Optional[torch.Tensor] = None,

            device: Union[torch.device, str] = "cpu"
    ):
        super().__init__(
            nn_diffusion, nn_condition, fix_mask, loss_weight, classifier, grad_clip_norm,
            0, ema_rate, optim_params, device)

        self.sigma_data, self.sigma_min, self.sigma_max = sigma_data, sigma_min, sigma_max
        self.rho, self.P_mean, self.P_std = rho, P_mean, P_std

        self.x_max = x_max.to(device) if isinstance(x_max, torch.Tensor) else x_max
        self.x_min = x_min.to(device) if isinstance(x_min, torch.Tensor) else x_min

        # ==================== Continuous Time-step Range ====================
        self.t_diffusion = [sigma_min, sigma_max]

        # ======================= Noise Schedule =========================
        # scale(t) = 1., dot_scale(t) = 0.
        # sigma(t) = t , dot_sigma(t) = 1.

    @property
    def supported_solvers(self):
        return ["euler", "heun"]

    @property
    def clip_pred(self):
        return (self.x_max is not None) or (self.x_min is not None)

    # ===================== EDM Pre-conditioning =========================
    def c_skip(self, sigma):
        return self.sigma_data ** 2 / (self.sigma_data ** 2 + sigma ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data / (self.sigma_data ** 2 + sigma ** 2).sqrt()

    def c_in(self, sigma):
        return 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()

    def c_noise(self, sigma):
        return 0.25 * sigma.log()

    def D(self, x, sigma, condition=None, model=None):
        """ Prepositioning in EDM """
        c_skip, c_out, c_in, c_noise = self.c_skip(sigma), self.c_out(sigma), self.c_in(sigma), self.c_noise(sigma)
        if model is None:
            model = self.model
        c_skip, c_in, c_out = at_least_ndim(c_skip, x.dim()), at_least_ndim(c_in, x.dim()), at_least_ndim(c_out, x.dim())
        return c_skip * x + c_out * model["diffusion"](c_in * x, c_noise, condition)

    # ==================== Training: Score Matching ======================

    def add_noise(self, x0, t=None, eps=None):

        t = (torch.randn((x0.shape[0], ), device=self.device) * self.P_std + self.P_mean).exp() if t is None else t

        eps = torch.randn_like(x0) if eps is None else eps

        scale = 1.
        sigma = at_least_ndim(t, x0.dim())

        xt = scale * x0 + sigma * eps
        xt = (1. - self.fix_mask) * xt + self.fix_mask * x0

        return xt, t, eps

    def loss(self, x0, condition=None):

        xt, t, eps = self.add_noise(x0)

        condition = self.model["condition"](condition) if condition is not None else None

        loss = (self.D(xt, t, condition) - x0) ** 2

        edm_loss_weight = at_least_ndim((t ** 2 + self.sigma_data ** 2) / ((t * self.sigma_data) ** 2), x0.dim())

        return (loss * self.loss_weight * (1 - self.fix_mask) * edm_loss_weight).mean()

    def update(self, x0, condition=None, update_ema=True, **kwargs):
        """One-step gradient update.
        Inputs:
        - x0: torch.Tensor
            Samples from the target distribution.
        - condition: Optional
            Condition of x0. `None` indicates no condition.
        - update_ema: bool
            Whether to update the exponential moving average model.

        Outputs:
        - log: dict
            The log dictionary.
        """
        loss = self.loss(x0, condition)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm) \
            if self.grad_clip_norm else None
        self.optimizer.step()
        self.optimizer.zero_grad()

        if update_ema:
            self.ema_update()

        log = {"loss": loss.item(), "grad_norm": grad_norm}

        return log

    def update_classifier(self, x0, condition):

        xt, t, eps = self.add_noise(x0)

        log = self.classifier.update(xt, t.log() / 4., condition)

        return log

    # ==================== Sampling: Solving SDE/ODE ======================

    def classifier_guidance(
            self, xt, t, sigma,
            model, condition=None, w: float = 1.0,
            pred=None):
        """
        Guided Sampling CG:
        bar_eps = eps - w * sigma * grad
        bar_x0  = x0 + w * (sigma ** 2) * alpha * grad
        """
        if pred is None:
            pred = self.D(xt, t, None, model)
        if self.classifier is None or w == 0.0 or condition is None:
            return pred, None
        else:
            log_p, grad = self.classifier.gradients(
                xt.clone(), t.log() / 4., condition)
            pred = pred + w * (at_least_ndim(sigma, pred.dim()) ** 2) * grad

        return pred, log_p

    def classifier_free_guidance(
            self, xt, t,
            model, condition=None, w: float = 1.0,
            pred=None, pred_uncond=None,
            requires_grad: bool = False):
        """
        Guided Sampling CFG:
        bar_eps = w * pred + (1 - w) * pred_uncond
        bar_x0  = w * pred + (1 - w) * pred_uncond
        """
        with torch.set_grad_enabled(requires_grad):
            if w != 0.0 and w != 1.0:
                if pred is None or pred_uncond is None:
                    b = xt.shape[0]
                    repeat_dim = [2 if i == 0 else 1 for i in range(xt.dim())]
                    condition = torch.cat([condition, torch.zeros_like(condition)], 0)
                    pred_all = self.D(
                        xt.repeat(*repeat_dim),
                        t.repeat(2), condition, model)
                    pred, pred_uncond = pred_all[:b], pred_all[b:]
            elif w == 0.0:
                pred = 0.
                pred_uncond = self.D(xt, t, None, model)
            else:
                pred = self.D(xt, t, condition, model)
                pred_uncond = 0.

        bar_pred = w * pred + (1 - w) * pred_uncond

        return bar_pred

    def guided_sampling(
            self, xt, t, sigma,
            model,
            condition_cfg=None, w_cfg: float = 0.0,
            condition_cg=None, w_cg: float = 0.0,
            requires_grad: bool = False):
        """
        One-step epsilon/x0 prediction with guidance.
        """

        pred = self.classifier_free_guidance(
            xt, t, model, condition_cfg, w_cfg, None, None, requires_grad)

        pred, logp = self.classifier_guidance(
            xt, t, sigma, model, condition_cg, w_cg, pred)

        return pred, logp

    def sample(
            self,
            # ---------- the known fixed portion ---------- #
            prior: torch.Tensor,
            # ----------------- sampling ----------------- #
            solver: str = "euler",  # euler or heun
            n_samples: int = 1,
            sample_steps: int = 5,
            use_ema: bool = True,
            temperature: float = 1.0,
            # ------------------ guidance ------------------ #
            condition_cfg=None,
            mask_cfg=None,
            w_cfg: float = 0.0,
            condition_cg=None,
            w_cg: float = 0.0,
            # ----------- Diffusion-X sampling ----------
            diffusion_x_sampling_steps: int = 0,
            # ----------- Warm-Starting -----------
            warm_start_reference: Optional[torch.Tensor] = None,
            warm_start_forward_level: float = 0.3,
            # ------------------ others ------------------ #
            requires_grad: bool = False,
            preserve_history: bool = False,
            **kwargs,
    ):
        """Sampling.
        
        Inputs:
        - prior: torch.Tensor
            The known fixed portion of the input data. Should be in the shape of generated data.
            Use `torch.zeros((n_samples, *x_shape))` for non-prior sampling.
        
        - solver: str
            The solver for the reverse process. Check `supported_solvers` property for available solvers.
        - n_samples: int
            The number of samples to generate.
        - sample_steps: int
            The number of sampling steps. Should be greater than 1 and less than or equal to the number of diffusion steps.
        - sample_step_schedule: Union[str, Callable]
            The schedule for the sampling steps.
        - use_ema: bool
            Whether to use the exponential moving average model.
        - temperature: float
            The temperature for sampling.
        
        - condition_cfg: Optional
            Condition for Classifier-free-guidance.
        - mask_cfg: Optional
            Mask for Classifier-guidance.
        - w_cfg: float
            Weight for Classifier-free-guidance.
        - condition_cg: Optional
            Condition for Classifier-guidance.
        - w_cg: float
            Weight for Classifier-guidance.
            
        - diffusion_x_sampling_steps: int
            The number of diffusion steps for diffusion-x sampling.
        
        - warm_start_reference: Optional[torch.Tensor]
            Reference data for warm-starting sampling. `None` indicates no warm-starting.
        - warm_start_forward_level: float
            The forward noise level to perturb the reference data. Should be in the range of `[0., 1.]`, where `1` indicates pure noise.
        
        - requires_grad: bool
            Whether to preserve gradients.
        - preserve_history: bool
            Whether to preserve the sampling history.
            
        Outputs:
        - x0: torch.Tensor
            Generated samples. Be in the shape of `(n_samples, *x_shape)`.
        - log: dict
            The log dictionary.
        """
        assert solver in ["euler", "heun"], f"Solver {solver} is not supported. Use 'euler' or 'heun' instead."

        # ===================== Initialization =====================
        log = {
            "sample_history": np.empty((n_samples, sample_steps + 1, *prior.shape)) if preserve_history else None, }

        model = self.model if not use_ema else self.model_ema

        prior = prior.to(self.device)
        if isinstance(warm_start_reference, torch.Tensor) and warm_start_forward_level > 0.:
            fwd_sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * warm_start_forward_level
            xt = warm_start_reference + fwd_sigma * torch.randn_like(warm_start_reference)
        else:
            fwd_sigma = self.sigma_max
            xt = torch.randn_like(prior) * self.sigma_max * temperature
        xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
        if preserve_history:
            log["sample_history"][:, 0] = xt.cpu().numpy()

        with torch.set_grad_enabled(requires_grad):
            condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
            condition_vec_cg = condition_cg

        # ===================== Sampling Schedule ====================
        sample_step_schedule = ((self.sigma_min ** (1 / self.rho) + torch.arange(sample_steps + 1, device=self.device)
                                / sample_steps * (fwd_sigma ** (1 / self.rho) - self.sigma_min ** (1 / self.rho)))
                                ** self.rho)

        sigmas = sample_step_schedule

        # ===================== Denoising Loop ========================
        loop_steps = [1] * diffusion_x_sampling_steps + list(range(1, sample_steps + 1))
        for i in reversed(loop_steps):

            t = torch.full((n_samples,), sample_step_schedule[i], dtype=torch.float32, device=self.device)

            # guided sampling
            pred, logp = self.guided_sampling(
                xt, t, sigmas[i],
                model, condition_vec_cfg, w_cfg, condition_vec_cg, w_cg, requires_grad)

            # clip the prediction
            if self.clip_pred:
                pred = pred.clip(self.x_min, self.x_max)

            # one-step update
            dot_x = (xt - pred) / at_least_ndim(sigmas[i], xt.dim())
            delta_t = sample_step_schedule[i] - sample_step_schedule[i - 1]
            x_next = xt - dot_x * delta_t
            x_next = x_next * (1. - self.fix_mask) + prior * self.fix_mask

            if solver == "heun" and i > 1:
                pred, logp = self.guided_sampling(
                    x_next, t / sample_step_schedule[i] * sample_step_schedule[i - 1], sigmas[i - 1],
                    model, condition_vec_cfg, w_cfg, condition_vec_cg, w_cg, requires_grad)
                if self.clip_pred:
                    pred = pred.clip(self.x_min, self.x_max)
                dot_x_prime = (x_next - pred) / at_least_ndim(sigmas[i - 1], xt.dim())
                x_next = xt - (dot_x + dot_x_prime) / 2. * delta_t
                x_next = x_next * (1. - self.fix_mask) + prior * self.fix_mask

            xt = x_next

            if preserve_history:
                log["sample_history"][:, sample_steps - i + 1] = xt.cpu().numpy()

        # ================= Post-processing =================
        if self.classifier is not None:
            with torch.no_grad():
                t = torch.ones((n_samples,), dtype=torch.long, device=self.device) * self.sigma_min
                logp = self.classifier.logp(xt, t.log() / 4., condition_vec_cg)
            log["log_p"] = logp

        if self.clip_pred:
            xt = xt.clip(self.x_min, self.x_max)

        return xt, log
