from typing import Optional, Union, Callable, Dict

import numpy as np
import torch
import torch.nn as nn

from cleandiffuser.classifier import BaseClassifier
from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import (
    at_least_ndim,
    SUPPORTED_NOISE_SCHEDULES, SUPPORTED_DISCRETIZATIONS, SUPPORTED_SAMPLING_STEP_SCHEDULE)
from .basic import DiffusionModel

SUPPORTED_SOLVERS = [
    "ddpm", "ddim",
    "ode_dpmsolver_1", "ode_dpmsolver++_1", "ode_dpmsolver++_2M",
    "sde_dpmsolver_1", "sde_dpmsolver++_1", "sde_dpmsolver++_2M",]


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


class BaseDiffusionSDE(DiffusionModel):

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
            epsilon: float = 1e-3,

            noise_schedule: Union[str, Dict[str, Callable]] = "cosine",
            noise_schedule_params: Optional[dict] = None,

            x_max: Optional[torch.Tensor] = None,
            x_min: Optional[torch.Tensor] = None,

            predict_noise: bool = True,

            device: Union[torch.device, str] = "cpu"
    ):
        super().__init__(
            nn_diffusion, nn_condition, fix_mask, loss_weight, classifier, grad_clip_norm,
            0, ema_rate, optim_params, device)

        self.predict_noise = predict_noise
        self.epsilon = epsilon
        self.x_max = x_max.to(device) if isinstance(x_max, torch.Tensor) else x_max
        self.x_min = x_min.to(device) if isinstance(x_min, torch.Tensor) else x_min

    @property
    def supported_solvers(self):
        return SUPPORTED_SOLVERS

    @property
    def clip_pred(self):
        return (self.x_max is not None) or (self.x_min is not None)

    # ==================== Training: Score Matching ======================

    def add_noise(self, x0, t=None, eps=None):
        raise NotImplementedError

    def loss(self, x0, condition=None, **kwargs):

        xt, t, eps = self.add_noise(x0)

        condition = self.model["condition"](condition) if condition is not None else None

        if self.predict_noise:
            loss = (self.model["diffusion"](xt, t, condition) - eps) ** 2
        else:
            loss = (self.model["diffusion"](xt, t, condition) - x0) ** 2
        
        loss = loss * self.loss_weight * (1 - self.fix_mask)
        
        # find weighted_regression_tensor in kwargs
        weighted_regression_tensor = kwargs.get("weighted_regression_tensor", None)
        if weighted_regression_tensor is not None:
            loss *= weighted_regression_tensor.unsqueeze(-1)

        return loss.mean()

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
        loss = self.loss(x0, condition, **kwargs)

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

        log = self.classifier.update(xt, t, condition)

        return log

    # ==================== Sampling: Solving SDE/ODE ======================

    def classifier_guidance(
            self, xt, t, alpha, sigma,
            model, condition=None, w: float = 1.0,
            pred=None):
        """
        Guided Sampling CG:
        bar_eps = eps - w * sigma * grad
        bar_x0  = x0 + w * (sigma ** 2) * alpha * grad
        """
        if pred is None:
            pred = model["diffusion"](xt, t, None)
        if self.classifier is None or w == 0.0:
            return pred, None
        else:
            log_p, grad = self.classifier.gradients(xt.clone(), t, condition)
            if self.predict_noise:
                pred = pred - w * sigma * grad
            else:
                pred = pred + w * ((sigma ** 2) / alpha) * grad

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
                    pred_all = model["diffusion"](
                        xt.repeat(*repeat_dim), t.repeat(2), condition)
                    pred, pred_uncond = pred_all[:b], pred_all[b:]
            elif w == 0.0:
                pred = 0.
                pred_uncond = model["diffusion"](xt, t, None)
            else:
                pred = model["diffusion"](xt, t, condition)
                pred_uncond = 0.

        if self.predict_noise or not self.predict_noise:
            bar_pred = w * pred + (1 - w) * pred_uncond
        else:
            bar_pred = pred

        return bar_pred

    def clip_prediction(self, pred, xt, alpha, sigma):
        """
        Clip the prediction at each sampling step to stablize the generation.
        (xt - alpha * x_max) / sigma <= eps <= (xt - alpha * x_min) / sigma
                               x_min <= x0  <= x_max
        """
        if self.predict_noise:
            if self.clip_pred:
                upper_bound = (xt - alpha * self.x_min) / sigma if self.x_min is not None else None
                lower_bound = (xt - alpha * self.x_max) / sigma if self.x_max is not None else None
                pred = pred.clip(lower_bound, upper_bound)
        else:
            if self.clip_pred:
                pred = pred.clip(self.x_min, self.x_max)

        return pred

    def guided_sampling(
            self, xt, t, alpha, sigma,
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
            xt, t, alpha, sigma, model, condition_cg, w_cg, pred)

        return pred, logp

    def sample(self, *args, **kwargs):
        raise NotImplementedError


class DiscreteDiffusionSDE(BaseDiffusionSDE):
    """Discrete-time Diffusion SDE (VP-SDE)
    
    The Diffusion SDE is currently one of the most commonly used formulations of diffusion processes. 
    Its training process involves utilizing neural networks to estimate its scaled score function, 
    which is used to compute the reverse process. The Diffusion SDE has reverse processes 
    in both SDE and ODE forms, sharing the same marginal distribution. 
    The first-order discretized forms of both are equivalent to well-known models such as DDPM and DDIM. 
    DPM-Solvers have observed the semi-linearity of the reverse process and have computed its exact solution.
    
    The DiscreteDiffusionSDE is the discrete-time version of the Diffusion SDE. 
    It discretizes the continuous time interval into a finite number of diffusion steps 
    and only estimates the score function on these steps. 
    Therefore, in sampling, solvers can only work on these learned steps.
    The sampling steps are required to be greater than 1 and less than or equal to the number of diffusion steps.

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
        
    - epsilon: float
        The minimum time step for the diffusion reverse process. 
        In practice, using a very small value instead of `0` can avoid numerical instability.
    
    - diffusion_steps: int
        The discretization steps for discrete-time diffusion models.
    - discretization: Union[str, Callable]
        The discretization method for the diffusion steps.
        
    - noise_schedule: Union[str, Dict[str, Callable]]
        The noise schedule for the diffusion process. Can be "linear" or "cosine".
    - noise_schedule_params: Optional[dict]
        The parameters for the noise schedule.
        
    - x_max: Optional[torch.Tensor]
        The maximum value for the input data. `None` indicates no constraint.
    - x_min: Optional[torch.Tensor]
        The minimum value for the input data. `None` indicates no constraint.
        
    - predict_noise: bool
        Whether to predict the noise or the data.
        
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
            epsilon: float = 1e-3,

            diffusion_steps: int = 1000,
            discretization: Union[str, Callable] = "uniform",

            noise_schedule: Union[str, Dict[str, Callable]] = "cosine",
            noise_schedule_params: Optional[dict] = None,

            x_max: Optional[torch.Tensor] = None,
            x_min: Optional[torch.Tensor] = None,

            predict_noise: bool = True,

            device: Union[torch.device, str] = "cpu"
    ):
        super().__init__(
            nn_diffusion, nn_condition, fix_mask, loss_weight, classifier, grad_clip_norm, ema_rate, optim_params,
            epsilon, noise_schedule, noise_schedule_params, x_max, x_min, predict_noise, device)

        self.diffusion_steps = diffusion_steps

        if 1. / diffusion_steps < epsilon:
            raise ValueError("epsilon is too large for the number of diffusion steps")

        # ================= Discretization =================
        # - Map the continuous range [epsilon, 1.] to the discrete range [0, T-1]
        if isinstance(discretization, str):
            if discretization in SUPPORTED_DISCRETIZATIONS.keys():
                self.t_diffusion = SUPPORTED_DISCRETIZATIONS[discretization](diffusion_steps, epsilon).to(device)
            else:
                Warning(f"Discretization method {discretization} is not supported. "
                        f"Using uniform discretization instead.")
                self.t_diffusion = SUPPORTED_DISCRETIZATIONS["uniform"](diffusion_steps, epsilon).to(device)
        elif callable(discretization):
            self.t_diffusion = discretization(diffusion_steps, epsilon).to(device)
        else:
            raise ValueError("discretization must be a callable or a string")

        # ================= Noise Schedule =================
        if isinstance(noise_schedule, str):
            if noise_schedule in SUPPORTED_NOISE_SCHEDULES.keys():
                self.alpha, self.sigma = SUPPORTED_NOISE_SCHEDULES[noise_schedule]["forward"](
                    self.t_diffusion, **(noise_schedule_params or {}))
            else:
                raise ValueError(f"Noise schedule {noise_schedule} is not supported.")
        elif isinstance(noise_schedule, dict):
            self.alpha, self.sigma = noise_schedule["forward"](self.t_diffusion, **(noise_schedule_params or {}))
        else:
            raise ValueError("noise_schedule must be a callable or a string")

        self.logSNR = torch.log(self.alpha / self.sigma)

    # ==================== Training: Score Matching ======================

    def add_noise(self, x0, t=None, eps=None):

        t = torch.randint(self.diffusion_steps, (x0.shape[0],), device=self.device) if t is None else t
        eps = torch.randn_like(x0) if eps is None else eps

        alpha, sigma = at_least_ndim(self.alpha[t], x0.dim()), at_least_ndim(self.sigma[t], x0.dim())

        xt = alpha * x0 + sigma * eps
        xt = (1. - self.fix_mask) * xt + self.fix_mask * x0

        return xt, t, eps

    # ==================== Sampling: Solving SDE/ODE ======================

    def sample(
            self,
            # ---------- the known fixed portion ---------- #
            prior: torch.Tensor,
            # ----------------- sampling ----------------- #
            solver: str = "ddpm",
            n_samples: int = 1,
            sample_steps: int = 5,
            sample_step_schedule: Union[str, Callable] = "uniform",
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
        assert solver in SUPPORTED_SOLVERS, f"Solver {solver} is not supported."

        # ===================== Initialization =====================
        log = {
            "sample_history": np.empty((n_samples, sample_steps + 1, *prior.shape)) if preserve_history else None, }

        model = self.model if not use_ema else self.model_ema

        prior = prior.to(self.device)
        if isinstance(warm_start_reference, torch.Tensor):
            diffusion_steps = int(warm_start_forward_level * self.diffusion_steps)
            fwd_alpha, fwd_sigma = self.alpha[diffusion_steps], self.sigma[diffusion_steps]
            xt = warm_start_reference * fwd_alpha + fwd_sigma * torch.randn_like(warm_start_reference)
        else:
            diffusion_steps = self.diffusion_steps
            xt = torch.randn_like(prior) * temperature
        xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
        if preserve_history:
            log["sample_history"][:, 0] = xt.cpu().numpy()

        with torch.set_grad_enabled(requires_grad):
            condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
            condition_vec_cg = condition_cg

        # ===================== Sampling Schedule ====================
        if isinstance(sample_step_schedule, str):
            if sample_step_schedule in SUPPORTED_SAMPLING_STEP_SCHEDULE.keys():
                sample_step_schedule = SUPPORTED_SAMPLING_STEP_SCHEDULE[sample_step_schedule](
                    diffusion_steps, sample_steps)
            else:
                raise ValueError(f"Sampling step schedule {sample_step_schedule} is not supported.")
        elif callable(sample_step_schedule):
            sample_step_schedule = sample_step_schedule(diffusion_steps, sample_steps)
        else:
            raise ValueError("sample_step_schedule must be a callable or a string")

        alphas = self.alpha[sample_step_schedule]
        sigmas = self.sigma[sample_step_schedule]
        logSNRs = torch.log(alphas / sigmas)
        hs = torch.zeros_like(logSNRs)
        hs[1:] = logSNRs[:-1] - logSNRs[1:]  # hs[0] is not correctly calculated, but it will not be used.
        stds = torch.zeros((sample_steps + 1,), device=self.device)
        stds[1:] = sigmas[:-1] / sigmas[1:] * (1 - (alphas[1:] / alphas[:-1]) ** 2).sqrt()

        buffer = []

        # ===================== Denoising Loop ========================
        loop_steps = [1] * diffusion_x_sampling_steps + list(range(1, sample_steps + 1))
        for i in reversed(loop_steps):

            t = torch.full((n_samples,), sample_step_schedule[i], dtype=torch.long, device=self.device)

            # guided sampling
            pred, logp = self.guided_sampling(
                xt, t, alphas[i], sigmas[i],
                model, condition_vec_cfg, w_cfg, condition_vec_cg, w_cg, requires_grad)

            # clip the prediction
            pred = self.clip_prediction(pred, xt, alphas[i], sigmas[i])

            # noise & data prediction
            eps_theta = pred if self.predict_noise else xtheta_to_epstheta(xt, alphas[i], sigmas[i], pred)
            x_theta = pred if not self.predict_noise else epstheta_to_xtheta(xt, alphas[i], sigmas[i], pred)

            # one-step update
            if solver == "ddpm":
                xt = (
                        (alphas[i - 1] / alphas[i]) * (xt - sigmas[i] * eps_theta) +
                        (sigmas[i - 1] ** 2 - stds[i] ** 2 + 1e-8).sqrt() * eps_theta)
                if i > 1:
                    xt += (stds[i] * torch.randn_like(xt))

            elif solver == "ddim":
                xt = (alphas[i - 1] * ((xt - sigmas[i] * eps_theta) / alphas[i]) + sigmas[i - 1] * eps_theta)

            elif solver == "ode_dpmsolver_1":
                xt = (alphas[i - 1] / alphas[i]) * xt - sigmas[i - 1] * torch.expm1(hs[i]) * eps_theta

            elif solver == "ode_dpmsolver++_1":
                xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * x_theta

            elif solver == "ode_dpmsolver++_2M":
                buffer.append(x_theta)
                if i < sample_steps:
                    r = hs[i + 1] / hs[i]
                    D = (1 + 0.5 / r) * buffer[-1] - 0.5 / r * buffer[-2]
                    xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * D
                else:
                    xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * x_theta

            elif solver == "sde_dpmsolver_1":
                xt = ((alphas[i - 1] / alphas[i]) * xt -
                      2 * sigmas[i - 1] * torch.expm1(hs[i]) * eps_theta +
                      sigmas[i - 1] * torch.expm1(2 * hs[i]).sqrt() * torch.randn_like(xt))

            elif solver == "sde_dpmsolver++_1":
                xt = ((sigmas[i - 1] / sigmas[i]) * (-hs[i]).exp() * xt -
                      alphas[i - 1] * torch.expm1(-2 * hs[i]) * x_theta +
                      sigmas[i - 1] * (-torch.expm1(-2 * hs[i])).sqrt() * torch.randn_like(xt))

            elif solver == "sde_dpmsolver++_2M":
                buffer.append(x_theta)
                if i < sample_steps:
                    r = hs[i + 1] / hs[i]
                    D = (1 + 0.5 / r) * buffer[-1] - 0.5 / r * buffer[-2]
                    xt = ((sigmas[i - 1] / sigmas[i]) * (-hs[i]).exp() * xt -
                          alphas[i - 1] * torch.expm1(-2 * hs[i]) * D +
                          sigmas[i - 1] * (-torch.expm1(-2 * hs[i])).sqrt() * torch.randn_like(xt))
                else:
                    xt = ((sigmas[i - 1] / sigmas[i]) * (-hs[i]).exp() * xt -
                          alphas[i - 1] * torch.expm1(-2 * hs[i]) * x_theta +
                          sigmas[i - 1] * (-torch.expm1(-2 * hs[i])).sqrt() * torch.randn_like(xt))

            # fix the known portion, and preserve the sampling history
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
            if preserve_history:
                log["sample_history"][:, sample_steps - i + 1] = xt.cpu().numpy()

        # ================= Post-processing =================
        if self.classifier is not None:
            with torch.no_grad():
                t = torch.zeros((n_samples,), dtype=torch.long, device=self.device)
                logp = self.classifier.logp(xt, t, condition_vec_cg)
            log["log_p"] = logp

        if self.clip_pred:
            xt = xt.clip(self.x_min, self.x_max)

        return xt, log


class ContinuousDiffusionSDE(BaseDiffusionSDE):
    """Continuous-time Diffusion SDE (VP-SDE)
    
    The Diffusion SDE is currently one of the most commonly used formulations of diffusion processes. 
    Its training process involves utilizing neural networks to estimate its scaled score function, 
    which is used to compute the reverse process. The Diffusion SDE has reverse processes 
    in both SDE and ODE forms, sharing the same marginal distribution. 
    The first-order discretized forms of both are equivalent to well-known models such as DDPM and DDIM. 
    DPM-Solvers have observed the semi-linearity of the reverse process and have computed its exact solution.
    
    The ContinuousDiffusionSDE is the continuous-time version of the Diffusion SDE.
    It estimates the score function at any $t\in[0,T]$ and solves the reverse process in continuous time.
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
        
    - epsilon: float
        The minimum time step for the diffusion reverse process. 
        In practice, using a very small value instead of `0` can avoid numerical instability.
        
    - noise_schedule: Union[str, Dict[str, Callable]]
        The noise schedule for the diffusion process. Can be "linear" or "cosine".
    - noise_schedule_params: Optional[dict]
        The parameters for the noise schedule.
        
    - x_max: Optional[torch.Tensor]
        The maximum value for the input data. `None` indicates no constraint.
    - x_min: Optional[torch.Tensor]
        The minimum value for the input data. `None` indicates no constraint.
        
    - predict_noise: bool
        Whether to predict the noise or the data.
        
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
            epsilon: float = 1e-3,

            noise_schedule: Union[str, Dict[str, Callable]] = "cosine",
            noise_schedule_params: Optional[dict] = None,

            x_max: Optional[torch.Tensor] = None,
            x_min: Optional[torch.Tensor] = None,

            predict_noise: bool = True,

            device: Union[torch.device, str] = "cpu"
    ):
        super().__init__(
            nn_diffusion, nn_condition, fix_mask, loss_weight, classifier, grad_clip_norm, ema_rate, optim_params,
            epsilon, noise_schedule, noise_schedule_params, x_max, x_min, predict_noise, device)

        # ==================== Continuous Time-step Range ====================
        if noise_schedule == "cosine":
            self.t_diffusion = [epsilon, 0.9946]
        else:
            self.t_diffusion = [epsilon, 1.]

        # ===================== Noise Schedule ======================
        if isinstance(noise_schedule, str):
            if noise_schedule in SUPPORTED_NOISE_SCHEDULES.keys():
                self.noise_schedule_funcs = SUPPORTED_NOISE_SCHEDULES[noise_schedule]
                self.noise_schedule_params = noise_schedule_params
            else:
                raise ValueError(f"Noise schedule {noise_schedule} is not supported.")
        elif isinstance(noise_schedule, dict):
            self.noise_schedule_funcs = noise_schedule
            self.noise_schedule_params = noise_schedule_params
        else:
            raise ValueError("noise_schedule must be a callable or a string")

    # ==================== Training: Score Matching ======================

    def add_noise(self, x0, t=None, eps=None):

        t = (torch.rand((x0.shape[0],), device=self.device) *
             (self.t_diffusion[1] - self.t_diffusion[0]) + self.t_diffusion[0]) if t is None else t

        eps = torch.randn_like(x0) if eps is None else eps

        alpha, sigma = self.noise_schedule_funcs["forward"](t, **(self.noise_schedule_params or {}))
        alpha = at_least_ndim(alpha, x0.dim())
        sigma = at_least_ndim(sigma, x0.dim())

        xt = alpha * x0 + sigma * eps
        xt = (1. - self.fix_mask) * xt + self.fix_mask * x0

        return xt, t, eps

    # ==================== Sampling: Solving SDE/ODE ======================

    def sample(
            self,
            # ---------- the known fixed portion ---------- #
            prior: torch.Tensor,
            # ----------------- sampling ----------------- #
            solver: str = "ddpm",
            n_samples: int = 1,
            sample_steps: int = 5,
            sample_step_schedule: Union[str, Callable] = "uniform_continuous",
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
            The number of sampling steps. Should be greater than 1.
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
        assert solver in SUPPORTED_SOLVERS, f"Solver {solver} is not supported."

        # ===================== Initialization =====================
        log = {
            "sample_history": np.empty((n_samples, sample_steps + 1, *prior.shape)) if preserve_history else None, }

        model = self.model if not use_ema else self.model_ema

        prior = prior.to(self.device)
        if isinstance(warm_start_reference, torch.Tensor) and warm_start_forward_level > 0.:
            warm_start_forward_level = self.epsilon + warm_start_forward_level * (1. - self.epsilon)
            fwd_alpha, fwd_sigma = self.noise_schedule_funcs["forward"](
                torch.ones((1,), device=self.device) * warm_start_forward_level, **(self.noise_schedule_params or {}))
            xt = warm_start_reference * fwd_alpha + fwd_sigma * torch.randn_like(warm_start_reference)
        else:
            xt = torch.randn_like(prior) * temperature
        xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
        if preserve_history:
            log["sample_history"][:, 0] = xt.cpu().numpy()

        with torch.set_grad_enabled(requires_grad):
            condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
            condition_vec_cg = condition_cg

        # ===================== Sampling Schedule ====================
        if isinstance(warm_start_reference, torch.Tensor) and warm_start_forward_level > 0.:
            t_diffusion = [self.t_diffusion[0], warm_start_forward_level]
        else:
            t_diffusion = self.t_diffusion
        if isinstance(sample_step_schedule, str):
            if sample_step_schedule in SUPPORTED_SAMPLING_STEP_SCHEDULE.keys():
                sample_step_schedule = SUPPORTED_SAMPLING_STEP_SCHEDULE[sample_step_schedule](
                    t_diffusion, sample_steps)
            else:
                raise ValueError(f"Sampling step schedule {sample_step_schedule} is not supported.")
        elif callable(sample_step_schedule):
            sample_step_schedule = sample_step_schedule(t_diffusion, sample_steps)
        else:
            raise ValueError("sample_step_schedule must be a callable or a string")

        alphas, sigmas = self.noise_schedule_funcs["forward"](
            sample_step_schedule, **(self.noise_schedule_params or {}))
        logSNRs = torch.log(alphas / sigmas)
        hs = torch.zeros_like(logSNRs)
        hs[1:] = logSNRs[:-1] - logSNRs[1:]  # hs[0] is not correctly calculated, but it will not be used.
        stds = torch.zeros((sample_steps + 1,), device=self.device)
        stds[1:] = sigmas[:-1] / sigmas[1:] * (1 - (alphas[1:] / alphas[:-1]) ** 2).sqrt()

        buffer = []

        # ===================== Denoising Loop ========================
        loop_steps = [1] * diffusion_x_sampling_steps + list(range(1, sample_steps + 1))
        for i in reversed(loop_steps):

            t = torch.full((n_samples,), sample_step_schedule[i], dtype=torch.float32, device=self.device)

            # guided sampling
            pred, logp = self.guided_sampling(
                xt, t, alphas[i], sigmas[i],
                model, condition_vec_cfg, w_cfg, condition_vec_cg, w_cg, requires_grad)

            # clip the prediction
            pred = self.clip_prediction(pred, xt, alphas[i], sigmas[i])

            # transform to eps_theta
            eps_theta = pred if self.predict_noise else xtheta_to_epstheta(xt, alphas[i], sigmas[i], pred)
            x_theta = pred if not self.predict_noise else epstheta_to_xtheta(xt, alphas[i], sigmas[i], pred)

            # one-step update
            if solver == "ddpm":
                xt = (
                        (alphas[i - 1] / alphas[i]) * (xt - sigmas[i] * eps_theta) +
                        (sigmas[i - 1] ** 2 - stds[i] ** 2 + 1e-8).sqrt() * eps_theta)
                if i > 1:
                    xt += (stds[i] * torch.randn_like(xt))

            elif solver == "ddim":
                xt = (alphas[i - 1] * ((xt - sigmas[i] * eps_theta) / alphas[i]) + sigmas[i - 1] * eps_theta)

            elif solver == "ode_dpmsolver_1":
                xt = (alphas[i - 1] / alphas[i]) * xt - sigmas[i - 1] * torch.expm1(hs[i]) * eps_theta

            elif solver == "ode_dpmsolver++_1":
                xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * x_theta

            elif solver == "ode_dpmsolver++_2M":
                buffer.append(x_theta)
                if i < sample_steps:
                    r = hs[i + 1] / hs[i]
                    D = (1 + 0.5 / r) * buffer[-1] - 0.5 / r * buffer[-2]
                    xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * D
                else:
                    xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * x_theta

            elif solver == "sde_dpmsolver_1":
                xt = ((alphas[i - 1] / alphas[i]) * xt -
                      2 * sigmas[i - 1] * torch.expm1(hs[i]) * eps_theta +
                      sigmas[i - 1] * torch.expm1(2 * hs[i]).sqrt() * torch.randn_like(xt))

            elif solver == "sde_dpmsolver++_1":
                xt = ((sigmas[i - 1] / sigmas[i]) * (-hs[i]).exp() * xt -
                      alphas[i - 1] * torch.expm1(-2 * hs[i]) * x_theta +
                      sigmas[i - 1] * (-torch.expm1(-2 * hs[i])).sqrt() * torch.randn_like(xt))

            elif solver == "sde_dpmsolver++_2M":
                buffer.append(x_theta)
                if i < sample_steps:
                    r = hs[i + 1] / hs[i]
                    D = (1 + 0.5 / r) * buffer[-1] - 0.5 / r * buffer[-2]
                    xt = ((sigmas[i - 1] / sigmas[i]) * (-hs[i]).exp() * xt -
                          alphas[i - 1] * torch.expm1(-2 * hs[i]) * D +
                          sigmas[i - 1] * (-torch.expm1(-2 * hs[i])).sqrt() * torch.randn_like(xt))
                else:
                    xt = ((sigmas[i - 1] / sigmas[i]) * (-hs[i]).exp() * xt -
                          alphas[i - 1] * torch.expm1(-2 * hs[i]) * x_theta +
                          sigmas[i - 1] * (-torch.expm1(-2 * hs[i])).sqrt() * torch.randn_like(xt))

            # fix the known portion, and preserve the sampling history
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
            if preserve_history:
                log["sample_history"][:, sample_steps - i + 1] = xt.cpu().numpy()

        # ================= Post-processing =================
        if self.classifier is not None and w_cg != 0.:
            with torch.no_grad():
                t = torch.zeros((n_samples,), dtype=torch.long, device=self.device)
                logp = self.classifier.logp(xt, t, condition_vec_cg)
            log["log_p"] = logp

        if self.clip_pred:
            xt = xt.clip(self.x_min, self.x_max)

        return xt, log
