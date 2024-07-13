from typing import Optional, Union, Callable

import numpy as np
import torch
import torch.nn as nn

from cleandiffuser.classifier import BaseClassifier
from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import (
    at_least_ndim,
    SUPPORTED_DISCRETIZATIONS, SUPPORTED_SAMPLING_STEP_SCHEDULE)
from .basic import DiffusionModel


class DiscreteRectifiedFlow(DiffusionModel):
    """Discrete-time Rectified Flow
    
    Rectified flow learns a straight flow for transporting between two probability distributions. 
    The straight characteristic of this flow enables ODEs to achieve high-quality data generation 
    with very few sampling steps (achieving one-step sampling when the ODE is completely straight). 
    Rectified flow can continuously optimize model performance by refining the learned ODE through multiple reflow procedures.
    Since it does not require Gaussian to be the source distribution, our implementation supports transformations 
    between two arbitrary distributions. However, if not specifically specified during training, 
    the model will default to learning the transport from a Gaussian distribution to the target distribution.
    
    The DiscreteRectifiedFlow is the discrete-time version of Rectified flow.
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
        
    - diffusion_steps: int
        The discretization steps for discrete-time diffusion models.
    - discretization: Union[str, Callable]
        The discretization method for the diffusion steps.
        
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
            diffusion_steps: int = 1000,

            discretization: Union[str, Callable] = "uniform",

            x_max: Optional[torch.Tensor] = None,
            x_min: Optional[torch.Tensor] = None,

            device: Union[torch.device, str] = "cpu"
    ):
        super().__init__(
            nn_diffusion, nn_condition, fix_mask, loss_weight, classifier, grad_clip_norm,
            diffusion_steps, ema_rate, optim_params, device)
        
        assert classifier is None, "Rectified Flow does not support classifier-guidance."

        self.x_max, self.x_min = x_max, x_min

        # ================= Discretization =================
        # - Map the continuous range [0., 1.] to the discrete range [0, T-1]
        if isinstance(discretization, str):
            if discretization in SUPPORTED_DISCRETIZATIONS.keys():
                self.t_diffusion = SUPPORTED_DISCRETIZATIONS[discretization](diffusion_steps, 0.).to(device)
            else:
                Warning(f"Discretization method {discretization} is not supported. "
                        f"Using uniform discretization instead.")
                self.t_diffusion = SUPPORTED_DISCRETIZATIONS["uniform"](diffusion_steps, 0.).to(device)
        elif callable(discretization):
            self.t_diffusion = discretization(diffusion_steps, 0.).to(device)
        else:
            raise ValueError("discretization must be a callable or a string")

    @property
    def supported_solvers(self):
        return ["euler"]

    @property
    def clip_pred(self):
        return (self.x_max is not None) or (self.x_min is not None)

    # ==================== Training: Straighten Flow ======================

    def loss(self, x0, x1=None, condition=None):

        # x1 is the samples of source distribution.
        # If x1 is None, then we assume x1 is from a standard Gaussian distribution.
        if x1 is None:
            x1 = torch.randn_like(x0)
        else:
            assert x0.shape == x1.shape, "x0 and x1 must have the same shape"

        t = torch.randint(self.diffusion_steps, (x0.shape[0],), device=self.device)
        t_c = at_least_ndim(self.t_diffusion[t], x0.dim())

        xt = t_c * x1 + (1 - t_c) * x0
        xt = xt * (1. - self.fix_mask) + x0 * self.fix_mask

        condition = self.model["condition"](condition) if condition is not None else None

        loss = (self.model["diffusion"](xt, t, condition) - (x0 - x1)) ** 2

        return (loss * self.loss_weight * (1 - self.fix_mask)).mean()

    def update(self, x0, condition=None, update_ema=True, x1=None, **kwargs):
        """One-step gradient update.
        Inputs:
        - x0: torch.Tensor
            Samples from the target distribution.
        - condition: Optional
            Condition of x0. `None` indicates no condition.
        - update_ema: bool
            Whether to update the exponential moving average model.
        - x1: torch.Tensor
            Samples from the source distribution. `None` indicates standard Gaussian samples.

        Outputs:
        - log: dict
            The log dictionary.
        """
        loss = self.loss(x0, x1, condition)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm) \
            if self.grad_clip_norm else None
        self.optimizer.step()
        self.optimizer.zero_grad()

        if update_ema:
            self.ema_update()

        log = {"loss": loss.item(), "grad_norm": grad_norm}

        return log

    # ==================== Sampling: Solving a straight ODE flow ======================

    def sample(
            self,
            # ---------- the known fixed portion ---------- #
            prior: torch.Tensor,
            x1: torch.Tensor = None,
            # ----------------- sampling ----------------- #
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
        - x1: torch.Tensor
            The samples from the source distribution. `None` indicates standard Gaussian samples.
        
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
        assert w_cg == 0.0 and condition_cg is None, "Rectified Flow does not support classifier-guidance."

        prior = prior.to(self.device)
        if isinstance(warm_start_reference, torch.Tensor):
            diffusion_steps = int(warm_start_forward_level * self.diffusion_steps)
            t_c = at_least_ndim(self.t_diffusion[diffusion_steps], prior.dim())
            x1 = torch.randn_like(prior) * t_c + warm_start_reference * (1 - t_c)
        else:
            diffusion_steps = self.diffusion_steps
            if x1 is None:
                x1 = torch.randn_like(prior) * temperature
            else:
                assert prior.shape == x1.shape, "prior and x1 must have the same shape"

        # ===================== Initialization =====================
        log = {
            "sample_history": np.empty((n_samples, sample_steps + 1, *prior.shape)) if preserve_history else None, }

        model = self.model if not use_ema else self.model_ema

        xt = x1.clone()
        xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
        if preserve_history:
            log["sample_history"][:, 0] = xt.cpu().numpy()

        with torch.set_grad_enabled(requires_grad):
            condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None

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

        # ===================== Denoising Loop ========================
        loop_steps = [1] * diffusion_x_sampling_steps + list(range(1, sample_steps + 1))
        for i in reversed(loop_steps):

            t = torch.full((n_samples,), sample_step_schedule[i], dtype=torch.long, device=self.device)

            start_t, end_t = self.t_diffusion[sample_step_schedule[i]], self.t_diffusion[sample_step_schedule[i - 1]]
            delta_t = start_t - end_t

            # velocity
            if w_cfg != 0.0 and w_cfg != 1.0 and condition_vec_cfg is not None:
                repeat_dim = [2 if i == 0 else 1 for i in range(xt.dim())]
                vel_all = model["diffusion"](
                    xt.repeat(*repeat_dim), t.repeat(2),
                    torch.cat([condition_vec_cfg, torch.zeros_like(condition_vec_cfg)], 0))
                vel_cond, vel_uncond = vel_all.chunk(2, dim=0)
                vel = w_cfg * vel_cond + (1 - w_cfg) * vel_uncond
            elif w_cfg == 0.0 or condition_vec_cfg is None:
                vel = model["diffusion"](xt, t, None)
            else:
                vel = model["diffusion"](xt, t, condition_vec_cfg)

            # one-step update
            xt = xt + delta_t * vel

            # fix the known portion, and preserve the sampling history
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
            if preserve_history:
                log["sample_history"][:, sample_steps - i + 1] = xt.cpu().numpy()

        # ================= Post-processing =================
        if self.clip_pred:
            xt = xt.clip(self.x_min, self.x_max)

        return xt, log


class ContinuousRectifiedFlow(DiffusionModel):
    """Continuous-time Rectified Flow
    
    Rectified flow learns a straight flow for transporting between two probability distributions. 
    The straight characteristic of this flow enables ODEs to achieve high-quality data generation 
    with very few sampling steps (achieving one-step sampling when the ODE is completely straight). 
    Rectified flow can continuously optimize model performance by refining the learned ODE through multiple reflow procedures.
    Since it does not require Gaussian to be the source distribution, our implementation supports transformations 
    between two arbitrary distributions. However, if not specifically specified during training, 
    the model will default to learning the transport from a Gaussian distribution to the target distribution.
    
    The ContinuousRectifiedFlow is the continuous-time version of Rectified flow.
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
            x_max: Optional[torch.Tensor] = None,
            x_min: Optional[torch.Tensor] = None,

            device: Union[torch.device, str] = "cpu"
    ):
        super().__init__(
            nn_diffusion, nn_condition, fix_mask, loss_weight, classifier, grad_clip_norm,
            0, ema_rate, optim_params, device)

        assert classifier is None, "Rectified Flow does not support classifier-guidance."

        self.x_max, self.x_min = x_max, x_min

    @property
    def supported_solvers(self):
        return ["euler"]

    @property
    def clip_pred(self):
        return (self.x_max is not None) or (self.x_min is not None)

    # ==================== Training: Straighten Flow ======================

    def loss(self, x0, x1=None, condition=None):

        # x1 is the samples of source distribution.
        # If x1 is None, then we assume x1 is from a standard Gaussian distribution.
        if x1 is None:
            x1 = torch.randn_like(x0)
        else:
            assert x0.shape == x1.shape, "x0 and x1 must have the same shape"

        t = torch.rand((x0.shape[0],), device=self.device)

        xt = x0 + at_least_ndim(t, x0.dim()) * (x1 - x0)

        xt = xt * (1. - self.fix_mask) + x0 * self.fix_mask

        condition = self.model["condition"](condition) if condition is not None else None

        loss = (self.model["diffusion"](xt, t, condition) - (x0 - x1)) ** 2

        return (loss * self.loss_weight * (1 - self.fix_mask)).mean()

    def update(self, x0, condition=None, update_ema=True, x1=None, **kwargs):
        """One-step gradient update.
        Inputs:
        - x0: torch.Tensor
            Samples from the target distribution.
        - condition: Optional
            Condition of x0. `None` indicates no condition.
        - update_ema: bool
            Whether to update the exponential moving average model.
        - x1: torch.Tensor
            Samples from the source distribution. `None` indicates standard Gaussian samples.

        Outputs:
        - log: dict
            The log dictionary.
        """
        loss = self.loss(x0, x1, condition)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm) \
            if self.grad_clip_norm else None
        self.optimizer.step()
        self.optimizer.zero_grad()

        if update_ema:
            self.ema_update()

        log = {"loss": loss.item(), "grad_norm": grad_norm}

        return log

    # ==================== Sampling: Solving a straight ODE flow ======================

    def sample(
            self,
            # ---------- the known fixed portion ---------- #
            prior: torch.Tensor,
            x1: torch.Tensor = None,
            # ----------------- sampling ----------------- #
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
        - x1: torch.Tensor
            The samples from the source distribution. `None` indicates standard Gaussian samples.
        
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
        assert w_cg == 0.0 and condition_cg is None, "Rectified Flow does not support classifier-guidance."

        prior = prior.to(self.device)
        if isinstance(warm_start_reference, torch.Tensor):
            t_c = torch.ones_like(prior) * warm_start_forward_level
            x1 = torch.randn_like(prior) * t_c + warm_start_reference * (1 - t_c)
        else:
            if x1 is None:
                x1 = torch.randn_like(prior) * temperature
            else:
                assert prior.shape == x1.shape, "prior and x1 must have the same shape"

        # ===================== Initialization =====================
        log = {
            "sample_history": np.empty((n_samples, sample_steps + 1, *prior.shape)) if preserve_history else None, }

        model = self.model if not use_ema else self.model_ema

        xt = x1.clone()
        xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
        if preserve_history:
            log["sample_history"][:, 0] = xt.cpu().numpy()

        with torch.set_grad_enabled(requires_grad):
            condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None

        # ===================== Sampling Schedule ====================
        if isinstance(warm_start_reference, torch.Tensor) and warm_start_forward_level > 0.:
            final_t = warm_start_forward_level
        else:
            final_t = 1.
        if isinstance(sample_step_schedule, str):
            if sample_step_schedule in SUPPORTED_SAMPLING_STEP_SCHEDULE.keys():
                sample_step_schedule = SUPPORTED_SAMPLING_STEP_SCHEDULE[sample_step_schedule](
                    [0., final_t], sample_steps)
            else:
                raise ValueError(f"Sampling step schedule {sample_step_schedule} is not supported.")
        elif callable(sample_step_schedule):
            sample_step_schedule = sample_step_schedule([0., final_t], sample_steps)
        else:
            raise ValueError("sample_step_schedule must be a callable or a string")

        # ===================== Denoising Loop ========================
        loop_steps = [1] * diffusion_x_sampling_steps + list(range(1, sample_steps + 1))
        for i in reversed(loop_steps):

            t = torch.full((n_samples,), sample_step_schedule[i], dtype=torch.float32, device=self.device)

            delta_t = sample_step_schedule[i] - sample_step_schedule[i - 1]

            # velocity
            if w_cfg != 0.0 and w_cfg != 1.0 and condition_vec_cfg is not None:
                repeat_dim = [2 if i == 0 else 1 for i in range(xt.dim())]
                vel_all = model["diffusion"](
                    xt.repeat(*repeat_dim), t.repeat(2),
                    torch.cat([condition_vec_cfg, torch.zeros_like(condition_vec_cfg)], 0))
                vel_cond, vel_uncond = vel_all.chunk(2, dim=0)
                vel = w_cfg * vel_cond + (1 - w_cfg) * vel_uncond
            elif w_cfg == 0.0 or condition_vec_cfg is None:
                vel = model["diffusion"](xt, t, None)
            else:
                vel = model["diffusion"](xt, t, condition_vec_cfg)

            # one-step update
            xt = xt + delta_t * vel

            # fix the known portion, and preserve the sampling history
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
            if preserve_history:
                log["sample_history"][:, sample_steps - i + 1] = xt.cpu().numpy()

        # ================= Post-processing =================
        if self.clip_pred:
            xt = xt.clip(self.x_min, self.x_max)

        return xt, log
