from typing import Callable, List
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from cleandiffuser.diffusion import DiffusionModel
from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import at_least_ndim
from .newedm import ContinuousEDM


def erf(x):
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y


def compare_properties(obj1, obj2, properties: List[str]):
    differences = []
    for prop in properties:
        obj1_prop = getattr(obj1, prop)
        obj2_prop = getattr(obj2, prop)
        if isinstance(obj1_prop, torch.Tensor):
            if not torch.allclose(obj1_prop, obj2_prop):
                differences.append(prop)
        elif isinstance(obj1_prop, np.ndarray):
            if not np.allclose(obj1_prop, obj2_prop):
                differences.append(prop)
        else:
            if obj1_prop != obj2_prop:
                differences.append(prop)
    return differences


def pseudo_huber_loss(source: torch.Tensor, target: torch.Tensor, c: float = 0.0):
    return ((source - target) ** 2 + c ** 2).sqrt() - c


class CMCurriculumLogger:
    def __init__(
            self, s0: int = 10, s1: int = 1280, curriculum_cycle: int = 100_000,
            sigma_min: float = 0.002, sigma_max: float = 80., rho: float = 7.,
            P_mean: float = -1.1, P_std: float = 2.0
    ):
        self.Kprime = np.ceil(curriculum_cycle / (np.log2(np.ceil(s1 / s0)) + 1))
        self.Nk = s0
        self.s0, self.s1 = s0, s1
        self.curriculum_cycle = curriculum_cycle
        self.sigma_min, self.sigma_max, self.rho = sigma_min, sigma_max, rho
        self.P_mean, self.P_std = P_mean, P_std

        self.ceil_k_div_Kprime, self.k = None, None

        self.update_k(0)

    def update_k(self, k):
        self.k = k
        if np.ceil(k / self.Kprime) != self.ceil_k_div_Kprime:
            self.ceil_k_div_Kprime = np.ceil(k / self.Kprime)
            self.Nk = int(min(self.s0 * (2 ** self.ceil_k_div_Kprime), self.s1))

            self.sigmas = ((self.sigma_min ** (1 / self.rho) + np.arange(self.Nk + 1, dtype=np.float32)
                            / self.Nk * (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (
                            1 / self.rho))) ** self.rho)

            self.p_sigmas = (erf((np.log(self.sigmas[1:]) - self.P_mean) / (self.P_std * (2 ** 0.5))) -
                             erf((np.log(self.sigmas[:-1]) - self.P_mean) / (self.P_std * (2 ** 0.5))))
            self.p_sigmas = self.p_sigmas / self.p_sigmas.sum()

    def incremental_update_k(self):
        self.update_k(self.k + 1)

    @property
    def curriculum_process(self):
        return (self.k % self.curriculum_cycle) / self.curriculum_cycle


class ContinuousConsistencyModel(DiffusionModel):
    """**Continuous-time Consistency Model**
    
    The Consistency Model defines a consistency function. 
    A consistency function has the property of self-consistency: 
    its outputs are consistent for arbitrary pairs of (x_t, t) that belong to the same PF ODE trajectory. 
    To learn such a consistency function, the Consistency Model needs to be distilled either from a pre-trained EDM 
    or learned directly through consistency training loss.
    This self-consistency property allows the Consistency Model in theory to achieve one-step generation.

    The current implementation of Consistency Model only supports continuous-time ODEs.
    The sampling steps are required to be greater than 0.

    Args:
        nn_diffusion: BaseNNDiffusion,
            The neural network backbone for the Diffusion model.
        nn_condition: Optional[BaseNNCondition],
            The neural network backbone for the condition embedding.
        
        fix_mask: Union[list, np.ndarray, torch.Tensor],
            Fix some portion of the input data, and only allow the diffusion model to complete the rest part.
            The mask should be in the shape of `x_shape`.
        loss_weight: Union[list, np.ndarray, torch.Tensor],
            Add loss weight. The weight should be in the shape of `x_shape`.
        
        classifier: Optional[BaseClassifier],
            The Consistency Model does not support classifier guidance; please set this option to `None`.
        
        grad_clip_norm: Optional[float],
            Gradient clipping norm.
        ema_rate: float,
            Exponential moving average rate.
        optim_params: Optional[dict],
            Optimizer parameters.
        
        s0: int,
            The minimum number of noise levels. Default: 10.
        s1: int,
            The maximum number of noise levels. Default: 1280.
        data_dim: int,
            The dimension of the data, which affects the `pseudo_huber_constant`.
            As suggested in `improved Consistency Models`, `pseudo_huber_constant` = 0.00054 * np.sqrt(data_dim).
            If `data_dim` is `None`, then `pseudo_huber_constant` = 0.01 will be used.
        P_mean: float,
            Hyperparameter for noise sampling during training. Default: -1.1.
        P_std: float,
            Hyperparameter for noise sampling during training. Default: 2.0.
        sigma_min: float,
            The minimum standard deviation of the noise. Default: 0.002.
        sigma_max: float,
            The maximum standard deviation of the noise. Default: 80.
        sigma_data: float,
            The standard deviation of the data. Default: 0.5.
        rho: float,
            The power of the noise schedule. Default: 7.
        curriculum_cycle: int,
            The cycle of the curriculum process.
            It is best to set `curriculum_cycle` to the number of model training iterations. Default: 100_000.
    
        x_max: Optional[torch.Tensor],
            The maximum value for the input data. `None` indicates no constraint.
        x_min: Optional[torch.Tensor],
            The minimum value for the input data. `None` indicates no constraint.
        
        device: Union[torch.device, str],
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
            # Do not support CG
            classifier=None,

            # ------------------ Training Params ---------------- #
            grad_clip_norm: Optional[float] = None,
            ema_rate: float = 0.9999,
            optim_params: Optional[dict] = None,

            # ------------------- Consistency Model Params ------------------- #
            s0: int = 10,
            s1: int = 1280,
            data_dim: int = None,
            P_mean: float = -1.1,
            P_std: float = 2.0,
            sigma_min: float = 0.002,
            sigma_max: float = 80.,
            sigma_data: float = 0.5,
            rho: float = 7.0,
            curriculum_cycle: int = 100_000,

            x_max: Optional[torch.Tensor] = None,
            x_min: Optional[torch.Tensor] = None,

            device: Union[torch.device, str] = "cpu"
    ):
        super().__init__(
            nn_diffusion, nn_condition, fix_mask, loss_weight, classifier, grad_clip_norm,
            0, ema_rate, optim_params, device)

        self.cur_logger = CMCurriculumLogger(
            s0, s1, curriculum_cycle, sigma_min, sigma_max, rho, P_mean, P_std)

        self.pseudo_huber_constant = 0.01 if data_dim is None else 0.00054 * np.sqrt(data_dim)

        self.rho = rho
        self.sigma_data, self.sigma_max, self.sigma_min = sigma_data, sigma_max, sigma_min
        self.x_max = x_max.to(device) if isinstance(x_max, torch.Tensor) else x_max
        self.x_min = x_min.to(device) if isinstance(x_min, torch.Tensor) else x_min

        self.edm = None
        self.distillation_sigmas, self.distillation_N = None, None

    def prepare_distillation(self, edm: ContinuousEDM, distillation_N: int = 18):
        checklist = [
            "sigma_data", "sigma_max", "sigma_min", "rho", "x_max", "x_min",
            "fix_mask", "loss_weight", "device"]
        differences = compare_properties(self, edm, checklist)
        if len(differences) != 0:
            raise ValueError(f"Properties {differences} are different between the EDM and the Consistency Model.")
        self.edm = edm
        self.model.load_state_dict(edm.model.state_dict())
        self.model_ema.load_state_dict(edm.model_ema.state_dict())
        self.distillation_N = distillation_N
        self.distillation_sigmas = self.training_noise_schedule(distillation_N)

    @property
    def supported_solvers(self):
        return ["none"]

    @property
    def clip_pred(self):
        return (self.x_max is not None) or (self.x_min is not None)

    def training_noise_schedule(self, N):
        sigma = ((self.sigma_min ** (1 / self.rho) + np.arange(N + 1)
                  / N * (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))) ** self.rho)
        return torch.tensor(sigma, device=self.device, dtype=torch.float32)

    # ===================== CM Pre-conditioning =======================
    def c_skip(self, sigma):
        return self.sigma_data ** 2 / (self.sigma_data ** 2 + (sigma - self.sigma_min) ** 2)

    def c_out(self, sigma):
        return (sigma - self.sigma_min) * self.sigma_data / (self.sigma_data ** 2 + sigma ** 2).sqrt()

    def c_in(self, sigma):
        return 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()

    def c_noise(self, sigma):
        return 0.25 * sigma.log()

    def f(self, x, t, condition=None, model=None):
        c_skip, c_out, c_in, c_noise = self.c_skip(t), self.c_out(t), self.c_in(t), self.c_noise(t)
        if model is None:
            model = self.model
        c_skip, c_in, c_out = (
            at_least_ndim(c_skip, x.dim()), at_least_ndim(c_in, x.dim()), at_least_ndim(c_out, x.dim()))
        pred_x = c_skip * x + c_out * model["diffusion"](c_in * x, c_noise, condition)
        if self.clip_pred:
            pred_x = pred_x.clip(self.x_min, self.x_max)
        return pred_x

    def distillation_loss(self, x0, condition=None):

        assert self.edm is not None, "Please call `prepare_distillation` before distillation."

        idx = torch.randint(self.distillation_N, (x0.shape[0],), device=self.device)
        t_m, t_n = self.distillation_sigmas[idx + 1], self.distillation_sigmas[idx]
        x_m, t_m, eps = self.edm.add_noise(x0, t_m, None)

        with torch.no_grad():
            condition_vec_cfg = self.edm.model_ema["condition"](condition) if condition is not None else None
            pred, logp = self.edm.guided_sampling(
                x_m, t_m, None, self.edm.model_ema, condition_vec_cfg, 1.0, None, 0.0, False)
            dot_x = (x_m - pred) / at_least_ndim(t_m, x_m.dim())
            delta_t = t_m - t_n
            x_n = x_m - dot_x * at_least_ndim(delta_t, x_m.dim())
            x_n = x_n * (1. - self.fix_mask) + x0 * self.fix_mask

        condition_vec = self.model["condition"](condition) if condition is not None else None
        pred_x_m = self.f(x_m, t_m, condition_vec, self.model)
        with torch.no_grad():
            condition_vec_ema = self.model_ema["condition"](condition) if condition is not None else None
            pred_x_n = self.f(x_n, t_n, condition_vec_ema, self.model_ema)

        loss = (((pred_x_n - pred_x_m) ** 2) * (1 - self.fix_mask) *
                self.loss_weight * at_least_ndim((1 / (t_m - t_n)), pred_x_n.dim()))

        return loss.mean(), None

    def training_loss(self, x0, condition=None):

        idx = np.random.choice(self.cur_logger.Nk, size=x0.shape[0], p=self.cur_logger.p_sigmas)

        # m = n + 1
        sigma_n = torch.tensor(self.cur_logger.sigmas[idx], device=self.device)
        sigma_m = torch.tensor(self.cur_logger.sigmas[idx + 1], device=self.device)

        eps = torch.randn_like(x0)

        x_n = x0 + at_least_ndim(sigma_n, x0.dim()) * eps
        x_m = x0 + at_least_ndim(sigma_m, x0.dim()) * eps

        condition = self.model["condition"](condition) if condition is not None else None

        pred_x_m = self.f(x_m, sigma_m, condition, self.model)
        with torch.no_grad():
            pred_x_n = self.f(x_n, sigma_n, condition.detach(), self.model)

        loss = pseudo_huber_loss(pred_x_m, pred_x_n, self.pseudo_huber_constant)

        unweighted_loss = (loss * (1 - self.fix_mask) * self.loss_weight)

        cm_loss_weight = at_least_ndim(1 / (sigma_m - sigma_n), x0.dim())

        return (unweighted_loss * cm_loss_weight).mean(), unweighted_loss.mean().item()

    def update(self, x0, condition=None, update_ema=True, loss_type="training", **kwargs):
        """ One-step gradient update.

        Args:
            x0: torch.Tensor,
                Samples from the target distribution.
            condition: Optional,
                Condition of x0. `None` indicates no condition.
            update_ema: bool,
                Whether to update the exponential moving average model.
            loss_type: str,
                The type of loss. `training` or `distillation`.

        Returns:
            log: dict,
                The log dictionary.

        Examples:
            >>> model = ContinuousConsistencyModel(...)
            >>> x0 = torch.randn(*x_shape)
            >>> condition = torch.randn(*condition_shape)
            >>> log = model.update(x0, condition, loss_type="training")  # training
            >>> log = model.update(x0, condition, loss_type="distillation")  # distillation
        """
        if loss_type == "training":
            loss, unweighted_loss = self.training_loss(x0, condition)
        elif loss_type == "distillation":
            loss, unweighted_loss = self.distillation_loss(x0, condition)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm) \
            if self.grad_clip_norm else None
        self.optimizer.step()
        self.optimizer.zero_grad()

        if update_ema:
            self.ema_update()

        if loss_type == "training":
            self.cur_logger.incremental_update_k()

        log = {"loss": loss.item(), "grad_norm": grad_norm, "unweighted_loss": unweighted_loss}

        return log

    def sample(
            self,
            # ---------- the known fixed portion ---------- #
            prior: torch.Tensor,
            # ----------------- sampling ----------------- #
            solver: str = "none",
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
        assert w_cg == 0.0 and condition_cg is None, "Consistency Distillation does not support classifier guidance."

        log = {
            "sample_history": np.empty((n_samples, sample_steps + 1, *prior.shape)) if preserve_history else None, }

        model = self.model if not use_ema else self.model_ema

        prior = prior.to(self.device)
        xt = torch.randn_like(prior) * self.sigma_max * temperature
        xt = xt * (1. - self.fix_mask) + prior * self.fix_mask

        if preserve_history:
            log["sample_history"][:, 0] = xt.cpu().numpy()

        with torch.set_grad_enabled(requires_grad):
            condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None

        # ===================== Sampling Schedule ====================
        sample_step_schedule = ((self.sigma_min ** (1 / self.rho) + torch.arange(sample_steps + 1, device=self.device)
                                 / sample_steps * (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho)))
                                ** self.rho)

        t = torch.full((n_samples,), sample_step_schedule[-1], dtype=torch.float32, device=self.device)
        pred_x = self.f(xt, t, condition_vec_cfg, model)
        pred_x = pred_x * (1. - self.fix_mask) + prior * self.fix_mask

        loop_steps = [1] * diffusion_x_sampling_steps + list(range(1, sample_steps))

        for i in reversed(loop_steps):
            t = torch.full((n_samples,), sample_step_schedule[i], dtype=torch.float32, device=self.device)
            xt = pred_x + (at_least_ndim(t, xt.dim()) ** 2 - self.sigma_min ** 2).sqrt() * torch.randn_like(xt)

            pred_x = self.f(xt, t, condition_vec_cfg, model)
            pred_x = pred_x * (1. - self.fix_mask) + prior * self.fix_mask

        return pred_x, log
