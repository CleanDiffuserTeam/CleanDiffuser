from typing import Optional, Union

import einops
import torch
import torch.nn as nn

from cleandiffuser.classifier import BaseClassifier
from cleandiffuser.diffusion.basic import DiffusionModel
from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import TensorDict, at_least_ndim, concat_zeros, dict_apply


class ContinuousEDM(DiffusionModel):
    """Continuous-time EDM

    EDM posits that the concepts of `t` in the diffusion process and the noise schedule are equivalent.
    Previous noise schedules can be interpreted as perturbing the data with Gaussian noise
    followed by scaling. EDM sets the standard deviation of noise as `t`, the scale as 1,
    and devises a series of preconditioning steps to aid in model learning.

    The current implementation of EDM only supports continuous-time ODEs.
    The sampling steps are required to be greater than 1.

    Args:
        nn_diffusion (BaseNNDiffusion):
            The neural network backbone for the Diffusion model.
        nn_condition (Optional[BaseNNCondition]):
            The neural network backbone for the condition embedding.

        fix_mask (Union[list, np.ndarray, torch.Tensor]):
            Fix some portion of the input data, and only allow the model to complete the rest parts.
            The mask should be in the shape of `x_shape` with value 1 for fixed parts and 0 for unfixed parts.
        loss_weight (Union[list, np.ndarray, torch.Tensor]):
            Loss weight. The weight should be in the shape of `x_shape`.

        classifier (Optional[BaseClassifier]):
            The classifier for classifier-guidance. Default: None.

        ema_rate (float):
            Exponential moving average rate. Default: 0.995.

        sigma_data (float):
            The standard deviation of the data. Default: 0.5.
        sigma_min (float):
            The minimum standard deviation of the noise. Default: 0.002.
        sigma_max (float):
            The maximum standard deviation of the noise. Default: 80.
        rho (float):
            The power of the noise schedule. Default: 7.
        P_mean (float):
            Hyperparameter for noise sampling during training. Default: -1.2.
        P_std (float):
            Hyperparameter for noise sampling during training. Default: 1.2.

        x_max (Optional[torch.Tensor]):
            The maximum value for the input data. `None` indicates no constraint.
        x_min (Optional[torch.Tensor]):
            The minimum value for the input data. `None` indicates no constraint.
    """

    def __init__(
        self,
        # ----------------- Neural Networks ----------------- #
        nn_diffusion: BaseNNDiffusion,
        nn_condition: Optional[BaseNNCondition] = None,
        # ----------------- Masks ----------------- #
        fix_mask: Optional[torch.Tensor] = None,
        loss_weight: Optional[torch.Tensor] = None,
        # ------------------ Plugins ---------------- #
        # Add a classifier to enable classifier-guidance
        classifier: Optional[BaseClassifier] = None,
        # ------------------ Training Params ---------------- #
        ema_rate: float = 0.995,
        optimizer_params: Optional[dict] = None,
        # ------------------- Diffusion Params ------------------- #
        sigma_data: float = 0.5,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        x_max: Optional[torch.Tensor] = None,
        x_min: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            nn_diffusion,
            nn_condition,
            fix_mask,
            loss_weight,
            classifier,
            ema_rate,
            optimizer_params,
        )

        self.sigma_data, self.sigma_min, self.sigma_max = sigma_data, sigma_min, sigma_max
        self.rho, self.P_mean, self.P_std = rho, P_mean, P_std

        self.x_max = nn.Parameter(x_max, requires_grad=False) if x_max is not None else None
        self.x_min = nn.Parameter(x_min, requires_grad=False) if x_min is not None else None

        self.t_diffusion = [sigma_min, sigma_max]

    @property
    def supported_solvers(self):
        return ["euler", "heun"]

    @property
    def clip_pred(self):
        return (self.x_max is not None) or (self.x_min is not None)

    # ===================== EDM Pre-conditioning =========================
    def c_skip(self, sigma):
        return self.sigma_data**2 / (self.sigma_data**2 + sigma**2)

    def c_out(self, sigma):
        return sigma * self.sigma_data / (self.sigma_data**2 + sigma**2).sqrt()

    def c_in(self, sigma):
        return 1 / (self.sigma_data**2 + sigma**2).sqrt()

    def c_noise(self, sigma):
        return 0.25 * sigma.log()

    def D(self, x, sigma, condition=None, model=None):
        """Prepositioning in EDM"""
        c_skip, c_out, c_in, c_noise = (
            self.c_skip(sigma),
            self.c_out(sigma),
            self.c_in(sigma),
            self.c_noise(sigma),
        )
        if model is None:
            model = self.model
        c_skip, c_in, c_out = (
            at_least_ndim(c_skip, x.dim()),
            at_least_ndim(c_in, x.dim()),
            at_least_ndim(c_out, x.dim()),
        )
        return c_skip * x + c_out * model["diffusion"](c_in * x, c_noise, condition)

    # ==================== Training: Score Matching ======================

    def add_noise(
        self,
        x0: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        eps: Optional[torch.Tensor] = None,
    ):
        """Forward process of the diffusion model.

        Args:
            x0 (torch.Tensor):
                Samples from the target distribution. shape: (batch_size, *x_shape)
            t (Optional[torch.Tensor]):
                Continuous time steps for the diffusion process. shape: (batch_size,).
                If `None`, randomly sample log(t) from Normal(`P_mean`, `P_std`).
            eps (Optional[torch.Tensor]):
                Noise for the diffusion process. shape: (batch_size, *x_shape).
                If `None`, randomly sample from Normal(0, I).

        Returns:
            xt (torch.Tensor):
                The noisy samples. shape: (batch_size, *x_shape).
            t (torch.Tensor):
                The continuous time steps. shape: (batch_size,).
            eps (torch.Tensor):
                The noise. shape: (batch_size, *x_shape).
        """
        t = (
            (torch.randn((x0.shape[0],), device=x0.device) * self.P_std + self.P_mean).exp()
            if t is None
            else t
        )
        eps = torch.randn_like(x0) if eps is None else eps

        scale = 1.0
        sigma = at_least_ndim(t, x0.dim())

        xt = scale * x0 + sigma * eps
        xt = (1.0 - self.fix_mask) * xt + self.fix_mask * x0

        return xt, t, eps

    def loss(self, x0: torch.Tensor, condition: Optional[Union[torch.Tensor, TensorDict]] = None):
        xt, t, eps = self.add_noise(x0)

        condition = self.model["condition"](condition) if condition is not None else None

        loss = (self.D(xt, t, condition) - x0) ** 2

        edm_loss_weight = at_least_ndim(
            (t**2 + self.sigma_data**2) / ((t * self.sigma_data) ** 2), x0.dim()
        )

        return (loss * self.loss_weight * (1 - self.fix_mask) * edm_loss_weight).mean()

    # ==================== Sampling: Solving SDE/ODE ======================

    def classifier_guidance(self, xt, t, sigma, model, condition=None, w: float = 1.0, pred=None):
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
            log_p, grad = self.classifier.gradients(xt.clone(), t.log() / 4.0, condition)
            pred = pred + w * (at_least_ndim(sigma, pred.dim()) ** 2) * grad

        return pred, log_p

    def classifier_free_guidance(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        model: BaseNNDiffusion,
        condition: Optional[Union[torch.Tensor, TensorDict]] = None,
        w: float = 0.0,
        pred: Optional[torch.Tensor] = None,
        pred_uncond: Optional[torch.Tensor] = None,
        requires_grad: bool = False,
    ):
        """Classifier-Free Guided Sampling.

        Add classifier-free guidance to the network prediction
        (epsilon_theta for `predict_noise` and x0_theta for not `predict_noise`).

        Args:
            xt (torch.Tensor):
                Noisy data at time `t`. (bs, *x_shape)
            t (torch.Tensor):
                Diffusion timestep. (bs, )
            model (BaseNNDiffusion):
                Diffusion model network backbone.
            condition (Optional[Union[torch.Tensor, TensorDict]]):
                CFG Condition. It is a tensor or a TensorDict that can be handled by `nn_condition`.
                Defaults to None.
            w (float):
                Guidance strength. Defaults to 0.0.
            pred (Optional[torch.Tensor]):
                Network prediction at time `t` for conditioned model. Defaults to None.
            pred_uncond (Optional[torch.Tensor]):
                Network prediction at time `t` for unconditioned model. Defaults to None.
            requires_grad (bool):
                Whether to calculate gradients. Defaults to False.

        Returns:
            pred (torch.Tensor):
                Prediction with classifier-free guidance. (bs, *x_shape)
        """
        with torch.set_grad_enabled(requires_grad):
            # fully conditioned prediction
            if w == 1.0:
                pred = self.D(xt, t, condition, model)
                pred_uncond = 0.0

            # unconditioned prediction
            elif w == 0.0:
                pred = 0.0
                pred_uncond = self.D(xt, t, None, model)

            else:
                if pred is None or pred_uncond is None:
                    condition = dict_apply(condition, concat_zeros, dim=0)

                    pred_all = self.D(
                        einops.repeat(xt, "b ... -> (2 b) ..."), t.repeat(2), condition, model
                    )

                    pred, pred_uncond = torch.chunk(pred_all, 2, dim=0)

        bar_pred = w * pred + (1 - w) * pred_uncond

        return bar_pred

    def guided_sampling(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        alpha: torch.Tensor,
        sigma: torch.Tensor,
        model: BaseNNDiffusion,
        condition_cfg: Optional[Union[torch.Tensor, TensorDict]] = None,
        w_cfg: float = 0.0,
        condition_cg: Optional[Union[torch.Tensor, TensorDict]] = None,
        w_cg: float = 0.0,
        requires_grad: bool = False,
    ):
        """
        One-step epsilon/x0 prediction with guidance.
        """

        pred = self.classifier_free_guidance(
            xt, t, model, condition_cfg, w_cfg, None, None, requires_grad
        )

        pred, logp = self.classifier_guidance(xt, t, sigma, model, condition_cg, w_cg, pred)

        return pred, logp

    def sample(
        self,
        # ---------- the known fixed portion ---------- #
        prior: torch.Tensor,
        # ----------------- sampling ----------------- #
        solver: str = "euler",  # euler or heun
        sample_steps: int = 5,
        sampling_schedule: None = None,
        sampling_schedule_params: None = None,
        use_ema: bool = True,
        temperature: float = 1.0,
        # ------------------ guidance ------------------ #
        condition_cfg: Optional[Union[torch.Tensor, TensorDict]] = None,
        mask_cfg: Optional[Union[torch.Tensor, TensorDict]] = None,
        w_cfg: float = 0.0,
        condition_cg: Optional[Union[torch.Tensor, TensorDict]] = None,
        w_cg: float = 0.0,
        # ----------- Diffusion-X sampling ----------
        diffusion_x_sampling_steps: int = 0,
        # ----------- Warm-Starting -----------
        warm_start_reference: Optional[torch.Tensor] = None,
        warm_start_forward_level: float = 0.3,
        # ------------------ others ------------------ #
        requires_grad: bool = False,
        preserve_history: bool = False,
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
        assert solver in ["euler", "heun"], (
            f"Solver {solver} is not supported. Use 'euler' or 'heun' instead."
        )

        # ===================== Initialization =====================
        n_samples = prior.shape[0]
        log = {"sample_history": []}

        model = self.model if not use_ema else self.model_ema

        prior = prior.to(self.device)
        if isinstance(warm_start_reference, torch.Tensor) and 0.0 < warm_start_forward_level < 1.0:
            warm_start_reference = warm_start_reference.to(self.device)
            fwd_sigma = (
                self.sigma_min + (self.sigma_max - self.sigma_min) * warm_start_forward_level
            )
            xt = warm_start_reference + fwd_sigma * torch.randn_like(warm_start_reference)
        else:
            fwd_sigma = self.sigma_max
            xt = torch.randn_like(prior) * self.sigma_max * temperature
        xt = xt * (1.0 - self.fix_mask) + prior * self.fix_mask

        if preserve_history:
            log["sample_history"].append(xt.cpu().numpy())

        with torch.set_grad_enabled(requires_grad):
            condition_vec_cfg = (
                model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
            )
            condition_vec_cg = condition_cg

        # ===================== Sampling Schedule ====================
        t_schedule = (
            self.sigma_min ** (1 / self.rho)
            + torch.arange(sample_steps + 1, device=self.device)
            / sample_steps
            * (fwd_sigma ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))
        ) ** self.rho

        sigmas = t_schedule

        # ===================== Denoising Loop ========================
        loop_steps = [1] * diffusion_x_sampling_steps + list(range(1, sample_steps + 1))
        for i in reversed(loop_steps):
            t = torch.full((n_samples,), t_schedule[i], dtype=torch.float32, device=self.device)

            # guided sampling
            pred, logp = self.guided_sampling(
                xt,
                t,
                0,
                sigmas[i],
                model,
                condition_vec_cfg,
                w_cfg,
                condition_vec_cg,
                w_cg,
                requires_grad,
            )

            # clip the prediction
            if self.clip_pred:
                pred = pred.clip(self.x_min, self.x_max)

            # one-step update
            dot_x = (xt - pred) / at_least_ndim(sigmas[i], xt.dim())
            delta_t = t_schedule[i] - t_schedule[i - 1]
            x_next = xt - dot_x * delta_t
            x_next = x_next * (1.0 - self.fix_mask) + prior * self.fix_mask

            if solver == "heun" and i > 1:
                pred, logp = self.guided_sampling(
                    x_next,
                    t / t_schedule[i] * t_schedule[i - 1],
                    0,
                    sigmas[i - 1],
                    model,
                    condition_vec_cfg,
                    w_cfg,
                    condition_vec_cg,
                    w_cg,
                    requires_grad,
                )
                if self.clip_pred:
                    pred = pred.clip(self.x_min, self.x_max)
                dot_x_prime = (x_next - pred) / at_least_ndim(sigmas[i - 1], xt.dim())
                x_next = xt - (dot_x + dot_x_prime) / 2.0 * delta_t
                x_next = x_next * (1.0 - self.fix_mask) + prior * self.fix_mask

            xt = x_next

            if preserve_history:
                log["sample_history"].append(xt.cpu().numpy())

        # ================= Post-processing =================
        if self.classifier is not None:
            with torch.no_grad():
                t = torch.ones((n_samples,), dtype=torch.long, device=self.device) * self.sigma_min
                logp = self.classifier.logp(xt, t.log() / 4.0, condition_vec_cg)
            log["log_p"] = logp

        if self.clip_pred:
            xt = xt.clip(self.x_min, self.x_max)

        log["sampling_schedule"] = t_schedule
        log["sigma"] = sigmas

        return xt, log


if __name__ == "__main__":
    from cleandiffuser.nn_diffusion import MlpNNDiffusion

    device = "cuda:0"

    nn_diffusion = MlpNNDiffusion(10, 16).to(device)

    prior = torch.zeros((2, 10))
    warm_start_reference = torch.zeros((2, 10))
    diffusion = ContinuousEDM(nn_diffusion).to(device)

    y, log = diffusion.sample(
        prior,
        sample_steps=20,
        warm_start_reference=warm_start_reference,
        warm_start_forward_level=0.64,
    )
