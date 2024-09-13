from typing import Callable, Optional, Union

import einops
import torch
import torch.nn as nn

from cleandiffuser.classifier import BaseClassifier
from cleandiffuser.diffusion.basic import DiffusionModel
from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import (
    TensorDict,
    at_least_ndim,
    concat_zeros,
    dict_apply,
    get_sampling_scheduler,
)


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
        nn_diffusion (BaseNNDiffusion):
            The neural network backbone for the Diffusion model.

        nn_condition (Optional[BaseNNCondition]):
            The neural network backbone for the condition embedding.

        fix_mask (Optional[torch.Tensor]):
            Fix some portion of the input data, and only allow the diffusion model to complete the rest part.
            The mask should be in the shape of `x_shape`.

        loss_weight (Optional[torch.Tensor]):
            Add loss weight. The weight should be in the shape of `x_shape`.

        classifier (Optional[BaseClassifier]):
            Add a classifier to enable classifier-guidance.

        ema_rate (float):
            EMA rate for ema model.

        x_max (Optional[torch.Tensor]):
            The maximum value of the input data.

        x_min (Optional[torch.Tensor]):
            The minimum value of the input data.

        diffusion_steps (int):
            The number of diffusion steps.

        discretization (Union[str, Callable]):
            The discretization method.
    """

    def __init__(
        self,
        # ----------------- Neural Networks ----------------- #
        nn_diffusion: BaseNNDiffusion,
        nn_condition: Optional[BaseNNCondition] = None,
        # ----------------- Masks ----------------- #
        # Fix some portion of the input data, and only allow the diffusion model to complete the rest part.
        fix_mask: Optional[torch.Tensor] = None,
        # Add loss weight
        loss_weight: Optional[torch.Tensor] = None,
        # ------------------ Plugins ---------------- #
        # Add a classifier to enable classifier-guidance
        classifier: Optional[BaseClassifier] = None,
        # ------------------ Training Params ---------------- #
        ema_rate: float = 0.995,
        optimizer_params: Optional[dict] = None,
        # ------------------- Diffusion Params ------------------- #
        x_max: Optional[torch.Tensor] = None,
        x_min: Optional[torch.Tensor] = None,
        diffusion_steps: int = 1000,
        discretization: Union[str, Callable] = "uniform",
    ):
        super().__init__(nn_diffusion, nn_condition, fix_mask, loss_weight, classifier, ema_rate, optimizer_params)

        assert classifier is None, "Rectified Flow does not support classifier-guidance."
        self.diffusion_steps = diffusion_steps

        self.x_max = nn.Parameter(x_max, requires_grad=False) if x_max is not None else None
        self.x_min = nn.Parameter(x_min, requires_grad=False) if x_min is not None else None

    @property
    def supported_solvers(self):
        return ["euler"]

    @property
    def clip_pred(self):
        return (self.x_max is not None) or (self.x_min is not None)

    # ==================== Training: Straighten Flow ======================

    def add_noise(
        self,
        x0: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        eps: Optional[torch.Tensor] = None,
    ):
        """Map x0 to xt.

        Args:
            x0 (torch.Tensor): Clean data.
            t (torch.Tensor): Diffusion timestep. Defaults to None.
            eps (torch.Tensor, optional): Noise. Defaults to None.

        Returns:
            xt (torch.Tensor): Noisy data.
            t (torch.Tensor): Diffusion timestep.
            eps (torch.Tensor): Noise.
        """
        t = torch.randint(1, self.diffusion_steps + 1, (x0.shape[0],), device=self.device) if t is None else t
        eps = torch.randn_like(x0) if eps is None else eps

        t_c = t / self.diffusion_steps

        xt = t_c * eps + (1 - t_c) * x0
        xt = xt * (1.0 - self.fix_mask) + x0 * self.fix_mask

        return xt, t, eps

    def loss(
        self,
        x0: torch.Tensor,
        condition: Optional[Union[torch.Tensor, TensorDict]] = None,
        x1: Optional[torch.Tensor] = None,
    ):
        # x1 is the samples from source distribution.
        # If x1 is None, then we assume x1 is from a standard Gaussian distribution.
        if x1 is None:
            x1 = torch.randn_like(x0)
        else:
            assert x0.shape == x1.shape, "x0 and x1 must have the same shape"

        xt, t, _ = self.add_noise(x0, eps=x1)

        condition = self.model["condition"](condition) if condition is not None else None

        loss = (self.model["diffusion"](xt, t, condition) - (x0 - x1)) ** 2

        return (loss * self.loss_weight * (1 - self.fix_mask)).mean()

    def update_diffusion(
        self,
        x0: torch.Tensor,
        condition_cfg: Optional[torch.Tensor] = None,
        update_ema: bool = True,
        x1: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        return super().update_diffusion(x0, condition_cfg, update_ema, x1=x1)

    def training_step(self, batch, batch_idx):
        """PyTorch Lightning training step.

        Training process for the diffusion model with pytorch-lightning.
        The batch should be a dictionary containing the key `x0` for the input data,
        and the key `condition_cfg` for the condition data.
        `x0` is the clean data and must be provided.
        `condition_cfg` is the CFG condition and is optional.

        Args:
            batch (dict):
                Dictionary containing "x0", "condition_cfg" and "x1".
                "x0" is the clean data and must be provided.
                "condition_cfg" is the CFG condition and is optional.
                "x1" is the data from source distribution and is optional.
            batch_idx (int): Batch index.
        """
        assert (
            isinstance(batch, dict) and "x0" in batch.keys()
        ), "The batch should contain the key `x0` for the input data."

        x0 = batch["x0"]
        condition_cfg = batch.get("condition_cfg", None)
        x1 = batch.get("x1", None)

        loss = 0.0

        loss_diffusion = self.loss(x0, condition_cfg, x1=x1)

        self.log("diffusion_loss", loss_diffusion, prog_bar=True)

        if self.ema_update_schedule(batch_idx):
            self.ema_update()

        loss += loss_diffusion

        return loss

    # ==================== Sampling: Solving a straight ODE flow ======================

    def sample(
        self,
        # ---------- the known fixed portion ---------- #
        prior: torch.Tensor,
        x1: Optional[torch.Tensor] = None,
        # ----------------- sampling ----------------- #
        solver: str = "euler",
        sample_steps: int = 5,
        sampling_schedule: str = "linear",
        sampling_schedule_params: Optional[dict] = None,
        use_ema: bool = True,
        temperature: float = 1.0,
        # ------------------ guidance ------------------ #
        condition_cfg: Optional[Union[torch.Tensor, TensorDict]] = None,
        mask_cfg: Optional[Union[torch.Tensor, TensorDict]] = None,
        w_cfg: float = 0.0,
        condition_cg: None = None,
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

        Args:
            priors (torch.Tensor):
                The known fixed portion of the input data. Should be in the shape of generated data.
                Use `torch.zeros((n_samples, *x_shape))` for non-prior sampling.
            x1 (Optional[torch.Tensor]):
                The samples from the source distribution. `None` indicates standard Gaussian samples.
            solver (str):
                The solver for solving the ODE.
            n_samples (int):
                The number of samples to generate.
            sample_steps (int):
                The number of sampling steps. Should be greater than 1 and less than or equal to the number of diffusion steps.
            sample_step_schedule (Union[str, Callable]):
                The schedule for the sampling steps.
            use_ema (bool):
                Whether to use the exponential moving average model.
            temperature (float):
                The temperature for sampling.

            condition_cfg (Optional[Union[torch.Tensor, TensorDict]]):
                Condition for Classifier-free-guidance.
            mask_cfg (Optional[Union[torch.Tensor, TensorDict]]):
                Mask for Classifier-guidance.
            w_cfg (float):
                Weight for Classifier-free-guidance.
            condition_cg (None):
                Condition for Classifier-guidance. Since Rectified Flow does not support classifier-guidance, it is a dummy argument.
            w_cg (float):
                Weight for Classifier-guidance. Since Rectified Flow does not support classifier-guidance, it is a dummy argument.

            diffusion_x_sampling_steps (int):
                The number of diffusion steps for diffusion-X sampling.
            warm_start_reference (Optional[torch.Tensor]):
                The reference data for warm-starting.
            warm_start_forward_level (float):
                The forward level for warm-starting.

            requires_grad (bool):
                Whether to require gradients for the model.
            preserve_history (bool):
                Whether to preserve the history of the model.
        """
        assert solver in self.supported_solvers, f"Solver {solver} is not supported."
        assert w_cg == 0.0 and condition_cg is None, "Rectified Flow does not support classifier-guidance."

        # ===================== Initialization =====================
        n_samples = prior.shape[0]
        log = {"sample_history": []}

        model = self.model if not use_ema else self.model_ema

        sampling_schedule_params = sampling_schedule_params or {}
        sampling_schedule_params["T"] = self.diffusion_steps

        prior = prior.to(self.device)
        if isinstance(warm_start_reference, torch.Tensor) and 0 < warm_start_forward_level < 1:
            warm_start_reference = warm_start_reference.to(self.device)
            diffusion_steps = int(warm_start_forward_level * self.diffusion_steps)
            t_c = at_least_ndim(diffusion_steps / self.diffusion_steps, prior.dim())
            x1 = torch.randn_like(prior) * t_c + warm_start_reference * (1 - t_c)
        else:
            if x1 is None:
                x1 = torch.randn_like(prior) * temperature
            else:
                assert prior.shape == x1.shape, "prior and x1 must have the same shape"

        xt = x1
        xt = xt * (1.0 - self.fix_mask) + prior * self.fix_mask

        if preserve_history:
            log["sample_history"].append(xt.cpu().numpy())

        with torch.set_grad_enabled(requires_grad):
            condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None

        sampling_scheduler = get_sampling_scheduler(sampling_schedule, **sampling_schedule_params)
        t_schedule = sampling_scheduler(sample_steps, device=self.device, **sampling_schedule_params)
        t_schedule[1:] = t_schedule[1:].clamp(1, None)

        # ===================== Denoising Loop ========================
        loop_steps = [1] * diffusion_x_sampling_steps + list(range(1, sample_steps + 1))
        for i in reversed(loop_steps):
            t = torch.full((n_samples,), t_schedule[i], dtype=torch.long, device=self.device)

            start_t, end_t = t_schedule[i] / self.diffusion_steps, t_schedule[i - 1] / self.diffusion_steps
            delta_t = start_t - end_t

            # velocity
            with torch.set_grad_enabled(requires_grad):
                # fully conditioned prediction
                if w_cfg == 1.0:
                    vel = model["diffusion"](xt, t, condition_vec_cfg)

                # unconditional prediction
                elif w_cfg == 0.0:
                    vel = model["diffusion"](xt, t, None)

                else:
                    condition = dict_apply(condition_vec_cfg, concat_zeros, dim=0)

                    vel_all = model["diffusion"](
                        einops.repeat(xt, "b ... -> (2 b) ...", t.repeat(2)), t.repeat(2), condition
                    )

                    vel, vel_uncond = torch.chunk(vel_all, 2, dim=0)
                    vel = w_cfg * vel + (1 - w_cfg) * vel_uncond

            # one-step update
            xt = xt + delta_t * vel

            # fix the known portion, and preserve the sampling history
            xt = xt * (1.0 - self.fix_mask) + prior * self.fix_mask
            if preserve_history:
                log["sample_history"].append(xt.cpu().numpy())

        # ================= Post-processing =================
        if self.clip_pred:
            xt = xt.clip(self.x_min, self.x_max)

        log["t_schedule"] = t_schedule

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
        fix_mask: Optional[torch.Tensor] = None,
        # Add loss weight
        loss_weight: Optional[torch.Tensor] = None,
        # ------------------ Plugins ---------------- #
        # Add a classifier to enable classifier-guidance
        classifier: Optional[BaseClassifier] = None,
        # ------------------ Training Params ---------------- #
        ema_rate: float = 0.995,
        optimizer_params: Optional[dict] = None,
        # ------------------- Diffusion Params ------------------- #
        x_max: Optional[torch.Tensor] = None,
        x_min: Optional[torch.Tensor] = None,
    ):
        super().__init__(nn_diffusion, nn_condition, fix_mask, loss_weight, classifier, ema_rate, optimizer_params)

        assert classifier is None, "Rectified Flow does not support classifier-guidance."

        self.x_max = nn.Parameter(x_max, requires_grad=False) if x_max is not None else None
        self.x_min = nn.Parameter(x_min, requires_grad=False) if x_min is not None else None

    @property
    def supported_solvers(self):
        return ["euler"]

    @property
    def clip_pred(self):
        return (self.x_max is not None) or (self.x_min is not None)

    # ==================== Training: Straighten Flow ======================

    def add_noise(
        self,
        x0: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        eps: Optional[torch.Tensor] = None,
    ):
        """Map x0 to xt.

        Args:
            x0 (torch.Tensor): Clean data.
            t (torch.Tensor): Diffusion timestep. Defaults to None.
            eps (torch.Tensor, optional): Noise. Defaults to None.

        Returns:
            xt (torch.Tensor): Noisy data.
            t (torch.Tensor): Diffusion timestep.
            eps (torch.Tensor): Noise.
        """
        t = torch.rand((x0.shape[0],), device=self.device) if t is None else t
        eps = torch.randn_like(x0) if eps is None else eps

        xt = x0 + at_least_ndim(t, x0.dim()) * (eps - x0)
        xt = xt * (1.0 - self.fix_mask) + x0 * self.fix_mask

        return xt, t, eps

    def loss(
        self,
        x0: torch.Tensor,
        condition: Optional[Union[torch.Tensor, TensorDict]] = None,
        x1: Optional[torch.Tensor] = None,
    ):
        # x1 is the samples of source distribution.
        # If x1 is None, then we assume x1 is from a standard Gaussian distribution.
        if x1 is None:
            x1 = torch.randn_like(x0)
        else:
            assert x0.shape == x1.shape, "x0 and x1 must have the same shape"

        xt, t, _ = self.add_noise(x0, eps=x1)

        condition = self.model["condition"](condition) if condition is not None else None

        loss = (self.model["diffusion"](xt, t, condition) - (x0 - x1)) ** 2

        return (loss * self.loss_weight * (1 - self.fix_mask)).mean()

    def update_diffusion(
        self,
        x0: torch.Tensor,
        condition_cfg: Optional[torch.Tensor] = None,
        update_ema: bool = True,
        x1: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        return super().update_diffusion(x0, condition_cfg, update_ema, x1=x1)

    def training_step(self, batch, batch_idx):
        """PyTorch Lightning training step.

        Training process for the diffusion model with pytorch-lightning.
        The batch should be a dictionary containing the key `x0` for the input data,
        and the key `condition_cfg` for the condition data.
        `x0` is the clean data and must be provided.
        `condition_cfg` is the CFG condition and is optional.

        Args:
            batch (dict):
                Dictionary containing "x0", "condition_cfg" and "x1".
                "x0" is the clean data and must be provided.
                "condition_cfg" is the CFG condition and is optional.
                "x1" is the data from source distribution and is optional.
            batch_idx (int): Batch index.
        """
        assert (
            isinstance(batch, dict) and "x0" in batch.keys()
        ), "The batch should contain the key `x0` for the input data."

        x0 = batch["x0"]
        condition_cfg = batch.get("condition_cfg", None)
        x1 = batch.get("x1", None)

        loss = 0.0

        loss_diffusion = self.loss(x0, condition_cfg, x1=x1)

        self.log("diffusion_loss", loss_diffusion, prog_bar=True)

        if self.ema_update_schedule(batch_idx):
            self.ema_update()

        loss += loss_diffusion

        return loss

    # ==================== Sampling: Solving a straight ODE flow ======================

    def sample(
        self,
        # ---------- the known fixed portion ---------- #
        prior: torch.Tensor,
        x1: Optional[torch.Tensor] = None,
        # ----------------- sampling ----------------- #
        solver: str = "euler",
        sample_steps: int = 5,
        sampling_schedule: str = "linear",
        sampling_schedule_params: Optional[dict] = None,
        use_ema: bool = True,
        temperature: float = 1.0,
        # ------------------ guidance ------------------ #
        condition_cfg: Optional[Union[torch.Tensor, TensorDict]] = None,
        mask_cfg: Optional[Union[torch.Tensor, TensorDict]] = None,
        w_cfg: float = 0.0,
        condition_cg: None = None,
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

        Args:
            priors (torch.Tensor):
                The known fixed portion of the input data. Should be in the shape of generated data.
                Use `torch.zeros((n_samples, *x_shape))` for non-prior sampling.
            x1 (Optional[torch.Tensor]):
                The samples from the source distribution. `None` indicates standard Gaussian samples.
            solver (str):
                The solver for solving the ODE.
            n_samples (int):
                The number of samples to generate.
            sample_steps (int):
                The number of sampling steps. Should be greater than 0.
            sample_step_schedule (Union[str, Callable]):
                The schedule for the sampling steps.
            use_ema (bool):
                Whether to use the exponential moving average model.
            temperature (float):
                The temperature for sampling.

            condition_cfg (Optional[Union[torch.Tensor, TensorDict]]):
                Condition for Classifier-free-guidance.
            mask_cfg (Optional[Union[torch.Tensor, TensorDict]]):
                Mask for Classifier-guidance.
            w_cfg (float):
                Weight for Classifier-free-guidance.
            condition_cg (None):
                Condition for Classifier-guidance. Since Rectified Flow does not support classifier-guidance, it is a dummy argument.
            w_cg (float):
                Weight for Classifier-guidance. Since Rectified Flow does not support classifier-guidance, it is a dummy argument.

            diffusion_x_sampling_steps (int):
                The number of diffusion steps for diffusion-X sampling.
            warm_start_reference (Optional[torch.Tensor]):
                The reference data for warm-starting.
            warm_start_forward_level (float):
                The forward level for warm-starting.

            requires_grad (bool):
                Whether to require gradients for the model.
            preserve_history (bool):
                Whether to preserve the history of the model.
        """
        assert solver in self.supported_solvers, f"Solver {solver} is not supported."
        assert w_cg == 0.0 and condition_cg is None, "Rectified Flow does not support classifier-guidance."

        # ===================== Initialization =====================
        n_samples = prior.shape[0]
        log = {"sample_history": []}

        model = self.model if not use_ema else self.model_ema

        sampling_schedule_params = sampling_schedule_params or {}

        prior = prior.to(self.device)
        if isinstance(warm_start_reference, torch.Tensor) and 0.0 < warm_start_forward_level < 1.0:
            warm_start_reference = warm_start_reference.to(self.device)
            t_c = torch.ones_like(prior) * warm_start_forward_level
            x1 = torch.randn_like(prior) * t_c + warm_start_reference * (1 - t_c)
        else:
            if x1 is None:
                x1 = torch.randn_like(prior) * temperature
            else:
                assert prior.shape == x1.shape, "prior and x1 must have the same shape"

        xt = x1
        xt = xt * (1.0 - self.fix_mask) + prior * self.fix_mask
        if preserve_history:
            log["sample_history"].append(xt.cpu().numpy())

        with torch.set_grad_enabled(requires_grad):
            condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None

        sampling_scheduler = get_sampling_scheduler(sampling_schedule, **sampling_schedule_params)
        t_schedule = sampling_scheduler(sample_steps, device=self.device, **sampling_schedule_params)

        # ===================== Denoising Loop ========================
        loop_steps = [1] * diffusion_x_sampling_steps + list(range(1, sample_steps + 1))
        for i in reversed(loop_steps):
            t = torch.full((n_samples,), t_schedule[i], dtype=torch.float32, device=self.device)

            delta_t = t_schedule[i] - t_schedule[i - 1]

            # velocity
            with torch.set_grad_enabled(requires_grad):
                # fully conditioned prediction
                if w_cfg == 1.0:
                    vel = model["diffusion"](xt, t, condition_vec_cfg)

                # unconditional prediction
                elif w_cfg == 0.0:
                    vel = model["diffusion"](xt, t, None)

                else:
                    condition = dict_apply(condition_vec_cfg, concat_zeros, dim=0)

                    vel_all = model["diffusion"](
                        einops.repeat(xt, "b ... -> (2 b) ...", t.repeat(2)), t.repeat(2), condition
                    )

                    vel, vel_uncond = torch.chunk(vel_all, 2, dim=0)
                    vel = w_cfg * vel + (1 - w_cfg) * vel_uncond

            # one-step update
            xt = xt + delta_t * vel

            # fix the known portion, and preserve the sampling history
            xt = xt * (1.0 - self.fix_mask) + prior * self.fix_mask
            if preserve_history:
                log["sample_history"][:, sample_steps - i + 1] = xt.cpu().numpy()

        # ================= Post-processing =================
        if self.clip_pred:
            xt = xt.clip(self.x_min, self.x_max)

        log["t_schedule"] = t_schedule

        return xt, log


if __name__ == "__main__":
    from cleandiffuser.nn_diffusion import MlpNNDiffusion

    device = "cuda:0"

    nn_diffusion = MlpNNDiffusion(10, 16).to(device)

    prior = torch.zeros((2, 10))
    warm_start_reference = torch.zeros((2, 10))
    # diffusion = DiscreteRectifiedFlow(nn_diffusion).to(device)

    # y, log = diffusion.sample(
    #     prior,
    #     sample_steps=20,
    #     sampling_schedule="quad",
    #     warm_start_reference=warm_start_reference,
    #     warm_start_forward_level=0.64,
    # )

    diffusion = ContinuousRectifiedFlow(nn_diffusion).to(device)

    y, log = diffusion.sample(
        prior,
        sample_steps=20,
        sampling_schedule="linear",
        warm_start_reference=warm_start_reference,
        warm_start_forward_level=0.64,
    )
