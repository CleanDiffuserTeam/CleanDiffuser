from copy import deepcopy
from typing import Optional, Union

import pytorch_lightning as L
import torch
import torch.nn as nn

from cleandiffuser.classifier import BaseClassifier
from cleandiffuser.nn_condition import BaseNNCondition, IdentityCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import TensorDict


class DiffusionModel(L.LightningModule):
    def __init__(
        self,
        # ----------------- Neural Networks ----------------- #
        # NN backbone for the diffusion model
        nn_diffusion: BaseNNDiffusion,
        # Add a condition-process NN to enable classifier-free-guidance
        nn_condition: Optional[BaseNNCondition] = None,
        # ----------------- Masks ----------------- #
        # Fix some portion of the input data, and only allow the diffusion model to complete the rest part.
        fix_mask: Optional[torch.Tensor] = None,
        # Add loss weight
        loss_weight: Optional[torch.Tensor] = None,
        # ------------------ Plugins ---------------- #
        # Add a classifier to enable classifier-guidance
        classifier: Optional[BaseClassifier] = None,
        # ------------------ Params ---------------- #
        ema_rate: float = 0.995,
    ):
        super().__init__()
        self.ema_rate = ema_rate

        # nn_condition is None means that the model is not conditioned on any input.
        nn_condition = nn_condition or IdentityCondition()

        # Use EMA model for stable generation outcomes
        self.model = nn.ModuleDict({"diffusion": nn_diffusion, "condition": nn_condition}).train()
        self.model_ema = deepcopy(self.model).requires_grad_(False).eval()

        self.classifier = classifier

        self.fix_mask = nn.Parameter(
            fix_mask if fix_mask is not None else torch.Tensor([0.0]),
            requires_grad=False,
        )
        self.loss_weight = nn.Parameter(
            loss_weight if loss_weight is not None else torch.Tensor([1.0]),
            requires_grad=False,
        )

        self.optimizer = self.configure_optimizers()

        self.update_list = ["diffusion", "classifier"]

    def configure_optimizers(self):
        """PyTorch Lightning optimizer configuration."""
        param_group = [{"params": self.model.parameters(), "lr": 3e-4}]
        if self.classifier is not None:
            param_group.append({"params": self.classifier.model.parameters(), "lr": 3e-4})
        return torch.optim.Adam(param_group)

    def ema_update(self):
        """Update the EMA model."""
        with torch.no_grad():
            for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
                p_ema.data.mul_(self.ema_rate).add_(p.data, alpha=1.0 - self.ema_rate)

    @staticmethod
    def ema_update_schedule(batch_idx: int):
        _ = batch_idx
        return True

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
        return x0, t, eps

    def loss(self, x0: torch.Tensor, condition: Optional[Union[torch.Tensor, TensorDict]] = None):
        """Loss function.

        Args:
            x0 (torch.Tensor): Clean data sampled from the target distribution.
            condition (Optional[Union[torch.Tensor, TensorDict]]): CFG Condition. Defaults to None.

        Returns:
            loss (torch.Tensor): Loss value.
        """
        return x0.mean()

    def training_step(self, batch, batch_idx):
        """PyTorch Lightning training step.

        Training process for the diffusion model with pytorch-lightning.
        The batch should be a dictionary containing the key `x0` for the input data,
        and the key `condition_cfg` or `condition_cg` for the condition data.
        `x0` is the clean data and must be provided.
        `condition_cfg` is the CFG condition and is optional.
        `condition_cg` is the CG condition and is optional.

        Args:
            batch (dict):
                Dictionary containing "x0" and "condition_cfg" or "condition_cg".
                "x0" is the clean data and must be provided.
                "condition_cfg" is the CFG condition and is optional.
                "condition_cg" is the CG condition and is optional.
            batch_idx (int): Batch index.
        """

        assert self.update_list, "The update list should not be empty."

        assert (
            isinstance(batch, dict) and "x0" in batch.keys()
        ), "The batch should contain the key `x0` for the input data."

        x0 = batch["x0"]
        condition_cfg = batch.get("condition_cfg", None)
        condition_cg = batch.get("condition_cg", None)

        loss = 0.0

        if "diffusion" in self.update_list:
            loss_diffusion = self.loss(x0, condition_cfg)

            self.log("diffusion_loss", loss_diffusion, prog_bar=True)

            if self.ema_update_schedule(batch_idx):
                self.ema_update()

            loss += loss_diffusion

        if "classifier" in self.update_list and self.classifier is not None:
            xt, t, eps = self.add_noise(x0)

            loss_classifier = self.classifier.loss(xt, t, condition_cg)

            self.log("classifier_loss", loss_classifier, prog_bar=True)

            if self.classifier.ema_update_schedule(batch_idx):
                self.classifier.ema_update()

            loss += loss_classifier

        return loss

    def update_diffusion(
        self, x0: torch.Tensor, condition_cfg: Optional[torch.Tensor] = None, update_ema: bool = True, **kwargs
    ):
        """One-step diffusion update.

        Args:
            x0 (torch.Tensor):
                Samples from the target distribution. shape: (batch_size, *x_shape)
            condition_cfg (Optional[Union[torch.Tensor, TensorDict]]):
                Condition of x0. `None` indicates no condition. It can be a tensor or a dictionary of tensors.
                The update function will automatically handle the condition dropout as defined in NNCondition.
            update_ema (bool):
                Whether to update the EMA model.

        Returns:
            log (dict),
                The log dictionary.
        """
        loss = self.loss(x0, condition_cfg)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        if update_ema:
            self.ema_update()
        return {"diffusion_loss": loss.item()}

    def update_classifier(
        self, x0: torch.Tensor, condition_cg: Optional[torch.Tensor] = None, update_ema: bool = True, **kwargs
    ):
        xt, t, eps = self.add_noise(x0)
        log = self.classifier.update(xt, t, condition_cg, update_ema, kwargs)
        return {"classifier_loss": log["loss"]}

    def sample(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, path: str):
        torch.save(
            {
                "model": self.model.state_dict(),
                "model_ema": self.model_ema.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model_ema.load_state_dict(checkpoint["model_ema"])
