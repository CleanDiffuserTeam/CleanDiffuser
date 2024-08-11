from copy import deepcopy
from typing import Optional

import pytorch_lightning as L
import torch
import torch.nn as nn

from cleandiffuser.classifier import BaseClassifier
from cleandiffuser.nn_condition import BaseNNCondition, IdentityCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion


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
            fix_mask: Optional[torch.Tensor] = None,  # be in the shape of `x_shape`
            # Add loss weight
            loss_weight: Optional[torch.Tensor] = None,  # be in the shape of `x_shape`

            # ------------------ Plugs ---------------- #
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
            fix_mask if fix_mask is not None else torch.Tensor([0.]), requires_grad=False)
        self.loss_weight = nn.Parameter(
            loss_weight if loss_weight is not None else torch.Tensor([1.]), requires_grad=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=3e-4)

    def ema_update(self):
        with torch.no_grad():
            for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
                p_ema.data.mul_(self.ema_rate).add_(p.data, alpha=1. - self.ema_rate)

    @staticmethod
    def ema_update_schedule(batch_idx: int):
        _ = batch_idx
        return True

    def update(self, x0, condition=None, update_ema=True, **kwargs):
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, path: str):
        torch.save({
            "model": self.model.state_dict(),
            "model_ema": self.model_ema.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model_ema.load_state_dict(checkpoint["model_ema"])
        


