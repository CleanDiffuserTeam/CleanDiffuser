from typing import Optional

import torch
import torch.nn as nn

from cleandiffuser.nn_classifier import BaseNNClassifier
from .base import BaseClassifier


class MSEClassifier(BaseClassifier):
    """
    MSEClassifier defines logp(y | x, t) using negative MSE.
    Assuming nn_classifier is a NN used to predict y through x and t, i.e, pred_y = nn_classifier(x, t),
    logp is defined as - temperature * MSE(nn_classifier(x, t), y).
    """
    def __init__(
            self, nn_classifier: BaseNNClassifier, temperature: float = 1.0,
            ema_rate: float = 0.995, grad_clip_norm: Optional[float] = None,
            optim_params: Optional[dict] = None, device: str = "cpu"):
        super().__init__(nn_classifier, ema_rate, grad_clip_norm, optim_params, device)
        self.temperature = temperature

    def loss(self, x: torch.Tensor, noise: torch.Tensor, y: torch.Tensor):
        pred_y = self.model(x, noise)
        return nn.functional.mse_loss(pred_y, y)

    def logp(self, x: torch.Tensor, noise: torch.Tensor, c: torch.Tensor):
        pred_y = self.model_ema(x, noise)
        return -self.temperature * ((pred_y - c) ** 2).mean(-1, keepdim=True)
