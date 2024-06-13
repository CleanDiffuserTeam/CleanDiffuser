from typing import Optional

from cleandiffuser.nn_classifier import BaseNNClassifier
from .base import BaseClassifier


class CumRewClassifier(BaseClassifier):
    def __init__(
            self,
            nn_classifier: BaseNNClassifier,
            device: str = "cpu",
            optim_params: Optional[dict] = None,
    ):
        super().__init__(nn_classifier, 0.995, None, optim_params, device)

    def loss(self, x, noise, R):
        pred_R = self.model(x, noise, None)
        return ((pred_R - R) ** 2).mean()

    def update(self, x, noise, R):
        self.optim.zero_grad()
        loss = self.loss(x, noise, R)
        loss.backward()
        self.optim.step()
        self.ema_update()
        return {"loss": loss.item()}

    def logp(self, x, noise, c=None):
        return self.model_ema(x, noise)
