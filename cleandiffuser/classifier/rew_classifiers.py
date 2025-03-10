import torch

from cleandiffuser.classifier.base import BaseClassifier
from cleandiffuser.nn_classifier import BaseNNClassifier


class OptimalityClassifier(BaseClassifier):
    """Optimality classifier for classifier-guidance.

    The classifier predicts the logp(O|x_t, t), where O is the optimality of x_t.
    Suppose that x_t is the denoising decision-making trajectory, O is defined such that
    p(O|x_t, t) = exp(R(x_0)) / Z, where R(x_0) is the expected return of the trajectory and
    Z is a constant normalizing factor.

    Args:
        nn_classifier (BaseNNClassifier): Neural network backbone.
        ema_rate (float, optional): Exponential moving average rate. Defaults to 0.995.
    """

    def __init__(
        self,
        nn_classifier: BaseNNClassifier,
        ema_rate: float = 0.995,
    ):
        super().__init__(nn_classifier, ema_rate)

    def loss(self, x: torch.Tensor, t: torch.Tensor, R: torch.Tensor):
        """Loss function.

        Args:
            x (torch.Tensor): Noisy trajectory x_t.
            t (torch.Tensor): Diffusion timestep t.
            R (torch.Tensor): Expected return of the trajectory x_0.

        Returns:
            loss (torch.Tensor): Loss value.
        """
        pred_R = self.model(x, t, None)
        return ((pred_R - R) ** 2).mean()

    def update(self, x: torch.Tensor, t: torch.Tensor, R: torch.Tensor):
        """One-step update.

        Args:
            x (torch.Tensor): Noisy trajectory x_t.
            t (torch.Tensor): Diffusion timestep t.
            R (torch.Tensor): Expected return of the trajectory x_0.

        Returns:
            log (dict): Dictionary containing the loss value.
        """
        self.optim.zero_grad()
        loss = self.loss(x, t, R)
        loss.backward()
        self.optim.step()
        self.ema_update()
        return {"loss": loss.item()}

    def logp(self, x, t, c=None):
        return self.model_ema(x, t)
