from copy import deepcopy

import pytorch_lightning as L
import torch

from cleandiffuser.nn_classifier import BaseNNClassifier


class BaseClassifier(L.LightningModule):
    """
    Basic classifier for classifier-guidance.
    Generally, the classifier predicts the logp(c|x_t, noise),
    and then uses the gradient with respect to x_t to guide the diffusion model in sampling the distribution p(x_0|c).
    """

    def __init__(
        self,
        nn_classifier: BaseNNClassifier,
        ema_rate: float = 0.995,
    ):
        super().__init__()
        self.ema_rate = ema_rate

        self.model = nn_classifier
        self.model_ema = deepcopy(self.model).requires_grad_(False).eval()

        self.optimizer = self.configure_optimizers()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=3e-4)

    def ema_update(self):
        with torch.no_grad():
            for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
                p_ema.data.mul_(self.ema_rate).add_(p.data, alpha=1.0 - self.ema_rate)

    @staticmethod
    def ema_update_schedule(batch_idx: int):
        _ = batch_idx
        return True

    def loss(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        raise NotImplementedError

    def update(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, update_ema: bool = True):
        loss = self.loss(x, t, y)
        self.optimizer.zero_grad()
        loss.backward()
        if isinstance(self.grad_clip_norm, float):
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_norm
            ).item()
        else:
            grad_norm = None
        self.optimizer.step()
        if update_ema:
            self.ema_update()
        return {"loss": loss.item(), "grad_norm": grad_norm}

    def logp(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor):
        """
        Calculate logp(c|x_t / scale, t) for classifier-guidance.

        Input:
            - x: (batch, *x_shape)
            - t: (batch, )
            - c: (batch, *c_shape)

        Output:
            - logp(c|x, t): (batch, 1)
        """
        raise NotImplementedError

    def gradients(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor):
        x.requires_grad_()
        logp = self.logp(x, t, c)
        grad = torch.autograd.grad([logp.sum()], [x])[0]
        x.detach()
        return logp.detach(), grad.detach()

    def save(self, path):
        torch.save(
            {"model": self.model.state_dict(), "model_ema": self.model_ema.state_dict()}, path
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model_ema.load_state_dict(checkpoint["model_ema"])


# class CategoricalClassifier(BasicClassifier):
#     """
#     CategoricalClassifier is used for finite discrete conditional sets.
#     In this case, the training of the classifier can be transformed into a classification task.
#     """
#     def __init__(self, nn_classifier: nn.Module):
#         super().__init__()
#
#     def logp(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, scale: Union[torch.Tensor, float] = 1.) -> torch.Tensor:
#         """
#         Calculate logp(c|x_t / scale, t) for classifier-guidance.
#
#         Input:
#             - x:         (batch, *x_shape)
#             - t:         (batch, *t_shape)
#             - c:         (batch, *c_shape)
#             - scale      (batch, *x_shape) or float
#
#         Output:
#             - logp(c|x / scale, t): (batch, 1)
#         """
#         raise NotImplementedError
