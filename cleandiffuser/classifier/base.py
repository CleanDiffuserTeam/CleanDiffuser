from copy import deepcopy
from typing import Optional

import torch

from cleandiffuser.nn_classifier import BaseNNClassifier


class BaseClassifier:
    """
    Basic classifier for classifier-guidance.
    Generally, the classifier predicts the logp(c|x_t, noise),
    and then uses the gradient with respect to x_t to guide the diffusion model in sampling the distribution p(x_0|c).
    """

    def __init__(
            self,
            nn_classifier: BaseNNClassifier,
            ema_rate: float = 0.995,
            grad_clip_norm: Optional[float] = None,
            optim_params: Optional[dict] = None,
            device: str = "cpu",
    ):
        if optim_params is None:
            optim_params = {"lr": 2e-4, "weight_decay": 1e-4}
        self.device = device
        self.ema_rate, self.grad_clip_norm = ema_rate, grad_clip_norm
        self.model = nn_classifier.to(device)
        self.model_ema = deepcopy(self.model).eval()
        self.optim = torch.optim.Adam(self.model.parameters(), **optim_params)

    def eval(self):
        self.model.eval()
        self.model_ema.eval()

    def train(self):
        self.model.train()

    def ema_update(self):
        with torch.no_grad():
            for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
                p_ema.data.mul_(self.ema_rate).add_(p.data, alpha=1. - self.ema_rate)

    def loss(self, x: torch.Tensor, noise: torch.Tensor, y: torch.Tensor):
        raise NotImplementedError

    def update(self, x: torch.Tensor, noise: torch.Tensor, y: torch.Tensor, update_ema: bool = True):
        loss = self.loss(x, noise, y)
        self.optim.zero_grad()
        loss.backward()
        if isinstance(self.grad_clip_norm, float):
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm).item()
        else:
            grad_norm = None
        self.optim.step()
        if update_ema:
            self.ema_update()
        return {"loss": loss.item(), "grad_norm": grad_norm}

    def logp(self, x: torch.Tensor, noise: torch.Tensor, c: torch.Tensor):
        """
        Calculate logp(c|x_t / scale, noise) for classifier-guidance.

        Input:
            - x:         (batch, *x_shape)
            - noise:     (batch, )
            - c:         (batch, *c_shape)

        Output:
            - logp(c|x, noise): (batch, 1)
        """
        raise NotImplementedError

    def gradients(self, x: torch.Tensor, noise: torch.Tensor, c: torch.Tensor):
        x.requires_grad_()
        logp = self.logp(x, noise, c)
        grad = torch.autograd.grad([logp.sum()], [x])[0]
        x.detach()
        return logp.detach(), grad.detach()

    def save(self, path):
        torch.save({
            "model": self.model.state_dict(),
            "model_ema": self.model_ema.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model_ema.load_state_dict(checkpoint["model_ema"])

