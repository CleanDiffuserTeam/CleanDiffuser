from typing import Dict

import torch
import torch.nn.functional as F

from .base import BaseClassifier


class QGPOClassifier(BaseClassifier):
    """
    QGPOClassifier defines logp(y | x, t) using negative MSE.
    Assuming nn_classifier is a NN used to predict y through x and t, i.e, pred_y = nn_classifier(x, t),
    logp is defined as - temperature * MSE(nn_classifier(x, t), y).
    """

    def loss(self, x: torch.Tensor, t: torch.Tensor, y: Dict[str, torch.Tensor]):
        """ In-support Contrastive Energy Prediction Loss in https://arxiv.org/pdf/2304.12824

        Args:
            x: torch.Tensor,
                Noisy support actions. Shape: (batch_size, K, act_dim)
            t: torch.Tensor,
                Diffusion time steps. Shape: (batch_size, )
            y: Dict[str, torch.Tensor],
                {
                    "soft_label": torch.Tensor,
                        Softmax of Q values. Shape: (batch_size, K, 1)
                    "obs": torch.Tensor,
                        Observations. Shape: (batch_size, obs_dim)}

        Returns:
            loss: torch.Tensor
        """
        b, k = x.shape[:2]

        soft_label, obs = y["soft_label"], y["obs"]

        f = self.model(x, t.unsqueeze(1).repeat(1, k), obs.unsqueeze(1).repeat(1, k, 1))

        loss = - (soft_label * F.log_softmax(f, 1)).sum(1).mean()

        with torch.no_grad():
            f_max = f.max(1)[0].mean().item()
            f_mean = f.mean().item()
            f_min = f.min(1)[0].mean().item()

        return loss, {"f_max": f_max, "f_mean": f_mean, "f_min": f_min}

    def update(self, x: torch.Tensor, noise: torch.Tensor, y: Dict[str, torch.Tensor], update_ema: bool = True):
        loss, log = self.loss(x, noise, y)
        self.optim.zero_grad()
        loss.backward()
        if isinstance(self.grad_clip_norm, float):
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm).item()
        else:
            grad_norm = None
        self.optim.step()
        if update_ema:
            self.ema_update()
        log.update({"loss": loss.item(), "grad_norm": grad_norm})
        return log

    def logp(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor):
        """
        Args:
            x: torch.Tensor,
                Noisy actions. Shape (batch_size, act_dim)
            t: torch.Tensor,
                Timestep. Shape (batch_size, )
            c: torch.Tensor,
                Observations. Shape (batch_size, obs_dim)

        Returns:
            f: torch.Tensor,
                Energy prediction. Shape (batch_size, 1)
        """
        return self.model_ema(x, t, c)
