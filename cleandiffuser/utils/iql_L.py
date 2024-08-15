from copy import deepcopy

import pytorch_lightning as L
import torch
import torch.nn as nn


class TwinQ(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim: int = 256):
        super().__init__()
        self.Q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, 1))
        self.Q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, 1))

    def both(self, obs, act):
        q1, q2 = self.Q1(torch.cat([obs, act], -1)), self.Q2(torch.cat([obs, act], -1))
        return q1, q2

    def forward(self, obs, act):
        return torch.min(*self.both(obs, act))


class V(nn.Module):
    def __init__(self, obs_dim, hidden_dim: int = 256):
        super().__init__()
        self.V = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, 1))

    def forward(self, obs):
        v = self.V(obs)
        return v


class IQL(L.LightningModule):
    """ Simple Implicit Q-Learning (IQL) pytorch implementation.

    Args:
        obs_dim: int, observation space dimension.
        act_dim: int, action space dimension.
        tau: float, quantile level. Default is 0.7.
        discount: float, discount factor. Default is 0.99.
        hidden_dim: int, hidden dimension. Default is 256.

    Example:
        >>> iql = IQL(obs_dim, act_dim)
        >>> q = iql.Q(obs, act)
        >>> v = iql.V(obs)
    """
    def __init__(self, obs_dim: int, act_dim: int, tau: float = 0.7, discount: float = 0.99, hidden_dim: int = 256):
        super().__init__()
        self.save_hyperparameters()
        self.iql_tau, self.discount = tau, discount
        self.Q = TwinQ(obs_dim, act_dim, hidden_dim)
        self.Q_targ = deepcopy(self.Q).requires_grad_(False).eval()
        self.V = V(obs_dim, hidden_dim)
        
    def q(self, obs: torch.Tensor, act: torch.Tensor, use_ema: bool = False, requires_grad: bool = False):
        """ IQL Q function.
        
        Args:
            obs (torch.Tensor): Observation tensor in shape (..., obs_dim).
            act (torch.Tensor): Action tensor in shape (..., act_dim).
            use_ema (bool): Use the target network. Default is False.
            requires_grad (bool): Enable gradient computation. Default is False.
        
        Returns:
            q (torch.Tensor): Q tensor in shape (..., 1).
        """
        with torch.set_grad_enabled(requires_grad):
            if use_ema:
                q = self.Q_targ(obs, act)
            else:
                q = self.Q(obs, act)
        return q

    def v(self, obs: torch.Tensor, requires_grad: bool = False):
        """ IQL Value function.
        
        Args:
            obs (torch.Tensor): Observation tensor in shape (..., obs_dim).
            requires_grad (bool): Enable gradient computation. Default is False.
        
        Returns:
            v (torch.Tensor): Value tensor in shape (..., 1).
        """
        with torch.set_grad_enabled(requires_grad):
            v = self.V(obs)
        return v

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)

    def update_target(self, mu=0.995):
        for p, p_targ in zip(self.Q.parameters(), self.Q_targ.parameters()):
            p_targ.data = mu * p_targ.data + (1 - mu) * p.data

    def training_step(self, batch, batch_idx):

        obs, act, rew, obs_next, done = batch

        # update V
        self.Q_targ.eval()
        q = self.Q_targ(obs, act)
        v = self.V(obs)
        loss_v = (torch.abs(self.iql_tau - ((q - v) < 0).to(q.dtype)) * (q - v) ** 2).mean()
        self.log("loss_v", loss_v, prog_bar=True)
        self.log("pred_v", v.mean().detach(), prog_bar=True)

        # update Q
        self.V.eval()
        with torch.no_grad():
            td_target = rew + self.discount * (1 - done) * self.V(obs_next)
        q1, q2 = self.Q.both(obs, act)
        loss_q = ((q1 - td_target) ** 2 + (q2 - td_target) ** 2).mean()
        self.log("loss_q", loss_q, prog_bar=True)
        self.V.train()

        self.update_target()

        return loss_q + loss_v

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device):
        self.load_state_dict(torch.load(path, map_location=device))
