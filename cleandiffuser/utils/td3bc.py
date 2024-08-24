from copy import deepcopy

import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .utils import FreezeModules


class DeterministicPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256, act_scale: float = 1.0):
        super().__init__()
        self.act_scale = act_scale
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(
                hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(
                hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, act_dim), nn.Tanh())

    def forward(self, obs: torch.Tensor):
        return self.act_scale * self.net(obs)


class TwinQ(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.Q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim,
                      hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(
                hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, 1))
        self.Q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim,
                      hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(
                hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, 1))

    def both(self, obs, act):
        obs_act = torch.cat([obs, act], -1)
        q1, q2 = self.Q1(obs_act), self.Q2(obs_act)
        return q1, q2

    def forward(self, obs, act):
        return torch.min(*self.both(obs, act))


class TD3BC(L.LightningModule):
    """
    A pytorch-lightning implementation of TD3BC.

    The dataloader should return a tuple of (obs, act, rew, next_obs, done).

    Args:
        obs_dim (int): The dimension of the observation space.
        act_dim (int): The dimension of the action space.
        actor_hidden_dim (int): The number of hidden units in the actor network. Defaults to 256.
        critic_hidden_dim (int): The number of hidden units in the critic network. Defaults to 256.
        act_scale (float): The scale of the action. Defaults to 1.0.
        policy_noise (float): The noise added to the policy. Defaults to 0.2.
        noise_clip (float): The clip value for the noise. Defaults to 0.5.
        policy_freq (int): The frequency of policy updates. Defaults to 2.
        alpha (float): The alpha value for the TD3 algorithm. Defaults to 2.5.
        gamma (float): The discount factor. Defaults to 0.99.
        ema_rate (float): The rate for the exponential moving average. Defaults to 0.995.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        actor_hidden_dim: int = 256,
        critic_hidden_dim: int = 256,
        act_scale: float = 1.0,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        alpha: float = 2.5,
        gamma: float = 0.99,
        ema_rate: float = 0.995,
    ):
        super().__init__()
        self.automatic_optimization = False

        self.actor = DeterministicPolicy(
            obs_dim, act_dim, actor_hidden_dim, act_scale)
        self.actor_target = deepcopy(self.actor).requires_grad_(False).eval()

        self.critic = TwinQ(obs_dim, act_dim, critic_hidden_dim)
        self.critic_target = deepcopy(self.critic).requires_grad_(False).eval()

        self.act_scale = act_scale
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.ema_rate = ema_rate
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()
    def act(self, obs: np.ndarray):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        return self.actor(obs).cpu().numpy()

    def configure_optimizers(self):
        actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        return [actor_optimizer, critic_optimizer]

    def ema_update(self):
        for p, p_ema in zip(self.actor.parameters(), self.actor_target.parameters()):
            p_ema.data.mul_(self.ema_rate).add_(
                p.data, alpha=1. - self.ema_rate)
        for p, p_ema in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_ema.data.mul_(self.ema_rate).add_(
                p.data, alpha=1. - self.ema_rate)

    def training_step(self, batch, batch_idx):
        actor_optimizer, critic_optimizer = self.optimizers()

        obs, act, rew, next_obs, done = batch

        # --- Critic Update ---
        with torch.no_grad():

            noise = (torch.randn_like(act) *
                     self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_act = (self.actor_target(next_obs) +
                        noise).clamp(-self.act_scale, self.act_scale)

            target_q = rew + (1 - done) * self.gamma * \
                self.critic_target(next_obs, next_act)

        q1, q2 = self.critic.both(obs, act)

        critic_td_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        critic_optimizer.zero_grad()
        self.manual_backward(critic_td_loss)
        critic_optimizer.step()

        self.log("critic_td_loss", critic_td_loss, prog_bar=True)
        self.log("target_q", target_q.mean().detach(), prog_bar=True)

        # --- Policy Update ---
        if batch_idx % self.policy_freq == 0:

            self.critic.eval()

            with FreezeModules([self.critic, ]):
                pred_act = self.actor(obs)
                q = self.critic(obs, pred_act)
                lmbda = self.alpha / q.abs().mean().detach()

            policy_q_loss = -lmbda * q.mean()
            bc_loss = F.mse_loss(pred_act, act)

            actor_loss = policy_q_loss + bc_loss

            actor_optimizer.zero_grad()
            self.manual_backward(actor_loss)
            actor_optimizer.step()

            self.log("policy_q_loss", policy_q_loss, prog_bar=True)
            self.log("bc_loss", bc_loss, prog_bar=True)

            self.critic.train()

            self.ema_update()
