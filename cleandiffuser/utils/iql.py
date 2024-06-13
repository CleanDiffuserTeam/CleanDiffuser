from copy import deepcopy

import torch
import torch.nn as nn


class TwinQ(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.Q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.LayerNorm(256), nn.Mish(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.Mish(),
            nn.Linear(256, 1))
        self.Q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.LayerNorm(256), nn.Mish(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.Mish(),
            nn.Linear(256, 1))

    def both(self, obs, act):
        q1, q2 = self.Q1(torch.cat([obs, act], -1)), self.Q2(torch.cat([obs, act], -1))
        return q1, q2

    def forward(self, obs, act):
        return torch.min(*self.both(obs, act))


class V(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.V = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.LayerNorm(256), nn.Mish(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.Mish(),
            nn.Linear(256, 1))

    def forward(self, obs):
        v = self.V(obs)
        return v


class IQL(nn.Module):
    def __init__(self, obs_dim, act_dim, tau: float = 0.7):
        super().__init__()
        self.iql_tau = tau
        self.Q = TwinQ(obs_dim, act_dim)
        self.Q_targ = deepcopy(self.Q).requires_grad_(False).eval()
        self.V = V(obs_dim)
        self.optimV = torch.optim.Adam(self.V.parameters(), lr=3e-4)
        self.optimQ = torch.optim.Adam(self.Q.parameters(), lr=3e-4)

    def update_target(self, mu=0.995):
        for p, p_targ in zip(self.Q.parameters(), self.Q_targ.parameters()):
            p_targ.data = mu * p_targ.data + (1 - mu) * p.data

    def update_V(self, obs, act):
        q = self.Q_targ(obs, act)
        v = self.V(obs)
        loss = (torch.abs(self.iql_tau - ((q - v) < 0).float()) * (q - v) ** 2).mean()
        self.optimV.zero_grad()
        loss.backward()
        self.optimV.step()
        return loss.item()

    def update_Q(self, obs, act, rew, obs_next, done):
        with torch.no_grad():
            td_target = rew + 0.99 * (1 - done) * self.V(obs_next)
        q1, q2 = self.Q.both(obs, act)
        loss = ((q1 - td_target) ** 2 + (q2 - td_target) ** 2).mean()
        self.optimQ.zero_grad()
        loss.backward()
        self.optimQ.step()
        self.update_target()
        return loss.item()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device):
        self.load_state_dict(torch.load(path, map_location=device))
