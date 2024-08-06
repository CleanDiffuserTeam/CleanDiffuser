import pytorch_lightning as L
import torch
import torch.nn as nn

from cleandiffuser.utils import Mlp


class BasicInvDynamic(L.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, obs, next_obs):
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, obs, next_obs):
        return self.forward(obs, next_obs)

    def __call__(self, obs, next_obs):
        return self.predict(obs, next_obs)


class MlpInvDynamic(BasicInvDynamic):
    def __init__(
            self,
            obs_dim: int, act_dim: int,
            hidden_dim: int,
            out_activation: nn.Module = nn.Tanh(),
    ):
        super().__init__()
        self.mlp = Mlp(
            2 * obs_dim, [hidden_dim, hidden_dim], act_dim, nn.ReLU(), out_activation)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)

    def forward(self, obs: torch.Tensor, next_obs: torch.Tensor):
        return self.mlp(torch.cat([obs, next_obs], -1))

    def loss(self, obs, act, next_obs):
        pred_act = self.forward(obs, next_obs)
        return (pred_act - act).pow(2).mean()

    def training_step(self, batch, batch_idx):
        obs, act, next_obs = batch
        return self.loss(obs, act, next_obs)

    def validation_step(self, batch, batch_idx):
        obs, act, next_obs = batch
        return self.loss(obs, act, next_obs)


class FancyMlpInvDynamic(MlpInvDynamic):
    def __init__(
            self,
            obs_dim: int, act_dim: int,
            hidden_dim: int,
            out_activation: nn.Module = nn.Tanh(),
            add_norm: bool = False, add_dropout: bool = False,
    ):
        super(BasicInvDynamic, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * obs_dim, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim) if add_norm else nn.Identity(),
            nn.Dropout(0.1) if add_dropout else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, act_dim), out_activation)
