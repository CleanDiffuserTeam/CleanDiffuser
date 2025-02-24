import json
import os
from pathlib import Path

import pytorch_lightning as L
import torch
import torch.nn as nn


class BasicInvDynamic(L.LightningModule):
    """Basic Inverse Dynamics Model.

    Predicts the action given the current and next observation.

    Args:
    Examples:
    >>> inv_dynamic = BasicInvDynamic()
    >>> act = inv_dynamic.predict(obs, next_obs)
    """

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

    def forward(self, obs, next_obs):
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, obs, next_obs):
        return self.forward(obs, next_obs)


class MlpInvDynamic(BasicInvDynamic):
    """MLP Inverse Dynamics Model.

    TODO
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 512,
        tanh_out_activation: bool = True,
        action_scale: float = 1.0,
    ):
        super().__init__()
        self._tanh_out_activation = tanh_out_activation
        self.action_scale = action_scale
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh() if tanh_out_activation else nn.Identity(),
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)

    def forward(self, obs: torch.Tensor, next_obs: torch.Tensor):
        out = self.mlp(torch.cat([obs, next_obs], -1))
        if self._tanh_out_activation:
            out = out * self.action_scale
        return out

    def loss(self, obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor):
        pred_act = self.forward(obs, next_obs)
        return torch.nn.functional.mse_loss(pred_act, act)

    def training_step(self, batch, batch_idx):
        obs = batch["obs"]
        act = batch["act"]
        next_obs = batch["next_obs"]
        loss = self.loss(obs, act, next_obs)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        obs = batch["obs"]
        act = batch["act"]
        next_obs = batch["next_obs"]
        loss = self.loss(obs, act, next_obs)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss


class FancyMlpInvDynamic(MlpInvDynamic):
    """MLP Inverse Dynamics Model.

    TODO
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 512,
        tanh_out_activation: bool = True,
        action_scale: float = 1.0,
        add_norm: bool = True,
        add_dropout: bool = True,
    ):
        super().__init__(obs_dim, act_dim, hidden_dim, tanh_out_activation, action_scale)
        self.mlp = nn.Sequential(
            nn.Linear(2 * obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if add_norm else nn.Identity(),
            nn.GELU(approximate="tanh"),
            nn.Dropout(0.1) if add_dropout else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if add_norm else nn.Identity(),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh() if tanh_out_activation else nn.Identity(),
        )

    @classmethod
    def from_pretrained(self, env_name: str):
        try:
            path = (
                os.path.expanduser("~") + f"/.CleanDiffuser/pretrained/invdyn/{env_name}/"
            )
            path = Path(path)
            file_list = os.listdir(path)
            for each in file_list:
                if ".ckpt" in each:
                    print(f"Pretrained model loaded from {path / each}")
                    with open(path / "params.json", "r") as f:
                        params = json.load(f)
                    model = FancyMlpInvDynamic(**params["config"])
                    model.load_state_dict(torch.load(path / each, map_location="cpu")["state_dict"])
                    return model, params
            else:
                Warning(f"No pretrained model found in {path}")
                return None, None
        except Exception as e:
            print(e)
            return None, None
