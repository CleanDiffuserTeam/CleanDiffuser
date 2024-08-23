import os

import pytorch_lightning as L
import torch
import torch.nn as nn

from cleandiffuser.utils import Mlp


class BasicInvDynamic(L.LightningModule):
    """ Basic Inverse Dynamics Model.
    
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
    """ MLP Inverse Dynamics Model.
    
    Predicts the action given the current and next observation.
    It concatenates the current and next observation 
    and passes it through an MLP to predict the action.
    
    Args:
        obs_dim (int): Dimension of the observation space.
        act_dim (int): Dimension of the action space.
        hidden_dim (int): Dimension of the hidden layers.
        out_activation (nn.Module): Activation function for the output layer. Default: nn.Tanh().
        action_scale (float): Scale factor for the action. Default: 1.0.
    
    Examples:
    >>> inv_dynamic = BasicInvDynamic()
    >>> act = inv_dynamic.predict(obs, next_obs)
    """
    def __init__(
            self,
            obs_dim: int, act_dim: int,
            hidden_dim: int,
            out_activation: nn.Module = nn.Tanh(),
            action_scale: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["out_activation"])
        self.action_scale = action_scale
        self.mlp = Mlp(
            2 * obs_dim, [hidden_dim, hidden_dim], act_dim, nn.ReLU(), out_activation)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)

    def forward(self, obs: torch.Tensor, next_obs: torch.Tensor):
        return self.mlp(torch.cat([obs, next_obs], -1)) * self.action_scale

    def loss(self, obs, act, next_obs):
        pred_act = self.forward(obs, next_obs)
        return (pred_act - act).pow(2).mean()

    def training_step(self, batch, batch_idx):
        obs, act, next_obs = batch
        loss = self.loss(obs, act, next_obs)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        obs, act, next_obs = batch
        loss = self.loss(obs, act, next_obs)
        self.log("val/loss", loss)
        return loss


class FancyMlpInvDynamic(MlpInvDynamic):
    """ MLP Inverse Dynamics Model.
    
    Predicts the action given the current and next observation.
    It concatenates the current and next observation 
    and passes it through an MLP to predict the action.
    The MLP has LayerNorm and Dropout layers to improve generalization.
    The layers are arranged as follows:
    Linear -> GELU -> LayerNorm -> Dropout -> Linear -> GELU -> Linear
    
    Args:
        obs_dim (int): Dimension of the observation space.
        act_dim (int): Dimension of the action space.
        hidden_dim (int): Dimension of the hidden layers.
        out_activation (nn.Module): Activation function for the output layer. Default: nn.Tanh().
        action_scale (float): Scale factor for the action. Default: 1.0.
        add_norm (bool): Add a LayerNorm layer. Default: False.
        add_dropout (bool): Add a Dropout layer. Default: False.
    
    Examples:
    >>> inv_dynamic = BasicInvDynamic()
    >>> act = inv_dynamic.predict(obs, next_obs)
    """
    def __init__(
            self,
            obs_dim: int, act_dim: int,
            hidden_dim: int,
            out_activation: nn.Module = nn.Tanh(),
            action_scale: float = 1.0,
            add_norm: bool = False, add_dropout: bool = False,
    ):
        super(BasicInvDynamic, self).__init__()
        self.save_hyperparameters(ignore=["out_activation"])
        self.action_scale = action_scale
        self.mlp = nn.Sequential(
            nn.Linear(2 * obs_dim, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim) if add_norm else nn.Identity(),
            nn.Dropout(0.1) if add_dropout else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, act_dim), out_activation)

    @classmethod
    def from_pretrained(self, env_name: str, hidden_dim: int = 256):
        """ Load pretrained model.
        
        Load pretrained model from the CleanDiffuser pretrain directory `~/.CleanDiffuser/pretrain/invdyn/`.
        Only available for the following environments: D4RL-MuJoCo-v2, D4RL-Kitchen-v0, D4RL-AntMaze-v2.
        
        Note that the pretrained model should be used with the same state normalizer as the training data, i.e., 
        cleandiffuser.dataset.D4RLxxxTDDataset/D4RLxxxDataset.
        
        Args:
            env_name (str): Environment name.
            hidden_dim (int): Dimension of the hidden layers. Can be 256, 512, 1024. Default: 256.
        
        Returns:
            invdyn (FancyMlpInvDynamic): Pretrained model.
        """
        path = os.path.expanduser("~") + f"/.CleanDiffuser/pretrain/invdyn/{env_name}/"
        if not os.path.exists(path):
            raise FileNotFoundError("Pretrained model not found.")
        
        hparams_file = f"hparams_{hidden_dim}.yaml"
        ckpt_file = f"hidden_dim={hidden_dim}.ckpt"
        
        return self.load_from_checkpoint(
            path + ckpt_file, map_location="cpu", output_activation=nn.Tanh())
