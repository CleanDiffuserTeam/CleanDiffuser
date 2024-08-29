from typing import Optional

import os
import d4rl
import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cleandiffuser.dataset.d4rl_kitchen_dataset import D4RLKitchenDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import report_parameters


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MixerBlock(nn.Module):
    def __init__(
            self, seq_len: int = 5, hidden_size: int = 256,
            dim_s: int = 128, dim_c: int = 1024,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp1 = nn.Sequential(
            nn.Conv1d(seq_len, dim_s, 1), nn.GELU('tanh'), nn.Conv1d(dim_s, seq_len, 1))
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_size, dim_c), nn.GELU('tanh'), nn.Linear(dim_c, hidden_size))

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6))

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        shift_mlp1, scale_mlp1, gate_mlp1, shift_mlp2, scale_mlp2, gate_mlp2 = self.adaLN_modulation(emb).chunk(6,
                                                                                                                dim=1)
        x = modulate(self.norm1(x), shift_mlp1, scale_mlp1)
        x = x + gate_mlp1.unsqueeze(1) * self.mlp1(x)
        x = x + gate_mlp2.unsqueeze(1) * self.mlp2(modulate(self.norm2(x), shift_mlp2, scale_mlp2))
        return x


class MyMixerNNDiffusion(BaseNNDiffusion):
    def __init__(
            self,
            act_dim: int,
            Ta: int = 8, hidden_size: int = 256,
            dim_s: int = 128, dim_c: int = 1024,
            depth: int = 4,
            timestep_emb_type: str = "positional",
    ):
        super().__init__(hidden_size, timestep_emb_type)

        self.pre_linear = nn.Linear(act_dim, hidden_size)
        self.pos_linear = nn.Linear(hidden_size, act_dim)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.map_emb = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Mish(), nn.Linear(hidden_size, hidden_size), nn.Mish())

        self.mixer_blocks = nn.ModuleList([
            MixerBlock(seq_len=Ta, hidden_size=hidden_size, dim_s=dim_s, dim_c=dim_c) for _ in range(depth)])

        self.initialize_weights()

    def initialize_weights(self):

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize time step embedding MLP:
        nn.init.normal_(self.map_emb[0].weight, std=0.02)
        nn.init.normal_(self.map_emb[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.mixer_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.pos_linear.weight, 0)
        nn.init.constant_(self.pos_linear.bias, 0)

    def forward(self,
                x: torch.Tensor, noise: torch.Tensor,
                condition: Optional[torch.Tensor] = None):

        x = self.pre_linear(x)
        emb = self.map_noise(noise)
        if condition is not None:
            emb += condition
        emb = self.map_emb(emb)

        for block in self.mixer_blocks:
            x = block(x, emb)
        x = self.pos_linear(self.norm(x))

        return x


if __name__ == "__main__":

    device = "cuda:0"

    # --------------- Create Environment ---------------
    env = gym.make("kitchen-complete-v0")
    dataset = D4RLKitchenDataset(env.get_dataset(), horizon=8)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    # --------------- Network Architecture -----------------
    nn_diffusion = MyMixerNNDiffusion(
        act_dim, Ta=8, hidden_size=256, dim_s=128, dim_c=1024, depth=4,
        timestep_emb_type="positional")
    nn_condition = MLPCondition(obs_dim, 256, [256, ], dropout=0.0)

    report_parameters(nn_diffusion)

    # --------------- Diffusion Model Actor --------------------
    actor = DiscreteDiffusionSDE(
        nn_diffusion, nn_condition, predict_noise=False, optim_params={"lr": 3e-4},
        x_max=+1. * torch.ones((1, act_dim)),
        x_min=-1. * torch.ones((1, act_dim)),
        diffusion_steps=50, ema_rate=0.9999, device=device)

    # --------------- Training -------------------
    actor.train()

    avg_loss = 0.
    t = 0
    for batch in loop_dataloader(dataloader):

        obs = batch["obs"]["state"][:, 0].to(device)
        act = batch["act"].to(device)

        avg_loss += actor.update(act, obs)["loss"]

        t += 1
        if (t + 1) % 1000 == 0:
            print(f'[t={t + 1}] {avg_loss / 1000}')
            avg_loss = 0.

        if (t + 1) == 100000:
            break

    savepath = "tutorials/results/4_customize_your_diffusion_network_backbone/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    actor.save(savepath + "diffusion.pt")

    # -------------- Inference -----------------
    savepath = "tutorials/results/4_customize_your_diffusion_network_backbone/"
    actor.load(savepath + "diffusion.pt")
    actor.eval()

    env_eval = gym.vector.make("kitchen-complete-v0", num_envs=50)
    normalizer = dataset.get_normalizer()

    obs, cum_done, cum_rew = env_eval.reset(), 0., 0.
    prior = torch.zeros((50, 8, act_dim), device=device)

    for t in range(280 // 6):

        act, log = actor.sample(
            prior, solver="ddim", n_samples=50, sample_steps=20,
            temperature=0.5, w_cfg=1.0,
            condition_cfg=torch.tensor(normalizer.normalize(obs), device=device, dtype=torch.float32))
        act = act.cpu().numpy()

        for k in range(6):
            obs, rew, done, info = env_eval.step(act[:, k])
            cum_done = np.logical_or(cum_done, done)
            cum_rew += rew

        print(f'[t={t * 6}] cum_rew: {cum_rew}')

        if cum_done.all():
            break

    print(f'Mean score: {np.clip(cum_rew, 0., 4.).mean() * 25.}')
    env_eval.close()
