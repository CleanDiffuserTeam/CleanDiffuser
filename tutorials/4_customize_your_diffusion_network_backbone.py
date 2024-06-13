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
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import SinusoidalEmbedding


class MyNNDiffusion(BaseNNDiffusion):
    def __init__(
            self,
            obs_dim: int, act_dim: int,
            To: int = 4, Ta: int = 8,
            d_model: int = 128, nhead: int = 2, num_layers: int = 2,
            timestep_emb_type: str = "positional",
    ):
        super().__init__(d_model, timestep_emb_type)

        self.act_emb = nn.Linear(act_dim, d_model)
        self.obs_emb = nn.Linear(obs_dim, d_model)
        pos_emb = SinusoidalEmbedding(d_model)
        self.act_pos_emb = nn.Parameter(
            pos_emb(torch.arange(0, Ta))[None, ], requires_grad=False)
        self.obs_pos_emb = nn.Parameter(
            pos_emb(torch.arange(0, To))[None, ], requires_grad=False)

        self.tfm = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model, batch_first=True), num_layers=num_layers)

        self.final_layer = nn.Linear(d_model, act_dim)

    def forward(self,
                x: torch.Tensor, noise: torch.Tensor,
                condition: Optional[torch.Tensor] = None):
        noise = self.map_noise(noise)
        condition = self.obs_emb(condition) + self.obs_pos_emb if condition is not None else self.obs_pos_emb
        x = self.act_emb(x) + self.act_pos_emb
        c = torch.cat([condition, noise.unsqueeze(1)], 1)
        x = self.tfm(x, c)
        return self.final_layer(x)


if __name__ == "__main__":

    device = "cuda:3"

    # --------------- Create Environment ---------------
    env = gym.make("kitchen-complete-v0")
    dataset = D4RLKitchenDataset(env.get_dataset(), horizon=11)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    # --------------- Network Architecture -----------------
    nn_diffusion = MyNNDiffusion(
        obs_dim, act_dim, To=4, Ta=8,
        d_model=128, nhead=2, num_layers=2, timestep_emb_type="positional")
    nn_condition = IdentityCondition(dropout=0.0)

    # --------------- Diffusion Model Actor --------------------
    actor = DiscreteDiffusionSDE(
        nn_diffusion, nn_condition, predict_noise=False, optim_params={"lr": 3e-4},
        x_max=+1. * torch.ones((1, act_dim)),
        x_min=-1. * torch.ones((1, act_dim)),
        diffusion_steps=5, ema_rate=0.9999, device=device)

    # --------------- Training -------------------
    actor.train()

    avg_loss = 0.
    t = 0
    for batch in loop_dataloader(dataloader):

        obs = batch["obs"]["state"][:, :4].to(device)
        act = batch["act"][:, 3:].to(device)

        avg_loss += actor.update(act, obs)["loss"]

        t += 1
        if (t + 1) % 1000 == 0:
            print(f'[t={t + 1}] {avg_loss / 1000}')
            avg_loss = 0.

        if (t + 1) == 100000:
            break

    savepath = "tutorials/results/4_Customize_NN_diffusion/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    actor.save(savepath + "diffusion.pt")

    # -------------- Inference -----------------
    savepath = "tutorials/results/4_Customize_NN_diffusion/"
    actor.load(savepath + "diffusion.pt")
    actor.eval()

    env_eval = gym.vector.make("kitchen-complete-v0", num_envs=50)
    normalizer = dataset.get_normalizer()

    obs, cum_done, cum_rew = env_eval.reset(), 0., 0.
    prior = torch.zeros((50, 8, act_dim), device=device)
    condition = torch.zeros((50, 4, obs_dim), device=device)
    condition[:, -1] = torch.tensor(normalizer.normalize(obs), device=device, dtype=torch.float32)

    for t in range(280 // 4):

        act, log = actor.sample(
            prior, solver="ddpm", n_samples=50, sample_steps=5,
            temperature=0.5, w_cfg=1.0,
            condition_cfg=condition)
        act = act.cpu().numpy()

        for k in range(4):
            obs, rew, done, info = env_eval.step(act[:, k])
            cum_done = np.logical_or(cum_done, done)
            cum_rew += rew
            condition[:, :3] = condition[:, 1:]
            condition[:, -1] = torch.tensor(normalizer.normalize(obs), device=device, dtype=torch.float32)

        print(f'[t={t * 4}] cum_rew: {cum_rew}')

        if cum_done.all():
            break

    print(f'Mean score: {np.clip(cum_rew, 0., 4.).mean() * 25.}')
    env_eval.close()
