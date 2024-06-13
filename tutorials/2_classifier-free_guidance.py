import os

import d4rl
import gym
import numpy as np
import torch
import torch.nn as nn

from cleandiffuser.diffusion.diffusionsde import DiscreteDiffusionSDE
from cleandiffuser.nn_condition import BaseNNCondition, get_mask, MLPCondition
from cleandiffuser.nn_diffusion import PearceMlp

"""
In this tutorial, we will review the applications of classifier-free guidance and how to customize it for a diffusion model. In Appendix A.2 of the paper, 
we provide detailed information on classifier-free guidance. In simple terms, we achieve this by learning a conditional diffusion model \epsilon(x_t,t,c) 
and an unconditional diffusion model \epsilon(x_t, t), using their weighted sum w*\epsilon(x_t,t,c)+(1-w)*\epsilon(x_t, t) as the predicted noise. 
The weight w between them influences the strength of guidance. 
When w=0, it is equivalent to sampling directly from p(x); when w=1, it is akin to sampling directly from p(x|c); when w>1, the generated samples x increasingly reflect c. 
In practice, we do not actually train two diffusion models; instead, we treat the unconditional model as a special case of the conditional model 
by introducing a specific variable to represent the unconditional scenario. 
Following common practices, we encode the condition variable into features using `nn_condition`, and use an all-zero feature indicates no condition. 
Therefore, during the creation of `nn_condition`, a `dropout` parameter is included, which determines the probability of setting the condition feature to zero during training. 
Choosing a `dropout` greater than 0 (typically around dropout=0.25) enables the diffusion model to utilize classifier-free guidance. 
In tutorial 1, we set dropout=0 because we directly fit the diffusion model to p(a|s), equivalent to w=1, and do not require classifier-free guidance. 
In this tutorial, we will attempt to customize a version that incorporates classifier-free guidance.
"""

class MyObsNNCondition(BaseNNCondition):
    def __init__(self, obs_dim, emb_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, emb_dim))
        self.dropout = dropout

    def forward(self, condition: torch.Tensor, mask: torch.Tensor = None):
        mask = get_mask(mask, (condition.shape[0], 1), self.dropout, self.training, condition.device)
        return self.mlp(condition) * mask


if __name__ == "__main__":

    device = "cuda:1"
    use_customized_nn_condition = True

    # --------------- Create Environment ---------------
    env = gym.make("kitchen-complete-v0")
    dataset = d4rl.qlearning_dataset(env)
    obs_dim, act_dim = dataset['observations'].shape[-1], dataset['actions'].shape[-1]
    size = len(dataset['observations'])

    # --------------- Network Architecture -----------------
    nn_diffusion = PearceMlp(act_dim, To=1, emb_dim=64, hidden_dim=256, timestep_emb_type="positional")

    if use_customized_nn_condition:
        nn_condition = MyObsNNCondition(obs_dim, emb_dim=64, hidden_dim=64, dropout=0.2)
    else:
        nn_condition = MLPCondition(
            in_dim=obs_dim, out_dim=64, hidden_dims=[64, ], act=nn.SiLU(), dropout=0.2)

    # --------------- Diffusion Model Actor --------------------
    actor = DiscreteDiffusionSDE(
        nn_diffusion, nn_condition, predict_noise=False, optim_params={"lr": 3e-4},
        x_max=+1. * torch.ones((1, act_dim)),
        x_min=-1. * torch.ones((1, act_dim)),
        diffusion_steps=5, ema_rate=0.9999, device=device)

    # --------------- Training -------------------
    actor.train()

    avg_loss = 0.
    for t in range(100000):

        idx = np.random.randint(0, size, (256,))
        obs = torch.tensor(dataset['observations'][idx], device=device).float()
        act = torch.tensor(dataset['actions'][idx], device=device).float()

        avg_loss += actor.update(act, obs)["loss"]

        if (t + 1) % 1000 == 0:
            print(f'[t={t + 1}] {avg_loss / 1000}')
            avg_loss = 0.

    savepath = "tutorials/results/2_CFG/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    actor.save(savepath + "diffusion.pt")

    # -------------- Inference -----------------
    savepath = "tutorials/results/2_CFG/"
    actor.load(savepath + "diffusion.pt")
    actor.eval()

    env_eval = gym.vector.make("kitchen-complete-v0", num_envs=50)

    obs, cum_done, cum_rew = env_eval.reset(), 0., 0.
    prior = torch.zeros((50, act_dim), device=device)
    for t in range(280):

        act, log = actor.sample(
            prior, solver="ddpm", n_samples=50, sample_steps=5,
            temperature=0.5, w_cfg=1.2,
            condition_cfg=torch.tensor(obs, device=device, dtype=torch.float32))
        act = act.cpu().numpy()

        obs, rew, done, info = env_eval.step(act)
        cum_done = np.logical_or(cum_done, done)
        cum_rew += rew

        print(f'[t={t}] cum_rew: {cum_rew}')

        if cum_done.all():
            break

    print(f'Mean score: {np.clip(cum_rew, 0., 4.).mean() * 25.}')
    env_eval.close()
