import d4rl
import gym
import numpy as np
import torch
import torch.nn as nn
import os

from cleandiffuser.classifier import BaseClassifier, MSEClassifier
from cleandiffuser.diffusion.diffusionsde import DiscreteDiffusionSDE
from cleandiffuser.nn_classifier import BaseNNClassifier, MLPNNClassifier
from cleandiffuser.nn_diffusion import PearceMlp


class MyObsNNClassifier(BaseNNClassifier):
    """ pred_obs = nn_classifier(act, t) """
    def __init__(self, obs_dim, act_dim, emb_dim, timestep_emb_type):
        super().__init__(emb_dim, timestep_emb_type)
        self.mlp = nn.Sequential(
            nn.Linear(act_dim + emb_dim, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, obs_dim))

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor = None):
        pred_obs = self.mlp(torch.cat([x, self.map_noise(t)], dim=-1))
        return pred_obs


class MyObsClassifier(BaseClassifier):
    """ logp(s | a, t) = - MSE(pred_obs - obs) """
    def __init__(self, nn_classifier: MyObsNNClassifier, device: str = "cpu"):
        super().__init__(
            nn_classifier, ema_rate=0.995, grad_clip_norm=None, optim_params=None, device=device)

    def loss(self, x: torch.Tensor, noise: torch.Tensor, y: torch.Tensor):
        pred_obs = self.model(x, noise, y)
        return nn.functional.mse_loss(pred_obs, y)

    def logp(self, x: torch.Tensor, noise: torch.Tensor, c: torch.Tensor):
        pred_obs = self.model_ema(x, noise)
        logp = - ((pred_obs - c) ** 2).mean(-1, keepdim=True)
        return logp


if __name__ == "__main__":

    device = "cuda:2"
    use_customized_classifier = True

    # --------------- Create Environment ---------------
    env = gym.make("kitchen-complete-v0")
    dataset = d4rl.qlearning_dataset(env)
    obs_dim, act_dim = dataset['observations'].shape[-1], dataset['actions'].shape[-1]
    size = len(dataset['observations'])

    # --------------- Network Architecture -----------------
    nn_diffusion = PearceMlp(act_dim, To=0, emb_dim=64, hidden_dim=256, timestep_emb_type="positional")

    if use_customized_classifier:
        nn_classifier = MyObsNNClassifier(obs_dim, act_dim, 32, "positional")
        classifier = MyObsClassifier(nn_classifier, device)
    else:
        nn_classifier = MLPNNClassifier(
            x_dim=act_dim, out_dim=obs_dim, emb_dim=32, hidden_dims=[256, ],
            activation=nn.SiLU(), timestep_emb_type="positional")
        classifier = MSEClassifier(
            nn_classifier, temperature=1.0, device=device)

    # --------------- Diffusion Model Actor --------------------
    actor = DiscreteDiffusionSDE(
        nn_diffusion, classifier=classifier, predict_noise=False, optim_params={"lr": 3e-4},
        x_max=+1. * torch.ones((1, act_dim)),
        x_min=-1. * torch.ones((1, act_dim)),
        diffusion_steps=5, ema_rate=0.9999, device=device)

    # --------------- Training -------------------
    actor.train()

    avg_loss_diffusion, avg_loss_classifier = 0., 0.
    for t in range(100000):

        idx = np.random.randint(0, size, (256,))
        obs = torch.tensor(dataset['observations'][idx], device=device).float()
        act = torch.tensor(dataset['actions'][idx], device=device).float()

        avg_loss_diffusion += actor.update(act)["loss"]
        avg_loss_classifier += actor.update_classifier(act, obs)["loss"]

        if (t + 1) % 1000 == 0:
            print(f'[t={t + 1}] avg_loss_diffusion: {avg_loss_diffusion / 1000 :.3f} | '
                  f'avg_loss_classifier: {avg_loss_classifier / 1000 :.3f}')
            avg_loss_diffusion, avg_loss_classifier = 0., 0.

    savepath = "tutorials/results/3_CG/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    actor.save(savepath + "diffusion.pt")
    classifier.save(savepath + "classifier.pt")

    # -------------- Inference -----------------
    savepath = "tutorials/results/3_CG/"
    actor.load(savepath + "diffusion.pt")
    classifier.load(savepath + "classifier.pt")
    actor.eval()
    classifier.eval()

    env_eval = gym.vector.make("kitchen-complete-v0", num_envs=50)

    obs, cum_done, cum_rew = env_eval.reset(), 0., 0.
    prior = torch.zeros((50, act_dim), device=device)
    for t in range(280):

        act, log = actor.sample(
            prior, solver="ddpm", n_samples=50, sample_steps=5,
            temperature=0.5, w_cg=0.1,
            condition_cg=torch.tensor(obs, device=device, dtype=torch.float32))
        act = act.cpu().numpy()

        obs, rew, done, info = env_eval.step(act)
        cum_done = np.logical_or(cum_done, done)
        cum_rew += rew

        print(f'[t={t}] cum_rew: {cum_rew}')

        if cum_done.all():
            break

    print(f'Mean score: {np.clip(cum_rew, 0., 4.).mean() * 25.}')
    env_eval.close()
