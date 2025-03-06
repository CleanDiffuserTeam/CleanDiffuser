import argparse
from pathlib import Path
from typing import Union

import d4rl
import einops
import gym
import numpy as np
import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from termcolor import cprint
from torch.utils.data import DataLoader

from cleandiffuser.dataset.d4rl_antmaze_dataset import D4RLAntmazeTDDataset
from cleandiffuser.dataset.d4rl_kitchen_dataset import D4RLKitchenTDDataset
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.nn_diffusion import IDQLMlp
from cleandiffuser.utils import set_seed
from cleandiffuser.utils.valuefuncs import Iql


class BC_Wrapper(torch.utils.data.Dataset):
    def __init__(
        self, dataset: Union[D4RLMuJoCoTDDataset, D4RLAntmazeTDDataset, D4RLKitchenTDDataset]
    ):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, idx):
        return {
            "x0": self.act[idx],
            "condition_cfg": self.obs[idx],
        }


# -- config --
argparser = argparse.ArgumentParser()
argparser.add_argument("--seed", type=int, default=0)
argparser.add_argument("--env_name", type=str, default="halfcheetah-medium-expert-v2")
argparser.add_argument("--mode", type=str, default="bc_training")
argparser.add_argument("--devices", type=int, nargs="+", default=[0, 1])
argparser.add_argument("--training_steps", type=int, default=1000000)
argparser.add_argument("--save_every_n_steps", type=int, default=200000)
argparser.add_argument("--ckpt_file", type=str, default="diffusion_bc-step=1000000.ckpt")
argparser.add_argument("--sampling_steps", type=int, default=10)
argparser.add_argument("--num_envs", type=int, default=50)
argparser.add_argument("--num_episodes", type=int, default=3)
argparser.add_argument("--num_candidates", type=int, default=50)
args = argparser.parse_args()

seed = args.seed
env_name = args.env_name
mode = args.mode
devices = args.devices
training_steps = args.training_steps
save_every_n_steps = args.save_every_n_steps
ckpt_file = args.ckpt_file
sampling_steps = args.sampling_steps
num_envs = args.num_envs
num_episodes = args.num_episodes
num_candidates = args.num_candidates

if env_name == "halfcheetah-medium-expert-v2":
    weight_temperature = 40.0
elif env_name == "halfcheetah-medium-v2":
    weight_temperature = 50.0
elif env_name == "halfcheetah-medium-replay-v2":
    weight_temperature = 100.0
elif env_name == "hopper-medium-expert-v2":
    weight_temperature = 1.0
elif env_name == "hopper-medium-v2":
    weight_temperature = 50.0
elif env_name == "hopper-medium-replay-v2":
    weight_temperature = 250.0
elif env_name == "walker2d-medium-expert-v2":
    weight_temperature = 150.0
elif env_name == "walker2d-medium-v2":
    weight_temperature = 200.0
elif env_name == "walker2d-medium-replay-v2":
    weight_temperature = 100.0
elif env_name == "kitchen-mixed-v0":
    weight_temperature = 5.0
elif env_name == "kitchen-partial-v0":
    weight_temperature = 5.0
elif env_name == "antmaze-medium-play-v2":
    weight_temperature = 10.0
elif env_name == "antmaze-medium-diverse-v2":
    weight_temperature = 10.0
elif env_name == "antmaze-large-play-v2":
    weight_temperature = 10.0
elif env_name == "antmaze-large-diverse-v2":
    weight_temperature = 10.0
else:
    raise NotImplementedError(f"env_name={env_name} is not supported.")


if __name__ == "__main__":
    set_seed(seed)

    save_path = Path(__file__).parents[2] / f"results/idql/{env_name}/"

    # --- Create Dataset ---
    env = gym.make(env_name)
    if "kitchen" in env_name:
        dataset = D4RLKitchenTDDataset(d4rl.qlearning_dataset(env))
    elif "antmaze" in env_name:
        dataset = D4RLAntmazeTDDataset(d4rl.qlearning_dataset(env))
    else:
        dataset = D4RLMuJoCoTDDataset(d4rl.qlearning_dataset(env))
    obs_dim, act_dim = dataset.obs_dim, dataset.act_dim

    # --- Create Diffusion Model ---
    nn_diffusion = IDQLMlp(x_dim=act_dim, timestep_emb_type="untrainable_fourier")
    nn_condition = MLPCondition(in_dim=obs_dim, out_dim=64, hidden_dims=64, dropout=0.0)

    actor = ContinuousDiffusionSDE(
        nn_diffusion,
        nn_condition,
        ema_rate=0.999,
        x_max=+1.0 * torch.ones((act_dim,)),
        x_min=-1.0 * torch.ones((act_dim,)),
    )

    # --- BC Training ---
    if mode == "bc_training":
        dataloader = DataLoader(
            BC_Wrapper(dataset),
            batch_size=512,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )

        callback = ModelCheckpoint(
            dirpath=save_path,
            filename="diffusion_bc-{step}",
            every_n_train_steps=save_every_n_steps,
            save_top_k=-1,
        )

        trainer = L.Trainer(
            devices=devices,
            max_steps=training_steps,
            default_root_dir=save_path,
            callbacks=[callback],
            logger=wandb_logger,
        )

        trainer.fit(actor, dataloader)

    # --- IQL Training ---
    elif mode == "iql_training":
        cprint("Please run `cleandiffuser/utils/valuefuncs/iql.py`", "green")

    # --- Inference ---
    elif mode == "inference":
        device = f"cuda:{devices[0]}"

        iql, iql_params = Iql.from_pretrained(env_name)
        iql.to(device).eval()

        actor.load_state_dict(torch.load(save_path / ckpt_file, map_location=device)["state_dict"])
        actor = actor.to(device).eval()

        env_eval = gym.vector.make(env_name, num_envs=num_envs)
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        prior = torch.zeros((num_envs * num_candidates, act_dim))
        for i in range(num_episodes):
            obs, ep_reward, all_done, t = env_eval.reset(), 0.0, False, 0

            while not np.all(all_done) and t < 1000:
                obs = torch.tensor(normalizer.normalize(obs), dtype=torch.float32, device=device)
                obs = einops.repeat(obs, "b d -> (b k) d", k=num_candidates)

                act, _ = actor.sample(
                    prior,
                    solver="ddpm",
                    sample_steps=sampling_steps,
                    condition_cfg=obs,
                    w_cfg=1.0,
                    sample_step_schedule="quad",
                )

                with torch.no_grad():
                    q = iql.forward_q(obs, act)
                    v = iql.forward_v(obs)
                    act = einops.rearrange(act, "(b k) d -> b k d", k=num_candidates)
                    adv = einops.rearrange((q - v), "(b k) 1 -> b k 1", k=num_candidates)
                    w = torch.softmax(adv * weight_temperature, dim=1)

                    idx = torch.multinomial(w.squeeze(-1), num_samples=1).squeeze(-1)
                    act = act[torch.arange(num_envs), idx].cpu().numpy()

                obs, rew, done, _ = env_eval.step(act)

                t += 1
                done = np.logical_and(done, t < 1000)
                all_done = np.logical_or(all_done, done)
                if "kitchen" in env_name:
                    ep_reward = np.clip(ep_reward + rew, 0.0, 4.0)
                    print(f"[t={t}] finished tasks: {np.around(ep_reward)}")
                elif "antmaze" in env_name:
                    ep_reward = np.clip(ep_reward + rew, 0.0, 1.0)
                    print(f"[t={t}] xy: {np.around(obs[:, :2], 2)}")
                    print(f"[t={t}] reached goal: {np.around(ep_reward)}")
                else:
                    ep_reward += rew * (1 - all_done)
                    print(f"[t={t}] rew: {np.around((rew * (1 - all_done)), 2)}")

                if np.all(all_done):
                    break

            episode_rewards.append(ep_reward)

        episode_rewards = [
            list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards
        ]
        episode_rewards = np.array(episode_rewards).mean(-1) * 100.0
        print(f"Score: {episode_rewards.mean():.3f}±{episode_rewards.std():.3f}")

        env_eval.close()
