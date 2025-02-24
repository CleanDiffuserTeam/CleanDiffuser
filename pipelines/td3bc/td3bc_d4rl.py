import argparse
from pathlib import Path

import d4rl
import gym
import numpy as np
import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from cleandiffuser.dataset.d4rl_antmaze_dataset import D4RLAntmazeTDDataset
from cleandiffuser.dataset.d4rl_kitchen_dataset import D4RLKitchenTDDataset
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
from cleandiffuser.utils import set_seed
from cleandiffuser.utils.offlinerl.td3bc import TD3BC

# -- config --

argparser = argparse.ArgumentParser()
argparser.add_argument("--seed", type=int, default=0)
argparser.add_argument("--devices", type=int, nargs="+", default=[0])
argparser.add_argument("--env_name", type=str, default="halfcheetah-medium-expert-v2")
argparser.add_argument("--mode", type=str, default="training")
argparser.add_argument("--save_every_n_steps", type=int, default=200000)
argparser.add_argument("--training_steps", type=int, default=1000000)
argparser.add_argument("--ckpt_file", type=str, default="td3bc-step=1000000.ckpt")
argparser.add_argument("--num_envs", type=int, default=50)
argparser.add_argument("--num_episodes", type=int, default=3)
args = argparser.parse_args()

seed = args.seed
devices = args.devices
env_name = args.env_name
mode = args.mode
save_every_n_steps = args.save_every_n_steps
training_steps = args.training_steps
ckpt_file = args.ckpt_file
num_envs = args.num_envs
num_episodes = args.num_episodes

if __name__ == "__main__":
    set_seed(seed)

    save_path = Path(__file__).parents[2] / f"results/td3bc/{env_name}/"

    # --- Create Dataset ---
    env = gym.make(env_name)
    if "kitchen" in env_name:
        dataset = D4RLKitchenTDDataset(d4rl.qlearning_dataset(env))
    elif "antmaze" in env_name:
        dataset = D4RLAntmazeTDDataset(d4rl.qlearning_dataset(env))
    else:
        dataset = D4RLMuJoCoTDDataset(d4rl.qlearning_dataset(env))
    obs_dim, act_dim = dataset.obs_dim, dataset.act_dim

    # --- Create ActorCritic ---
    actor = TD3BC(obs_dim, act_dim)

    # --- Training ---
    if mode == "training":
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=256, shuffle=True, num_workers=4, persistent_workers=True
        )

        callback = ModelCheckpoint(
            dirpath=save_path,
            filename="td3bc-{step}",
            every_n_train_steps=save_every_n_steps,
            save_top_k=-1,
        )

        trainer = L.Trainer(
            devices=devices,
            max_steps=training_steps,
            default_root_dir=save_path,
            callbacks=[callback],
        )

        trainer.fit(actor, dataloader)

    # --- Inference ---
    elif mode == "inference":
        device = f"cuda:{devices[0]}"

        actor.load_state_dict(torch.load(save_path / ckpt_file, map_location=device)["state_dict"])
        actor = actor.to(device).eval()

        env_eval = gym.vector.make(env_name, num_envs=num_envs)
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        for i in range(num_episodes):
            obs, ep_reward, all_done, t = env_eval.reset(), 0.0, False, 0

            while not np.all(all_done) and t < 1000:
                act = actor.act(normalizer.normalize(obs.astype(np.float32)))
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
