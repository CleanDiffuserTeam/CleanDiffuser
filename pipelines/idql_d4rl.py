from pathlib import Path

import d4rl
import einops
import gym
import hydra
import numpy as np
import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.nn_diffusion import IDQLMlp
from cleandiffuser.utils import IQL


class BC_Wrapper(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, idx):
        return self.act[idx], self.obs[idx]


class IQL_Wrapper(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, idx):
        return self.obs[idx], self.act[idx], self.rew[idx], self.next_obs[idx], self.tml[idx]


@hydra.main(config_path="../configs/idql", config_name="d4rl", version_base=None)
def pipeline(args):

    L.seed_everything(args.seed, workers=True)

    env_name = args.task.env_name
    save_path = Path(__file__).parents[1] / \
        f"results/{args.pipeline_name}/{env_name}/"

    # --- Create Dataset ---
    env = gym.make(env_name)
    raw_dataset = d4rl.qlearning_dataset(env)
    dataset = D4RLMuJoCoTDDataset(raw_dataset, args.normalize_reward)
    obs_dim, act_dim = dataset.obs_dim, dataset.act_dim

    # --- Create Diffusion Model ---
    nn_diffusion = IDQLMlp(
        x_dim=act_dim, emb_dim=64, hidden_dim=256, n_blocks=3, dropout=0.1,
        timestep_emb_type="untrainable_fourier")
    nn_condition = MLPCondition(
        in_dim=obs_dim, out_dim=64, hidden_dims=[64, ], act=torch.nn.SiLU(), dropout=0.0)

    # --- Training ---
    if args.mode == "bc_training":

        actor = ContinuousDiffusionSDE(
            nn_diffusion, nn_condition, ema_rate=0.9999,
            x_max=+1.0*torch.ones((act_dim, )),
            x_min=-1.0*torch.ones((act_dim, )))

        dataloader = DataLoader(
            BC_Wrapper(dataset),
            batch_size=256, shuffle=True, num_workers=4, persistent_workers=True)

        callback = ModelCheckpoint(
            dirpath=save_path, filename="diffusion_bc-{step}",
            every_n_train_steps=args.save_interval)

        trainer = L.Trainer(
            accelerator='gpu', devices=[args.device_id,],
            max_steps=args.bc_training_steps, deterministic=True, log_every_n_steps=1000,
            default_root_dir=save_path, callbacks=[callback])

        trainer.fit(actor, dataloader)

    elif args.mode == "iql_training":

        iql = IQL(obs_dim, act_dim, tau=args.task.iql_tau,
                  discount=0.99, hidden_dim=256)

        dataloader = DataLoader(
            IQL_Wrapper(dataset),
            batch_size=256, shuffle=True, num_workers=4, persistent_workers=True)

        callback = ModelCheckpoint(
            dirpath=save_path, filename="iql-{step}",
            every_n_train_steps=args.save_interval)

        trainer = L.Trainer(
            accelerator='gpu', devices=[args.device_id,],
            max_steps=args.iql_training_steps, deterministic=True, log_every_n_steps=1000,
            default_root_dir=save_path, callbacks=[callback])

        trainer.fit(iql, dataloader)

    elif args.mode == "inference":

        num_envs = args.num_envs
        num_episodes = args.num_episodes
        num_candidates = args.num_candidates

        if args.iql_from_pretrain:
            iql, _ = IQL.from_pretrained(
                env_name, normalize_reward=args.normalize_reward, reward_tune="iql")
        else:
            iql = IQL.load_from_checkpoint(
                checkpoint_path=save_path / "iql-step=300000.ckpt",
                obs_dim=obs_dim, act_dim=act_dim, tau=args.task.iql_tau,
                hidden_dim=256, discount=0.99)
        iql.to(f"cuda:{args.device_id}").eval()

        actor = ContinuousDiffusionSDE.load_from_checkpoint(
            checkpoint_path=save_path / "diffusion_bc-step=500000.ckpt",
            nn_diffusion=nn_diffusion, nn_condition=nn_condition, ema_rate=0.9999,
            x_max=+1.0*torch.ones((act_dim, )),
            x_min=-1.0*torch.ones((act_dim, )))
        actor.to(f"cuda:{args.device_id}").eval()

        env_eval = gym.vector.make(env_name, num_envs=num_envs)
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        prior = torch.zeros((num_envs * num_candidates, act_dim))
        for i in range(num_episodes):

            obs, ep_reward, all_done, t = env_eval.reset(), 0., False, 0

            while not np.all(all_done) and t < 1000:

                obs = torch.tensor(normalizer.normalize(
                    obs), dtype=torch.float32, device=f"cuda:{args.device_id}")
                obs = einops.repeat(obs, "b d -> (b k) d", k=num_candidates)

                act, log = actor.sample(
                    prior, solver=args.solver, n_samples=num_envs * num_candidates,
                    sample_steps=args.sampling_steps,
                    condition_cfg=obs, w_cfg=1.0,
                    sample_step_schedule="quad_continuous")

                with torch.no_grad():
                    q = iql.q(obs, act)
                    v = iql.v(obs)
                    act = einops.rearrange(
                        act, "(b k) d -> b k d", k=num_candidates)
                    adv = einops.rearrange(
                        (q - v), "(b k) 1 -> b k 1", k=num_candidates)
                    w = torch.softmax(
                        adv * args.task.weight_temperature, dim=1)

                    idx = torch.multinomial(
                        w.squeeze(-1), num_samples=1).squeeze(-1)
                    act = act[torch.arange(num_envs), idx].cpu().numpy()

                obs, rew, done, _ = env_eval.step(act)

                t += 1
                done = np.logical_and(done, t < 1000)
                all_done = np.logical_or(all_done, done)
                if "kitchen" in env_name:
                    ep_reward = np.clip(ep_reward + rew, 0., 4.)
                    print(f'[t={t}] finished tasks: {np.around(ep_reward)}')
                elif "antmaze" in env_name:
                    ep_reward = np.clip(ep_reward + rew, 0., 1.)
                    print(f'[t={t}] xy: {np.around(obs[:, :2], 2)}')
                    print(f'[t={t}] reached goal: {np.around(ep_reward)}')
                else:
                    ep_reward += (rew * (1 - all_done))
                    print(
                        f'[t={t}] rew: {np.around((rew * (1 - all_done)), 2)}')

                if np.all(all_done):
                    break

            episode_rewards.append(ep_reward)

        episode_rewards = [
            list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
        episode_rewards = np.array(episode_rewards).mean(-1) * 100.
        print(episode_rewards.mean(), episode_rewards.std())

        env_eval.close()


if __name__ == "__main__":
    pipeline()
