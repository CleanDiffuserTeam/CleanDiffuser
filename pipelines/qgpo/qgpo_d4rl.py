"""
WARNING: This pipeline has not been fully tested. The results may not be accurate.
You may tune the hyperparameters in the config file before using it.
"""

from copy import deepcopy
from pathlib import Path

import d4rl
import einops
import gym
import numpy as np
import pytorch_lightning as L
import torch
import torch.utils
import torch.utils.data
import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint

from cleandiffuser.classifier import QGPOClassifier
from cleandiffuser.dataset.d4rl_antmaze_dataset import D4RLAntmazeTDDataset
from cleandiffuser.dataset.d4rl_kitchen_dataset import D4RLKitchenTDDataset
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.nn_classifier import QGPONNClassifier
from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.nn_diffusion import SfBCUNet
from cleandiffuser.utils import DatasetWrapper, TwinQ, loop_dataloader


class BC_Wrapper(DatasetWrapper):
    def __init__(self, dataset):
        super().__init__(dataset)

    def __getitem__(self, idx):
        batch = self.dataset[idx]
        return {
            "x0": batch["act"],
            "condition_cfg": batch["obs"]["state"],
        }


class SupportedActionWrapper(DatasetWrapper):
    def __init__(self, dataset, supported_act):
        super().__init__(dataset)
        self.supported_act = supported_act

    def __getitem__(self, idx):
        return {
            "obs": {
                "state": self.obs[idx],
            },
            "next_obs": {
                "state": self.next_obs[idx],
            },
            "act": self.act[idx],
            "supported_act": self.supported_act[idx],
            "rew": self.rew[idx],
            "tml": self.tml[idx],
        }


if __name__ == "__main__":
    L.seed_everything(42, workers=True)

    env_name = "halfcheetah-medium-v2"
    mode = "inference"
    device = "cuda:4"
    K = 16
    save_path = Path(__file__).parents[1] / f"results/qgpo_d4rl/{env_name}/"

    # --- Create Dataset ---
    env = gym.make(env_name)
    raw_dataset = d4rl.qlearning_dataset(env)
    if "kitchen" in env_name:
        dataset = D4RLKitchenTDDataset(raw_dataset)
    elif "antmaze" in env_name:
        dataset = D4RLAntmazeTDDataset(raw_dataset, reward_tune="iql")
    else:
        dataset = D4RLMuJoCoTDDataset(raw_dataset, normalize_reward=True)
    obs_dim, act_dim = dataset.obs_dim, dataset.act_dim

    nn_diffusion = SfBCUNet(x_dim=act_dim, emb_dim=64, timestep_emb_type="untrainable_fourier")
    nn_condition = MLPCondition(in_dim=obs_dim, out_dim=64, hidden_dims=64, dropout=0.0)

    actor = ContinuousDiffusionSDE(
        nn_diffusion,
        nn_condition,
        ema_rate=0.999,
        x_max=torch.full((act_dim,), 1.0),
        x_min=torch.full((act_dim,), -1.0),
    )

    if mode == "bc_training":
        dataloader = torch.utils.data.DataLoader(
            BC_Wrapper(dataset), batch_size=512, shuffle=True, num_workers=2, persistent_workers=True
        )

        callback = ModelCheckpoint(dirpath=save_path, filename="diffusion_bc-{step}", every_n_train_steps=5_000)

        trainer = L.Trainer(
            accelerator="gpu",
            devices=[0, 1, 2, 3],
            max_steps=5_000,
            deterministic=True,
            log_every_n_steps=500,
            default_root_dir=save_path,
            callbacks=[callback],
        )

        trainer.fit(actor, dataloader)

    elif mode == "supported_action_collecting":
        actor.load_state_dict(torch.load(save_path / "diffusion_bc-step=5000.ckpt", map_location=device)["state_dict"])
        actor.eval().to(device)

        batch_size = 5000

        supported_act = torch.empty((len(dataset), K, act_dim))

        prior = torch.zeros((batch_size * K, act_dim))
        for idx in tqdm.trange(0, len(dataset), batch_size):
            obs = dataset.next_obs[idx : idx + batch_size].to(device)
            obs = einops.repeat(obs, "b d -> (b k) d", k=K)

            n_samples = obs.size(0)

            act, _ = actor.sample(
                prior[:n_samples],
                solver="ddpm",
                sample_steps=10,
                condition_cfg=obs,
                w_cfg=1.0,
            )
            act = einops.rearrange(act, "(b k) d -> b k d", k=K)

            supported_act[idx : idx + batch_size] = act.cpu()

        assert idx + batch_size >= dataset.size

        torch.save(supported_act, save_path / "supported_act.pt")

    elif mode == "q_training":
        betaQ = 1.0

        sa_dataset = SupportedActionWrapper(dataset, torch.load(save_path / "supported_act.pt"))

        dataloader = torch.utils.data.DataLoader(
            sa_dataset, batch_size=256, shuffle=True, num_workers=4, persistent_workers=True
        )

        q = TwinQ(obs_dim, act_dim, 256).to(device)
        q_targ = deepcopy(q).eval().requires_grad_(False).to(device)
        optim = torch.optim.Adam(q.parameters(), lr=3e-4)

        n_gradient_step = 0
        log = {"q_loss": 0.0, "td_target": 0.0}
        for batch in loop_dataloader(dataloader):
            obs = batch["obs"]["state"].to(device)
            act = batch["act"].to(device)
            next_obs = batch["next_obs"]["state"].to(device)
            rew = batch["rew"].to(device)
            tml = batch["tml"].to(device)
            supported_act = batch["supported_act"].to(device)

            with torch.no_grad():
                next_q = q_targ(next_obs.unsqueeze(1).repeat(1, K, 1), supported_act)
                weight = torch.softmax(betaQ * next_q, 1)
                td_target = rew + 0.99 * (1 - tml) * (next_q * weight).sum(1)

            q1, q2 = q.both(obs, act)

            loss = ((q1 - td_target) ** 2 + (q2 - td_target) ** 2).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

            for p, p_targ in zip(q.parameters(), q_targ.parameters()):
                p_targ.data = 0.995 * p_targ.data + 0.005 * p.data

            log["q_loss"] += loss.item()
            log["td_target"] += td_target.mean().item()

            n_gradient_step += 1
            if n_gradient_step % 100 == 0:
                print(
                    f"Gradient step: {n_gradient_step}, Q Loss: {log['q_loss'] / 100:.6f}, TD Target: {log['td_target'] / 100:.2f}"
                )
                log = {"q_loss": 0.0, "td_target": 0.0}

            if n_gradient_step % 5000 == 0:
                torch.save(q.state_dict(), save_path / f"q_ckpt-step={n_gradient_step}.pt")

            if n_gradient_step >= 5000:
                break

    elif mode == "cep_training":
        beta = 3.0

        actor.to(device)
        nn_classifier = QGPONNClassifier(obs_dim, act_dim, 64, [256, 256, 256], "untrainable_fourier")
        clf = QGPOClassifier(nn_classifier, ema_rate=0.999).to(device)

        q = TwinQ(obs_dim, act_dim, 256).to(device).eval().requires_grad_(False)
        q.load_state_dict(torch.load(save_path / "q_ckpt-step=5000.pt", map_location=device))

        sa_dataset = SupportedActionWrapper(dataset, torch.load(save_path / "supported_act.pt"))

        dataloader = torch.utils.data.DataLoader(
            sa_dataset, batch_size=256, shuffle=True, num_workers=4, persistent_workers=True
        )

        n_gradient_step = 0
        log = dict.fromkeys(["loss", "f_max", "f_mean", "f_min"], 0.0)
        for batch in loop_dataloader(dataloader):
            obs = batch["next_obs"]["state"].to(device)
            supported_act = batch["supported_act"].to(device)

            pred_q = q(obs.unsqueeze(1).repeat(1, K, 1), supported_act)

            noisy_act, t, _ = actor.add_noise(supported_act)

            soft_label = torch.softmax(beta * pred_q, 1)

            _log = clf.update(noisy_act, t, {"soft_label": soft_label, "obs": obs})
            for key in log.keys():
                log[key] = log.get(key, 0.0) + _log[key]

            n_gradient_step += 1
            if n_gradient_step % 100 == 0:
                for key in log.keys():
                    log[key] /= 100
                print(f"Gradient step: {n_gradient_step}, {log}")
                log = dict.fromkeys(["loss", "f_max", "f_mean", "f_min"], 0.0)

            if n_gradient_step % 5000 == 0:
                clf.save(save_path / f"clf_ckpt-step={n_gradient_step}.pt")

            if n_gradient_step >= 5000:
                break

    elif mode == "inference":
        num_envs = 50
        num_episodes = 1

        num_candidates = 1
        sampling_steps = 10

        nn_classifier = QGPONNClassifier(obs_dim, act_dim, 64, [256, 256, 256], "untrainable_fourier")
        clf = QGPOClassifier(nn_classifier, ema_rate=0.999)
        clf.load(save_path / "clf_ckpt-step=5000.pt")

        actor.load_state_dict(torch.load(save_path / "diffusion_bc-step=5000.ckpt", map_location=device)["state_dict"])
        actor.classifier = clf
        actor.eval().to(device)

        env_eval = gym.vector.make(env_name, num_envs)
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        prior = torch.zeros((num_envs * num_candidates, act_dim))
        for i in range(num_episodes):
            obs, ep_reward, all_done, t = env_eval.reset(), 0.0, 0.0, 0

            while not np.all(all_done) and t < 1000 + 1:
                obs = torch.tensor(normalizer.normalize(obs), dtype=torch.float32).to(device)
                obs = einops.repeat(obs, "b d -> (b k) d", k=num_candidates)
                act, log = actor.sample(
                    prior,
                    solver="ddpm",
                    sample_steps=sampling_steps,
                    condition_cfg=obs,
                    w_cfg=1.0,
                    condition_cg=obs,
                    w_cg=0.1,
                )

                logp = einops.rearrange(log["log_p"], "(b k) 1 -> b k", k=num_candidates)
                idx = torch.multinomial(torch.softmax(logp, 1), 1).squeeze(-1)
                act = einops.rearrange(act, "(b k) d -> b k d", k=num_candidates)
                act = act[torch.arange(num_envs), idx]

                obs, rew, done, info = env_eval.step(act.cpu().numpy())

                t += 1
                all_done = done if all_done is None else np.logical_or(all_done, done)
                ep_reward += (rew * (1 - all_done)) if t < 1000 else rew
                print(f"[t={t}] rew: {np.around((rew * (1 - all_done)) if t < 1000 else rew, 2)}")

                if np.all(all_done):
                    break

            episode_rewards.append(ep_reward)

        episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
        episode_rewards = np.array(episode_rewards)
        print(np.mean(episode_rewards, -1), np.std(episode_rewards, -1))
