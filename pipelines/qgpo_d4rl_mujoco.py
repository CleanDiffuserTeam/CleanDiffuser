import argparse
import os
from copy import deepcopy

import d4rl
import einops
import gym
import hydra
import numpy as np
import torch
import tqdm
from numpy import ndarray
from torch.utils.data import DataLoader

from cleandiffuser.classifier import QGPOClassifier
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.nn_classifier import QGPONNClassifier
from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.nn_diffusion import SfBCUNet
from cleandiffuser.utils import loop_dataloader, set_seed
from cleandiffuser.utils.iql import TwinQ


class SupportedActionD4RLMuJoCoTDDataset(D4RLMuJoCoTDDataset):
    
    def __init__(self, dataset: torch.Dict[str, ndarray], normalize_reward: bool = False, K: int = 16):
        super().__init__(dataset, normalize_reward)
        self.supported_act = torch.empty((self.size, K, self.a_dim), dtype=torch.float32)
        
    def __getitem__(self, idx: int):
        data = {
            'obs': {
                'state': self.obs[idx], },
            'next_obs': {
                'state': self.next_obs[idx], },
            'act': self.act[idx],
            'supported_act': self.supported_act[idx],
            'rew': self.rew[idx],
            'tml': self.tml[idx], }
        return data

@hydra.main(config_path="../configs/qgpo/mujoco", config_name="mujoco", version_base=None)
def pipeline(args):
    
    set_seed(args.seed)
    device, env_name = args.device, args.task.env_name

    mode = args.mode
    K = args.K
    betaQ = args.betaQ
    beta = args.beta
    
    save_path = f'results/{args.pipeline_name}/{env_name}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    
    env = gym.make(env_name)
    dataset = SupportedActionD4RLMuJoCoTDDataset(d4rl.qlearning_dataset(env), True)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim
    
    nn_diffusion = SfBCUNet(act_dim)
    nn_condition = MLPCondition(obs_dim, 64, [64, ], torch.nn.SiLU())
    
    actor = ContinuousDiffusionSDE(
        nn_diffusion, nn_condition, ema_rate=args.ema_rate,
        x_max=+1.*torch.ones((act_dim,)),
        x_min=-1.*torch.ones((act_dim,)), device=device)
    
    if mode == "bc_training":
        
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4, persistent_workers=True)
        
        n_gradient_step = 0
        avg_bc_loss = 0.
        for batch in loop_dataloader(dataloader):
            
            obs = batch["obs"]["state"].to(device)
            act = batch["act"].to(device)
            
            avg_bc_loss += actor.update(act, obs)["loss"]
            
            n_gradient_step += 1
            if n_gradient_step % args.log_interval == 0:
                print(f"Gradient step: {n_gradient_step}, BC Loss: {avg_bc_loss / args.log_interval:.6f}")
                avg_bc_loss = 0.
            
            if n_gradient_step % args.save_interval == 0:
                actor.save(save_path + f'diffusion_ckpt_{n_gradient_step}.pt')
                actor.save(save_path + f'diffusion_ckpt_latest.pt')
            
            if n_gradient_step >= args.bc_gradient_steps:
                break
    
    elif mode == "supported_action_collecting":
        
        actor.load(save_path + 'diffusion_ckpt_latest.pt')
        
        batch_size = 5000
        
        prior = torch.zeros((batch_size * K, act_dim), dtype=torch.float32)
        for idx in tqdm.trange(0, dataset.size, batch_size):
            
            obs = dataset.next_obs[idx:idx + batch_size].to(device)
            obs = einops.repeat(obs, "b d -> (b k) d", k=K)
            
            n_samples = obs.size(0)
            
            act, _ = actor.sample(
                prior[:n_samples], solver="ddpm", n_samples=n_samples, sample_steps=10,
                sample_step_schedule="quad_continuous",
                condition_cfg=obs, w_cfg=1.0)
            act = einops.rearrange(act, "(b k) d -> b k d", k=K)
            
            dataset.supported_act[idx:idx + batch_size] = act.cpu()
            
        assert idx + batch_size >= dataset.size
        
        torch.save(dataset.supported_act, save_path + 'supported_act.pt')
        
    elif mode == "q_training":
        
        dataset.supported_act = torch.load(save_path + 'supported_act.pt')
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4, persistent_workers=True)
        
        q = TwinQ(obs_dim, act_dim, 256).to(device)
        q_targ = deepcopy(q).eval().requires_grad_(False).to(device)
        optim = torch.optim.Adam(q.parameters(), lr=3e-4)
        
        n_gradient_step = 0
        log = {"q_loss": 0., "td_target": 0.}
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
            if n_gradient_step % args.log_interval == 0:
                print(f"Gradient step: {n_gradient_step}, Q Loss: {log['q_loss'] / args.log_interval:.6f}, TD Target: {log['td_target'] / 1000:.2f}")
                log = {"q_loss": 0., "td_target": 0.}
            
            if n_gradient_step % args.save_interval == 0:                
                torch.save(q.state_dict(), save_path + f'q_ckpt_{n_gradient_step}.pt')
                torch.save(q.state_dict(), save_path + f'q_ckpt_latest.pt')
            
            if n_gradient_step >= args.q_gradient_steps:
                break
    
    elif mode == "cep_training":
        
        nn_classifier = QGPONNClassifier(obs_dim, act_dim, 64, [256, 256, 256], "untrainable_fourier")
        clf = QGPOClassifier(nn_classifier, ema_rate=args.ema_rate, device=device, optim_params={"lr": 1e-3})
        
        q = TwinQ(obs_dim, act_dim, 256).to(device).eval().requires_grad_(False)
        q.load_state_dict(torch.load(save_path + 'q_ckpt_latest.pt', map_location=device))
        
        dataset.supported_act = torch.load(save_path + 'supported_act.pt')
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4, persistent_workers=True)
        
        n_gradient_step = 0
        log = dict.fromkeys(["loss", "f_max", "f_mean", "f_min"], 0.)
        for batch in loop_dataloader(dataloader):
            
            obs = batch["next_obs"]["state"].to(device)
            supported_act = batch["supported_act"].to(device)
            
            pred_q = q(obs.unsqueeze(1).repeat(1, K, 1), supported_act)

            noisy_act, t, _ = actor.add_noise(supported_act)
            
            soft_label = torch.softmax(beta * pred_q, 1)
            
            _log = clf.update(noisy_act, t, {"soft_label": soft_label, "obs": obs})
            for key in log.keys():
                log[key] = log.get(key, 0.) + _log[key]
                
            n_gradient_step += 1
            if n_gradient_step % args.log_interval == 0:
                for key in log.keys():
                    log[key] /= args.log_interval
                print(f"Gradient step: {n_gradient_step}, {log}")
                log = dict.fromkeys(["loss", "f_max", "f_mean", "f_min"], 0.)
            
            if n_gradient_step % args.save_interval == 0:
                clf.save(save_path + f'clf_ckpt_{n_gradient_step}.pt')
                clf.save(save_path + f'clf_ckpt_latest.pt')
                
            if n_gradient_step >= args.cep_gradient_steps:
                break
            
    elif mode == "inference":
        
        num_envs = args.num_envs
        num_episodes = args.num_episodes
        
        num_candidates = 16 if env_name == "hopper-medium-replay-v2" else 1
        sampling_steps = 10 if env_name == "hopper-medium-replay-v2" else args.sampling_steps
        ckpt = "800000" if env_name == "hopper-medium-replay-v2" else "latest"
        
        nn_classifier = QGPONNClassifier(obs_dim, act_dim, 64, [256, 256, 256], "untrainable_fourier")
        clf = QGPOClassifier(nn_classifier, ema_rate=args.ema_rate, device=device)
        clf.load(save_path + 'clf_ckpt_latest.pt')
        
        actor.load(save_path + f'diffusion_ckpt_{ckpt}.pt')
        actor.classifier = clf
        actor.eval()
        
        env_eval = gym.vector.make(env_name, num_envs)
        normalizer = dataset.get_normalizer()
        episode_rewards = []
        
        prior = torch.zeros((num_envs * num_candidates, act_dim))
        for i in range(num_episodes):
        
            obs, ep_reward, all_done, t = env_eval.reset(), 0., 0., 0

            while not np.all(all_done) and t < 1000 + 1:

                obs = torch.tensor(normalizer.normalize(obs), dtype=torch.float32).to(device)
                obs = einops.repeat(obs, "b d -> (b k) d", k=num_candidates)
                act, log = actor.sample(
                    prior, "ddpm", n_samples=num_envs * num_candidates, 
                    sample_steps=sampling_steps, 
                    sample_step_schedule="quad_continuous",
                    condition_cfg=obs, w_cfg=1.0,
                    condition_cg=obs, w_cg=args.task.w_cg)

                logp = einops.rearrange(log["log_p"], "(b k) 1 -> b k", k=num_candidates)
                idx = torch.multinomial(torch.softmax(logp, 1), 1).squeeze(-1)
                act = einops.rearrange(act, "(b k) d -> b k d", k=num_candidates)
                act = act[torch.arange(num_envs), idx]
                    
                obs, rew, done, info = env_eval.step(act.cpu().numpy())
                
                t += 1
                all_done = done if all_done is None else np.logical_or(all_done, done)
                ep_reward += (rew * (1 - all_done)) if t < 1000 else rew
                print(f'[t={t}] rew: {np.around((rew * (1 - all_done)) if t < 1000 else rew, 2)}')

                if np.all(all_done):
                    break
                
            episode_rewards.append(ep_reward)
            
        episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
        episode_rewards = np.array(episode_rewards)
        print(np.mean(episode_rewards, -1), np.std(episode_rewards, -1))

if __name__ == "__main__":
    pipeline()
