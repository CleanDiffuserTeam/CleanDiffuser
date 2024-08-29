import os
from copy import deepcopy

import d4rl
import gym
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import ContinuousEDM
from cleandiffuser.diffusion.consistency_model import ContinuousConsistencyModel
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import IDQLMlp
from cleandiffuser.utils import (IDQLQNet, IDQLVNet)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    

if __name__ == "__main__":
    
    """
    test consistency distillation (CD):
        iql_training -> edm_training -> cd_training -> inference
    test consistency training (CT):
        iql_training -> ct_training -> inference
    """

    seed = 0
    device = "cuda:0"
    env_name = "halfcheetah-medium-v2"
    weight_temperature = 100. # 10 for me / 100 for m / 400 for mr
    mode = "iql_training"

    set_seed(seed)
    save_path = f'tutorials/results/sp_consistency_policy/{env_name}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
        
    # ---------------------- Create Dataset ----------------------
    env = gym.make(env_name)
    dataset = D4RLMuJoCoTDDataset(d4rl.qlearning_dataset(env), True)
    dataloader = DataLoader(
        dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    # ------------------------------------------------------------

    """ MODE1: IQL Training
    
    In IDQL, the Diffusion model simply behavior clones the dataset 
    and reselects actions during inference through IQL's value estimation. 
    Therefore, we need to train an IQL here.
    """
    if mode == "iql_training":
        
        # Create IQL Networks
        iql_q = IDQLQNet(obs_dim, act_dim).to(device)
        iql_q_target = deepcopy(iql_q).requires_grad_(False).eval()
        iql_v = IDQLVNet(obs_dim).to(device)

        q_optim = torch.optim.Adam(iql_q.parameters(), lr=3e-4)
        v_optim = torch.optim.Adam(iql_v.parameters(), lr=3e-4)
        
        q_lr_scheduler = CosineAnnealingLR(q_optim, T_max=1_000_000)
        v_lr_scheduler = CosineAnnealingLR(v_optim, T_max=1_000_000)
        
        iql_q.train()
        iql_v.train()

        # Begin Training
        n_gradient_step = 0
        log = {"q_loss": 0., "v_loss": 0.}
        for batch in loop_dataloader(dataloader):

            obs, next_obs = batch["obs"]["state"].to(device), batch["next_obs"]["state"].to(device)
            act = batch["act"].to(device)
            rew = batch["rew"].to(device)
            tml = batch["tml"].to(device)

            q = iql_q_target(obs, act)
            v = iql_v(obs)
            v_loss = (torch.abs(0.7 - ((q - v) < 0).float()) * (q - v) ** 2).mean()

            v_optim.zero_grad()
            v_loss.backward()
            v_optim.step()

            with torch.no_grad():
                td_target = rew + 0.99 * (1 - tml) * iql_v(next_obs)
            q1, q2 = iql_q.both(obs, act)
            q_loss = ((q1 - td_target) ** 2 + (q2 - td_target) ** 2).mean()
            q_optim.zero_grad()
            q_loss.backward()
            q_optim.step()

            q_lr_scheduler.step()
            v_lr_scheduler.step()

            for param, target_param in zip(iql_q.parameters(), iql_q_target.parameters()):
                target_param.data.copy_(0.995 * param.data + (1 - 0.995) * target_param.data)

            # Logging
            log["q_loss"] += q_loss.item()
            log["v_loss"] += v_loss.item()

            if (n_gradient_step + 1) % 1000 == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["q_loss"] /= 1000
                log["v_loss"] /= 1000
                print(log)
                log = {"q_loss": 0., "v_loss": 0.}

            # Saving
            if (n_gradient_step + 1) % 200_000 == 0:
                torch.save({
                    "iql_q": iql_q.state_dict(),
                    "iql_q_target": iql_q_target.state_dict(),
                    "iql_v": iql_v.state_dict(),
                }, save_path + f"iql_ckpt_{n_gradient_step + 1}.pt")
                torch.save({
                    "iql_q": iql_q.state_dict(),
                    "iql_q_target": iql_q_target.state_dict(),
                    "iql_v": iql_v.state_dict(),
                }, save_path + f"iql_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step >= 1_000_000:
                break
            
    elif mode == "edm_training":
        
        """ MODE2: EDM Training

        Consistency Distillation (CD) requires a well-trained EDM backbone. 
        If you only want to test Consistency Training, this step is not necessary.
        """
        
        nn_diffusion = IDQLMlp(obs_dim, act_dim, emb_dim=64, timestep_emb_type="untrainable_fourier")
        nn_condition = IdentityCondition(dropout=0.0)

        actor = ContinuousEDM(
            nn_diffusion, nn_condition, optim_params={"lr": 3e-4},
            x_max=+1. * torch.ones((1, act_dim)),
            x_min=-1. * torch.ones((1, act_dim)),
            ema_rate=0.9999, device=device)
        
        actor_lr_scheduler = CosineAnnealingLR(actor.optimizer, T_max=1_000_000)

        actor.train()

        n_gradient_step = 0
        log = {"bc_loss": 0.}

        for batch in loop_dataloader(dataloader):

            obs = batch["obs"]["state"].to(device)
            act = batch["act"].to(device)

            bc_loss = actor.update(act, obs)["loss"]
            actor_lr_scheduler.step()
            
            # Logging
            log["bc_loss"] += bc_loss

            if (n_gradient_step + 1) % 1000 == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["bc_loss"] /= 1000
                print(log)
                log = {"bc_loss": 0.}

            # Saving
            if (n_gradient_step + 1) % 200_000 == 0:
                actor.save(save_path + f"edm_ckpt_{n_gradient_step + 1}.pt")
                actor.save(save_path + f"edm_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step >= 1_000_000:
                break
    
    elif mode == "cd_training":
        
        """ MODE3: Consistency Distillation

        Train the Consistency Model with a pre-trained EDM.
        """
        
        # Load pre-trained EDM
        nn_diffusion_edm = IDQLMlp(obs_dim, act_dim, emb_dim=64, timestep_emb_type="untrainable_fourier")
        nn_condition_edm = IdentityCondition(dropout=0.0)

        edm_actor = ContinuousEDM(
            nn_diffusion_edm, nn_condition_edm, optim_params={"lr": 3e-4},
            x_max=+1. * torch.ones((1, act_dim)),
            x_min=-1. * torch.ones((1, act_dim)),
            ema_rate=0.9999, device=device)
        
        edm_actor.load(save_path + f"edm_ckpt_latest.pt")
        edm_actor.eval()
        
        # Create Consistency Model
        nn_diffusion = IDQLMlp(obs_dim, act_dim, emb_dim=64, timestep_emb_type="untrainable_fourier")
        nn_condition = IdentityCondition(dropout=0.0)
        
        actor = ContinuousConsistencyModel(
            nn_diffusion, nn_condition, optim_params={"lr": 3e-4}, 
            x_max=+1. * torch.ones((1, act_dim)),
            x_min=-1. * torch.ones((1, act_dim)),
            ema_rate=0.9999, device=device)

        actor.prepare_distillation(edm_actor, distillation_N=18)
        
        actor_lr_scheduler = CosineAnnealingLR(actor.optimizer, T_max=200_000)
        
        actor.train()
        
        n_gradient_step = 0
        log = {"loss": 0.}
        for batch in loop_dataloader(dataloader):
            
            obs = batch["obs"]["state"].to(device)
            act = batch["act"].to(device)

            loss = actor.update(act, obs, loss_type="distillation")["loss"]
            
            log["loss"] += loss
            
            if (n_gradient_step + 1) % 1000 == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["loss"] /= 1000
                print(log)
                log = {"loss": 0.}
            
            if (n_gradient_step + 1) % 200_000 == 0:
                actor.save(save_path + f"cd_ckpt_{n_gradient_step + 1}.pt")
                actor.save(save_path + f"cd_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step >= 200_000:
                break
      
    elif mode == "ct_training":
        
        """ MODE4: Consistency Training

        Train the Consistency Model without relying on any pre-trained Models.
        """
        
        # As suggested in "IMPROVED TECHNIQUES FOR TRAINING CONSISTENCY MODELS", the Fourier scale is set to 0.02 instead of default 16.0.
        nn_diffusion = IDQLMlp(obs_dim, act_dim, emb_dim=64, timestep_emb_type="untrainable_fourier", timestep_emb_params={"scale": 0.02})
        nn_condition = IdentityCondition(dropout=0.0)
            
        actor = ContinuousConsistencyModel(
            nn_diffusion, nn_condition, optim_params={"lr": 3e-4},
            curriculum_cycle=1000000,
            x_max=+1. * torch.ones((1, act_dim)),
            x_min=-1. * torch.ones((1, act_dim)),
            ema_rate=0.9999, device=device)
        
        actor_lr_scheduler = CosineAnnealingLR(actor.optimizer, T_max=1_000_000)

        actor.train()

        n_gradient_step = 0
        log = {"bc_loss": 0., "unweighted_bc_loss": 0.}

        for batch in loop_dataloader(dataloader):

            obs = batch["obs"]["state"].to(device)
            act = batch["act"].to(device)

            # -- Policy Training
            _log = actor.update(act, obs)

            actor_lr_scheduler.step()
            
            # ----------- Logging ------------
            log["bc_loss"] += _log["loss"]
            log["unweighted_bc_loss"] += _log["unweighted_loss"]

            if (n_gradient_step + 1) % 1000 == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["bc_loss"] /= 1000
                log["unweighted_bc_loss"] /= 1000
                log["curriculum_process"] = actor.cur_logger.curriculum_process
                log["Nk"] = actor.cur_logger.Nk
                print(log)
                log = {"bc_loss": 0., "unweighted_bc_loss": 0.}

            # ----------- Saving ------------
            if (n_gradient_step + 1) % 200_000 == 0:
                actor.save(save_path + f"ct_ckpt_{n_gradient_step + 1}.pt")
                actor.save(save_path + f"ct_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step >= 1_000_000:
                break
    
    elif mode == "inference":
        
        """ MODE5: Inference
    
        Test the trained Models.
        """
        
        test_model = "ct"
        num_envs = 50
        num_candidates = 256
        sampling_steps = 1
        
        if test_model == "edm":
            nn_diffusion_edm = IDQLMlp(obs_dim, act_dim, emb_dim=64, timestep_emb_type="untrainable_fourier")
            nn_condition_edm = IdentityCondition(dropout=0.0)

            actor = ContinuousEDM(
                nn_diffusion_edm, nn_condition_edm, optim_params={"lr": 3e-4},
                x_max=+1. * torch.ones((1, act_dim)),
                x_min=-1. * torch.ones((1, act_dim)),
                ema_rate=0.9999, device=device)
            
            actor.load(save_path + f"edm_ckpt_latest.pt")
            actor.eval()
        
        elif test_model == "cd":
            nn_diffusion = IDQLMlp(obs_dim, act_dim, emb_dim=64, timestep_emb_type="untrainable_fourier")
            nn_condition = IdentityCondition(dropout=0.0)

            actor = ContinuousConsistencyModel(
                nn_diffusion, nn_condition, optim_params={"lr": 3e-4},
                x_max=+1. * torch.ones((1, act_dim)),
                x_min=-1. * torch.ones((1, act_dim)),
                ema_rate=0.9999, device=device)
            
            actor.load(save_path + f"cd_ckpt_latest.pt")
            actor.eval()
        
        elif test_model == "ct":
            nn_diffusion = IDQLMlp(obs_dim, act_dim, emb_dim=64, timestep_emb_type="untrainable_fourier", timestep_emb_params={"scale": 0.02})
            nn_condition = IdentityCondition(dropout=0.0)

            actor = ContinuousConsistencyModel(
                nn_diffusion, nn_condition, optim_params={"lr": 3e-4},
                x_max=+1. * torch.ones((1, act_dim)),
                x_min=-1. * torch.ones((1, act_dim)),
                ema_rate=0.9999, device=device)
            
            actor.load(save_path + f"ct_ckpt_latest.pt")
            actor.eval()
        
        else:
            raise ValueError("Invalid test model.")
        
        iql_q = IDQLQNet(obs_dim, act_dim).to(device)
        iql_q_target = deepcopy(iql_q).requires_grad_(False).eval()
        iql_v = IDQLVNet(obs_dim).to(device)
                
        critic_ckpt = torch.load(save_path + f"iql_ckpt_latest.pt")
        iql_q.load_state_dict(critic_ckpt["iql_q"])
        iql_q_target.load_state_dict(critic_ckpt["iql_q_target"])
        iql_v.load_state_dict(critic_ckpt["iql_v"])

        iql_q.eval()
        iql_v.eval()

        env_eval = gym.vector.make(env_name, num_envs)
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        prior = torch.zeros((num_envs * num_candidates, act_dim), device=device)
        for i in range(3):

            obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

            while not np.all(cum_done) and t < 1000 + 1:
                # normalize obs
                obs = torch.tensor(normalizer.normalize(obs), device=device, dtype=torch.float32)
                obs = obs.unsqueeze(1).repeat(1, num_candidates, 1).view(-1, obs_dim)

                # sample actions
                act, log = actor.sample(
                    prior,
                    n_samples=num_envs * num_candidates,
                    sample_steps=sampling_steps,
                    condition_cfg=obs, w_cfg=1.0,
                    use_ema=True, temperature=1.0)

                # resample
                with torch.no_grad():
                    q = iql_q_target(obs, act)
                    v = iql_v(obs)
                    adv = (q - v)
                    adv = adv.view(-1, num_candidates, 1)

                    w = torch.softmax(adv * weight_temperature, 1)
                    act = act.view(-1, num_candidates, act_dim)

                    p = w / w.sum(1, keepdim=True)

                    indices = torch.multinomial(p.squeeze(-1), 1).squeeze(-1)
                    sampled_act = act[torch.arange(act.shape[0]), indices].cpu().numpy()

                # step
                obs, rew, done, info = env_eval.step(sampled_act)

                t += 1
                cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                ep_reward += (rew * (1 - cum_done)) if t < 1000 else rew
                print(f'[t={t}] rew: {np.around((rew * (1 - cum_done)), 2)}')

                if np.all(cum_done):
                    break

            episode_rewards.append(ep_reward)

        episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
        episode_rewards = np.array(episode_rewards)
        print(np.mean(episode_rewards, -1), np.std(episode_rewards, -1))
        
        env_eval.close()
