import hydra
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import gym
import pathlib
import time
import numpy as np
import torch
import torch.nn as nn
from utils import set_seed, Logger
from torch.optim.lr_scheduler import CosineAnnealingLR

from cleandiffuser.env import pusht
from cleandiffuser.env.wrapper import VideoRecordingWrapper, MultiStepWrapper
from cleandiffuser.env.utils import VideoRecorder
from cleandiffuser.dataset.pusht_dataset import PushTStateDataset, PushTKeypointDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.utils import report_parameters
    

def make_env(args, idx):
    def thunk():
        env = gym.make(args.env_name)  
        video_recorder = VideoRecorder.create_h264(
                            fps=10,
                            codec='h264',
                            input_pix_fmt='rgb24',
                            crf=22,
                            thread_type='FRAME',
                            thread_count=1
                        )
        env = VideoRecordingWrapper(env, video_recorder, file_path=None, steps_per_render=1)
        env = MultiStepWrapper(env, n_obs_steps=args.obs_steps, n_action_steps=args.action_steps, max_episode_steps=args.max_episode_steps)
        # env.seed(args.seed+idx)
        print("Env seed: ", args.seed+idx)
        return env

    return thunk


def inference(args, envs, dataset, agent, logger):
    """Evaluate a trained agent and optionally save a video."""
    # ---------------- Start Rollout ----------------
    episode_rewards = []
    episode_steps = []
    episode_success = []
    
    if args.diffusion == "ddpm":
        solver = None
    elif args.diffusion == "ddim":
        solver = "ddim"
    elif args.diffusion == "dpm":
        solver = "ode_dpmpp_2"
    elif args.diffusion == "edm":
        solver = "euler"
        
    for i in range(args.eval_episodes // args.num_envs): 
        ep_reward = [0.0] * args.num_envs
        step_reward = []
        obs, t = envs.reset(), 0

        # initialize video stream
        if args.save_video:
            logger.video_init(envs.envs[0], enable=True, video_id=str(i))  # save videos

        while t < args.max_episode_steps:
            if args.env_name == 'pusht-v0':
                obs_seq = obs.astype(np.float32)  # (num_envs, obs_steps, obs_dim)
                # normalize obs
                nobs = dataset.normalizer['obs']['state'].normalize(obs_seq)
                nobs = torch.tensor(nobs, device=args.device, dtype=torch.float32)  # (num_envs, obs_steps, obs_dim)
            elif args.env_name == 'pusht-keypoints-v0':
                # Note: keypoint env return 40 dim but need 20 dim
                obs_seq = obs[:, :args.obs_steps, :].astype(np.float32)  # (num_envs, obs_steps, obs_dim) 
                keypoint_obs_seq = obs_seq[:, :, :18]  # (num_envs, obs_steps, 18)
                agentpos_obs_seq = obs_seq[:, :, 18:20]  # (num_envs, obs_steps, 2)
                # normalize obs
                # keypoint
                keypoint_pos_dim = 18
                keypoint = keypoint_obs_seq.reshape(-1, 2)  # (num_envs*obs_staps*9, 2)
                nkeypoint = dataset.normalizer['obs']['keypoint'].normalize(keypoint)  # (num_envs*obs_staps*9, 2)
                nkeypoint = nkeypoint.reshape(args.num_envs, args.obs_steps, keypoint_pos_dim)  # (num_envs, obs_staps, 18)
                # agent_pos
                nagent_pos = dataset.normalizer['obs']['agent_pos'].normalize(agentpos_obs_seq)  # (num_envs, obs_steps, 2)
                nobs = np.concatenate((nkeypoint, nagent_pos), axis=-1)  # (num_envs, obs_steps, obs_dim)
                nobs = torch.tensor(nobs, device=args.device, dtype=torch.float32)  # (num_envs, obs_steps, obs_dim)
            with torch.no_grad():
                if args.nn == 'chi_unet' or args.nn == 'dit':
                    # reshape observation to (num_envs, obs_steps*obs_dim)  
                    condition = nobs.flatten(start_dim=1)
                else:
                    # reshape observation to (num_envs, obs_steps, obs_dim)  
                    condition = nobs
                # run sampling (num_envs, horizon, action_dim)
                prior = torch.zeros((args.num_envs, args.horizon, args.action_dim), device=args.device)
                naction, _ = agent.sample(prior=prior, n_samples=args.num_envs, sample_steps=args.sample_steps, solver=solver, condition_cfg=condition, w_cfg=1.0, use_ema=True)
                
            # unnormalize prediction
            naction = naction.detach().to('cpu').numpy()  # (num_envs, horizon, action_dim)
            action_pred = dataset.normalizer['action'].unnormalize(naction)  
            
            # get action
            start = args.obs_steps - 1
            end = start + args.action_steps
            action = action_pred[:, start:end, :]
            
            obs, reward, done, info = envs.step(action)
            ep_reward += reward
            step_reward.append(reward)
            t += args.action_steps
        
        ep_reward = np.around(np.array(ep_reward), 2)
        success = np.around(np.max(np.array(step_reward), axis=0), 2)
        print(f"[Episode {1+i*(args.num_envs)}-{(i+1)*(args.num_envs)}] reward: {ep_reward} success:{success}")
        episode_rewards.append(ep_reward)
        episode_steps.append(t)
        episode_success.append(success)
    print(f"Mean step: {np.nanmean(episode_steps)} Mean reward: {np.nanmean(episode_rewards)} Mean success: {np.nanmean(episode_success)}")
    return {'mean_step': np.nanmean(episode_steps), 'mean_reward': np.nanmean(episode_rewards), 'mean_success': np.nanmean(episode_success)}


@hydra.main(config_path="../configs/dp/pusht/dit", config_name="pusht")
def pipeline(args):
    # ---------------- Create Logger ----------------
    set_seed(args.seed)
    logger = Logger(pathlib.Path(args.work_dir), args)

    # ---------------- Create Environment ----------------
    envs = gym.vector.SyncVectorEnv(
        [make_env(args, idx) for idx in range(args.num_envs)],
    )
        
    # ---------------- Create Dataset ----------------
    dataset_path = os.path.expanduser(args.dataset_path)
    if args.env_name == 'pusht-v0':
        dataset = PushTStateDataset(dataset_path, horizon=args.horizon, obs_keys=args.obs_keys, 
                                pad_before=args.obs_steps-1, pad_after=args.action_steps-1, abs_action=args.abs_action)
    elif args.env_name == 'pusht-keypoints-v0':
        dataset = PushTKeypointDataset(dataset_path, horizon=args.horizon, obs_keys=args.obs_keys, 
                                pad_before=args.obs_steps-1, pad_after=args.action_steps-1, abs_action=args.abs_action)
    print(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )
    
    # --------------- Create Diffusion Model -----------------
    if args.nn == "dit":
        from cleandiffuser.nn_condition import MLPCondition
        from cleandiffuser.nn_diffusion import DiT1d
        
        nn_diffusion = DiT1d(
            args.action_dim, emb_dim=128, d_model=320, n_heads=10, depth=2, timestep_emb_type="fourier").to(args.device)
        nn_condition = MLPCondition(
            in_dim=args.obs_steps*args.obs_dim, out_dim=128, hidden_dims=[256, ], act=nn.ReLU(), dropout=0.0).to(args.device)
    elif args.nn == "chi_unet":
        from cleandiffuser.nn_condition import IdentityCondition
        from cleandiffuser.nn_diffusion import ChiUNet1d

        nn_diffusion = ChiUNet1d(
            args.action_dim, args.obs_dim, args.obs_steps, model_dim=256, emb_dim=256, dim_mult=[1, 2, 2],
            obs_as_global_cond=True, timestep_emb_type="positional").to(args.device)
        # dropout=0.0 to use no CFG but serve as FiLM encoder
        nn_condition = IdentityCondition(dropout=0.0).to(args.device)
    elif args.nn == "chi_transformer":
        from cleandiffuser.nn_condition import IdentityCondition
        from cleandiffuser.nn_diffusion import ChiTransformer
            
        nn_diffusion = ChiTransformer(
            args.action_dim, args.obs_dim, args.horizon, args.obs_steps, d_model=256, nhead=4, num_layers=4,
            timestep_emb_type="positional").to(args.device)
        # dropout=0.0 to use no CFG but serve as FiLM encoder
        nn_condition = IdentityCondition(dropout=0.0).to(args.device)
    else:
        raise ValueError(f"Invalid nn type {args.nn}")
    
    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"==============================================================================")

    if args.diffusion == "ddpm":
        from cleandiffuser.diffusion.ddpm import DDPM
        x_max = torch.ones((1, args.horizon, args.action_dim), device=args.device) * +1.0
        x_min = torch.ones((1, args.horizon, args.action_dim), device=args.device) * -1.0
        agent = DDPM(
            nn_diffusion=nn_diffusion, nn_condition=nn_condition, device=args.device,
            diffusion_steps=args.sample_steps, x_max=x_max, x_min=x_min,
            optim_params={"lr": args.lr})
    elif args.diffusion == "edm":
        from cleandiffuser.diffusion.edm import EDM
        agent = EDM(nn_diffusion=nn_diffusion, nn_condition=nn_condition, device=args.device,
                    optim_params={"lr": args.lr})
    else:
        raise NotImplementedError
    lr_scheduler = CosineAnnealingLR(agent.optimizer, T_max=args.gradient_steps)
    
    if args.mode == "train":
        # ----------------- Training ----------------------
        n_gradient_step = 0
        diffusion_loss_list = []
        start_time = time.time()
        for batch in loop_dataloader(dataloader):

            # preprocess 
            if args.env_name == 'pusht-v0':
                nobs = batch['obs']['state'].to(args.device)
                naction = batch['action'].to(args.device)
            elif args.env_name == 'pusht-keypoints-v0':
                nkeypoint = batch["obs"]["keypoint"].to(args.device)
                nagent_pos = batch["obs"]["agent_pos"].to(args.device)
                nobs = torch.cat([nkeypoint, nagent_pos], dim=-1)  
                naction = batch["action"].to(args.device)
            
            # get condition
            condition = nobs[:, :args.obs_steps, :]  # (B, obs_horizon, obs_dim)
            if args.nn == 'chi_unet' or args.nn == 'dit':
                condition = condition.flatten(start_dim=1)  # (B, obs_horizon*obs_dim)
            else:
                pass  # (B, obs_horizon, obs_dim)

            # update diffusion
            diffusion_loss = agent.update(naction, condition)['loss']
            lr_scheduler.step()
            diffusion_loss_list.append(diffusion_loss)

            if n_gradient_step % args.log_freq == 0:
                metrics = {
                    'step': n_gradient_step,
                    'total_time': time.time() - start_time,
                    'avg_diffusion_loss': np.mean(diffusion_loss_list)
                }
                logger.log(metrics, category='train')
                diffusion_loss_list = []
            
            if n_gradient_step % args.save_freq == 0:
                logger.save_agent(agent=agent, identifier=n_gradient_step)
                
            if n_gradient_step % args.eval_freq == 0:
                print("Evaluate model...")
                agent.model.eval()
                agent.model_ema.eval()
                metrics = {'step': n_gradient_step}
                metrics.update(inference(args, envs, dataset, agent, logger))
                logger.log(metrics, category='inference')
                agent.model.train()
                agent.model_ema.train()
            
            n_gradient_step += 1
            if n_gradient_step > args.gradient_steps:
                # finish
                logger.finish(agent)
                break
    elif args.mode == "inference":
        # ----------------- Inference ----------------------
        if args.model_path:
            agent.load(args.model_path)
        else:
            raise ValueError("Empty model for inference")
        agent.model.eval()
        agent.model_ema.eval()

        metrics = {'step': 0}
        metrics.update(inference(args, envs, dataset, agent, logger))
        logger.log(metrics, category='inference')
        
    else:
        raise ValueError("Illegal mode")


if __name__ == "__main__":
    pipeline()









    

