import os

import d4rl
import gym
import hydra, wandb, uuid
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.classifier import CumRewClassifier
from cleandiffuser.dataset.d4rl_maze2d_dataset import DV_D4RLMaze2DSeqDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader, loop_two_dataloaders
from cleandiffuser.diffusion import ContinuousDiffusionSDE, DiscreteDiffusionSDE
from cleandiffuser.invdynamic import MlpInvDynamic
from cleandiffuser.nn_condition import MLPCondition, IdentityCondition
from cleandiffuser.nn_diffusion import DiT1d, DVInvMlp
from cleandiffuser.nn_classifier import HalfJannerUNet1d
from cleandiffuser.nn_diffusion import JannerUNet1d
from cleandiffuser.utils import report_parameters, DD_RETURN_SCALE, DVHorizonCritic
from utils import set_seed
from tqdm import tqdm
from omegaconf import OmegaConf



@hydra.main(config_path="../configs/veteran/maze2d", config_name="maze2d", version_base=None)
def pipeline(args):
    args.device = args.device if torch.cuda.is_available() else "cpu"
    if args.enable_wandb and args.mode in ["inference", "train"]:
        wandb.require("core")
        print(args)
        wandb.init(
            reinit=True,
            id=str(uuid.uuid4()),
            project=str(args.project),
            group=str(args.group),
            name=str(args.name),
            config=OmegaConf.to_container(args, resolve=True)
        )

    set_seed(args.seed)
    
    # base config
    base_path = f"{args.pipeline_name}_H{args.task.planner_horizon}_Jump{args.task.stride}"
    base_path += f"_next{args.planner_next_obs_loss_weight}"
    # guidance type
    base_path += f"_{args.guidance_type}"
    # For Planner
    base_path += f"_{args.planner_net}"
    if args.planner_net == "transformer":
        base_path += f"_d{args.planner_depth}"
        base_path += f"_width{args.planner_d_model}"
    elif args.planner_net == "unet":
        base_path += f"_width{args.unet_dim}"
    
    if not args.planner_predict_noise:
        base_path += f"_pred_x0"
    
    # pipeline_type
    base_path += f"_{args.pipeline_type}"
    base_path += f"_dp{args.use_diffusion_invdyn}"
    # task name
    base_path += f"/{args.task.env_name}/"
    
    save_path = "results/" + base_path
    video_path = "video_outputs/" + base_path
    
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    
    if os.path.exists(video_path) is False:
        os.makedirs(video_path)

    # ---------------------- Create Dataset ----------------------
    env = gym.make(args.task.env_name)
    planner_dataset = DV_D4RLMaze2DSeqDataset(
        env.get_dataset(), horizon=args.task.planner_horizon, discount=args.reward_mode.discount, 
        continous_reward_at_done=args.reward_mode.continous_reward_at_done, reward_tune=args.reward_mode.reward_tune, 
        stride=args.task.stride, learn_policy=False, center_mapping=(args.guidance_type!="cfg")
    )
    policy_dataset = DV_D4RLMaze2DSeqDataset(
        env.get_dataset(), horizon=args.task.planner_horizon, discount=args.reward_mode.discount, 
        continous_reward_at_done=args.reward_mode.continous_reward_at_done, reward_tune=args.reward_mode.reward_tune, 
        stride=args.task.stride, learn_policy=True, center_mapping=(args.guidance_type!="cfg")
    )
    planner_dataloader = DataLoader(
        planner_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    obs_dim, act_dim = planner_dataset.o_dim, planner_dataset.a_dim
    
    policy_dataloader = DataLoader(
        policy_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    obs_dim, act_dim = planner_dataset.o_dim, planner_dataset.a_dim

    planner_dim = obs_dim if args.pipeline_type=="separate" else obs_dim + act_dim

    # --------------- Network Architecture -----------------
    if args.planner_net == "transformer":
        nn_diffusion_planner = DiT1d(
            planner_dim, emb_dim=args.planner_emb_dim,
            d_model=args.planner_d_model, n_heads=args.planner_d_model//64, depth=args.planner_depth, timestep_emb_type="fourier")
    elif args.planner_net == "unet":
        nn_diffusion_planner = JannerUNet1d(
            planner_dim, model_dim=args.unet_dim, emb_dim=args.unet_dim,
            timestep_emb_type="positional", attention=False, kernel_size=5)
    
    nn_condition_planner = None
    classifier = None
        
    if args.guidance_type == "MCSS":
        # --------------- Horizon Critic -----------------
        critic = DVHorizonCritic(
            planner_dim, emb_dim=args.planner_emb_dim,
            d_model=args.planner_d_model, n_heads=args.planner_d_model//64, depth=2, norm_type="pre").to(args.device)
        critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_learning_rate)
        print(f"=============== Parameter Report of Value ====================================")
        report_parameters(critic)
        print(f"==============================================================================")
        
    elif args.guidance_type=="cfg":
        if args.planner_net == "transformer":
            nn_condition_planner = MLPCondition(
                in_dim=1, out_dim=args.planner_emb_dim, hidden_dims=[args.planner_emb_dim, ], act=nn.SiLU(), dropout=0.25)
        elif args.planner_net == "unet":
            nn_condition_planner = MLPCondition(
                in_dim=1, out_dim=args.unet_dim, hidden_dims=[args.unet_dim, ], act=nn.SiLU(), dropout=0.25)
    
    elif args.guidance_type=="cg":
        nn_classifier = HalfJannerUNet1d(
            args.task.planner_horizon, planner_dim, out_dim=1,
            model_dim=args.unet_dim, emb_dim=args.unet_dim,
            timestep_emb_type="positional", kernel_size=3)
        classifier = CumRewClassifier(nn_classifier, device=args.device)
        print(f"=============== Parameter Report of Classifier ===============================")
        report_parameters(nn_classifier)
        print(f"==============================================================================")

    print(f"=============== Parameter Report of Planner ==================================")
    report_parameters(nn_diffusion_planner)
    print(f"==============================================================================")

    # ----------------- Masking -------------------
    fix_mask = torch.zeros((args.task.planner_horizon, planner_dim))
    fix_mask[0, :obs_dim] = 1.
    loss_weight = torch.ones((args.task.planner_horizon, planner_dim))
    loss_weight[1] = args.planner_next_obs_loss_weight

    # --------------- Diffusion Model with Classifier-Free Guidance --------------------
    planner = ContinuousDiffusionSDE(
        nn_diffusion_planner, nn_condition=nn_condition_planner,
        fix_mask=fix_mask, loss_weight=loss_weight, classifier=classifier, ema_rate=args.planner_ema_rate,
        device=args.device, predict_noise=args.planner_predict_noise, noise_schedule="linear")

    # --------------- Inverse Dynamic (Policy) -------------------
    if args.pipeline_type=="separate":
        if args.use_diffusion_invdyn:
            nn_diffusion_invdyn = DVInvMlp(obs_dim, act_dim, emb_dim=64, hidden_dim=args.policy_hidden_dim, timestep_emb_type="positional").to(args.device)
            nn_condition_invdyn = IdentityCondition(dropout=0.0).to(args.device)
            print(f"=============== Parameter Report of Policy ===================================")
            report_parameters(nn_diffusion_invdyn)
            print(f"==============================================================================")
            # --------------- Diffusion Model Actor --------------------
            policy = DiscreteDiffusionSDE(
                nn_diffusion_invdyn, nn_condition_invdyn, predict_noise=args.policy_predict_noise, optim_params={"lr": args.policy_learning_rate},
                x_max=+1. * torch.ones((1, act_dim), device=args.device),
                x_min=-1. * torch.ones((1, act_dim), device=args.device),
                diffusion_steps=args.policy_diffusion_steps, ema_rate=args.policy_ema_rate, device=args.device)
        else:
            invdyn = MlpInvDynamic(obs_dim, act_dim, 512, nn.Tanh(), {"lr": 2e-4}, device=args.device)

    # ---------------------- Training ----------------------
    if args.mode == "train":
        # Planner
        planner_lr_scheduler = CosineAnnealingLR(planner.optimizer, args.planner_diffusion_gradient_steps)
        planner.train()
        
        # Critic or classifier
        if args.guidance_type=="MCSS":
            critic_lr_scheduler = CosineAnnealingLR(critic_optim, args.planner_diffusion_gradient_steps)
            critic.train()
        elif args.guidance_type=="cg":
            classifier_lr_scheduler = CosineAnnealingLR(planner.classifier.optim, args.planner_diffusion_gradient_steps)
            classifier.train()
        
        # Policy
        if args.pipeline_type=="separate":
            if args.use_diffusion_invdyn:
                policy_lr_scheduler = CosineAnnealingLR(policy.optimizer, args.policy_diffusion_gradient_steps)
                policy.train()
            else:
                invdyn_lr_scheduler = CosineAnnealingLR(invdyn.optim, args.invdyn_gradient_steps)
                invdyn.train()

        n_gradient_step = 0
        log = {
            "val_pred": 0,
            "val_loss": 0,
            "avg_loss_planner": 0, 
            "bc_loss_policy": 0,
            "avg_loss_classifier": 0
        }
        
        pbar = tqdm(total=max(args.planner_diffusion_gradient_steps, args.policy_diffusion_gradient_steps)/args.log_interval)
        for planner_batch, policy_batch in loop_two_dataloaders(planner_dataloader, policy_dataloader):

            planner_horizon_obs = planner_batch["obs"]["state"].to(args.device)
            planner_horizon_action = planner_batch["act"].to(args.device)
            planner_horizon_obs_action = torch.cat([planner_horizon_obs, planner_horizon_action], -1)
            planner_horizon_data = planner_horizon_obs if args.pipeline_type == "separate" else planner_horizon_obs_action
            
            planner_td_val = planner_batch["val"].to(args.device)
            
            policy_horizon_obs = policy_batch["obs"]["state"].to(args.device)
            policy_horizon_action = policy_batch["act"].to(args.device)
            policy_td_obs, policy_td_next_obs, policy_td_act = policy_horizon_obs[:,0,:], policy_horizon_obs[:,1,:], policy_horizon_action[:,0,:]

            # ----------- Planner Gradient Step ------------
            if n_gradient_step <= args.planner_diffusion_gradient_steps:
                if args.guidance_type == "cfg":
                    log["avg_loss_planner"] += planner.update(planner_horizon_data, planner_td_val)['loss']
                else:
                    log["avg_loss_planner"] += planner.update(planner_horizon_data)['loss']
                planner_lr_scheduler.step()
            
            if args.guidance_type=="MCSS":
                # ----------- Horizon Critic Gradient Step ------------    
                if n_gradient_step <= args.planner_diffusion_gradient_steps:
                    val_pred = critic(planner_horizon_data)
                    assert val_pred.shape == planner_td_val.shape
                    critic_loss = F.mse_loss(val_pred, planner_td_val)
                    log["val_pred"] += val_pred.mean().item()
                    log["val_loss"] += critic_loss.item()
                    critic_optim.zero_grad()
                    critic_loss.backward()
                    critic_optim.step()
                    critic_lr_scheduler.step()
      
            elif args.guidance_type=="cg":
                if n_gradient_step <= args.planner_diffusion_gradient_steps:
                    log["avg_loss_classifier"] += planner.update_classifier(planner_horizon_data, planner_td_val)['loss']
                    classifier_lr_scheduler.step()
            
            if args.pipeline_type == "separate":
                if args.use_diffusion_invdyn:
                    # ----------- Policy Gradient Step ------------
                    if n_gradient_step <= args.policy_diffusion_gradient_steps:
                        log["bc_loss_policy"] += policy.update(policy_td_act, torch.cat([policy_td_obs, policy_td_next_obs], dim=-1))['loss']
                        policy_lr_scheduler.step()
                else:    
                    if n_gradient_step <= args.invdyn_gradient_steps:
                        log["bc_loss_policy"] += invdyn.update(policy_td_obs, policy_td_act, policy_td_next_obs)['loss']
                        invdyn_lr_scheduler.step()

            # ----------- Logging ------------
            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["val_pred"] /= args.log_interval
                log["val_loss"] /= args.log_interval
                log["avg_loss_planner"] /= args.log_interval
                log["bc_loss_policy"] /= args.log_interval
                log["avg_loss_classifier"] /= args.log_interval
                print(log)
                if args.enable_wandb:
                    wandb.log(log, step=n_gradient_step + 1)
                pbar.update(1)
                log = {
                    "val_pred": 0,
                    "val_loss": 0,
                    "avg_loss_planner": 0, 
                    "bc_loss_policy": 0,
                    "avg_loss_classifier": 0
                }

            # ----------- Saving ------------
            if (n_gradient_step + 1) % args.save_interval == 0:
                planner.save(save_path + f"planner_ckpt_{n_gradient_step + 1}.pt")
                planner.save(save_path + f"planner_ckpt_latest.pt")
                if args.guidance_type=="MCSS":
                    torch.save({"critic": critic.state_dict(),}, save_path + f"critic_ckpt_{n_gradient_step + 1}.pt")
                    torch.save({"critic": critic.state_dict(),}, save_path + f"critic_ckpt_latest.pt")
                elif args.guidance_type=="cg":
                    planner.classifier.save(save_path + f"classifier_ckpt_{n_gradient_step + 1}.pt")
                    planner.classifier.save(save_path + f"classifier_ckpt_latest.pt")
                
                if args.pipeline_type == "separate":
                    if args.use_diffusion_invdyn:
                        policy.save(save_path + f"policy_ckpt_{n_gradient_step + 1}.pt")
                        policy.save(save_path + f"policy_ckpt_latest.pt")
                    else:
                        invdyn.save(save_path + f"invdyn_ckpt_{n_gradient_step + 1}.pt")
                        invdyn.save(save_path + f"invdyn_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step >= args.planner_diffusion_gradient_steps and n_gradient_step >= args.policy_diffusion_gradient_steps:
                break

    # ---------------------- Inference ----------------------
    elif args.mode == "inference":
        
        if args.guidance_type=="MCSS":
            # load planner
            planner.load(save_path + f"planner_ckpt_{args.planner_ckpt}.pt")
            planner.eval()
            # load critic
            critic_ckpt = torch.load(save_path + f"critic_ckpt_{args.critic_ckpt}.pt")
            critic.load_state_dict(critic_ckpt["critic"])
            critic.eval()
            # load policy
            if args.pipeline_type == "separate":
                if args.use_diffusion_invdyn:
                    policy.load(save_path + f"policy_ckpt_{args.policy_ckpt}.pt")
                    policy.eval()
                else:
                    invdyn.load(save_path + f"invdyn_ckpt_{args.invdyn_ckpt}.pt")
                    invdyn.eval()
        
        elif args.guidance_type=="cfg":
            # load planner
            planner.load(save_path + f"planner_ckpt_{args.planner_ckpt}.pt")
            planner.eval()
            # load policy
            if args.pipeline_type == "separate":
                if args.use_diffusion_invdyn:
                    policy.load(save_path + f"policy_ckpt_{args.policy_ckpt}.pt")
                    policy.eval()
                else:
                    invdyn.load(save_path + f"invdyn_ckpt_{args.invdyn_ckpt}.pt")
                    invdyn.eval()
            
        elif args.guidance_type=="cg":
            # load planner
            planner.load(save_path + f"planner_ckpt_{args.planner_ckpt}.pt")
            # load classifier
            planner.classifier.load(save_path + f"classifier_ckpt_{args.planner_ckpt}.pt")
            planner.eval()
            # load policy
            if args.pipeline_type == "separate":
                if args.use_diffusion_invdyn:
                    policy.load(save_path + f"policy_ckpt_{args.policy_ckpt}.pt")
                    policy.eval()
                else:
                    invdyn.load(save_path + f"invdyn_ckpt_{args.invdyn_ckpt}.pt")
                    invdyn.eval()

        env_eval = gym.vector.make(args.task.env_name, args.num_envs)
        normalizer = planner_dataset.get_normalizer()
        episode_rewards = []
        
        for i in range(args.num_episodes):
            obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0
            finished = np.zeros(args.num_envs, dtype=bool)
            while not np.all(cum_done) and t < args.task.max_path_length + 1:
                
                # 1) generate plan
                if args.guidance_type == "MCSS":
                    planner_prior = torch.zeros((args.num_envs * args.planner_num_candidates, args.task.planner_horizon, planner_dim), device=args.device)
                    
                    obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
                    obs_repeat = obs.unsqueeze(1).repeat(1, args.planner_num_candidates, 1).view(-1, obs_dim)

                    # sample trajectories
                    planner_prior[:, 0, :obs_dim] = obs_repeat
                    traj, log = planner.sample(
                        planner_prior, solver=args.planner_solver,
                        n_samples=args.num_envs * args.planner_num_candidates, sample_steps=args.planner_sampling_steps, use_ema=args.planner_use_ema,
                        condition_cfg=None, w_cfg=1.0, temperature=args.task.planner_temperature)
                    
                    # resample
                    with torch.no_grad():
                        value = critic(traj)
                        value = value.view(args.num_envs, args.planner_num_candidates)
                        idx = torch.argmax(value, -1)
                        traj = traj.reshape(args.num_envs, args.planner_num_candidates, args.task.planner_horizon, planner_dim)
                        traj = traj[torch.arange(args.num_envs), idx]
                
                elif args.guidance_type == "cfg":
                    planner_prior = torch.zeros((args.num_envs, args.task.planner_horizon, planner_dim), device=args.device)
                    condition = torch.ones((args.num_envs, 1), device=args.device) * args.task.planner_target_return
                    
                    obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)

                    # sample trajectories
                    planner_prior[:, 0, :obs_dim] = obs
                    traj, log = planner.sample(
                        planner_prior, solver=args.planner_solver,
                        n_samples=args.num_envs, sample_steps=args.planner_sampling_steps, use_ema=args.planner_use_ema,
                        condition_cfg=condition, w_cfg=args.task.planner_w_cfg, temperature=args.task.planner_temperature)
                
                elif args.guidance_type == "cg":
                    planner_prior = torch.zeros((args.num_envs * args.planner_num_candidates, args.task.planner_horizon, planner_dim), device=args.device)
                    
                    obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
                    obs_repeat = obs.unsqueeze(1).repeat(1, args.planner_num_candidates, 1).view(-1, obs_dim)
                    
                    planner_prior[:, 0, :obs_dim] = obs_repeat
                    traj, log = planner.sample(
                        planner_prior, solver=args.planner_solver,
                        n_samples=args.num_envs * args.planner_num_candidates, sample_steps=args.planner_sampling_steps, use_ema=args.planner_use_ema,
                        w_cg=args.task.planner_w_cfg, temperature=args.task.planner_temperature)
                    
                    # resample
                    with torch.no_grad():
                        logp = log["log_p"].view(args.num_envs, args.planner_num_candidates)
                        idx = torch.argmax(logp, -1)
                        traj = traj.reshape(args.num_envs, args.planner_num_candidates, args.task.planner_horizon, planner_dim)
                        traj = traj[torch.arange(args.num_envs), idx]

                # 2) generate action
                if args.pipeline_type == "separate":
                    if args.use_diffusion_invdyn:
                        policy_prior = torch.zeros((args.num_envs, act_dim), device=args.device)
                        with torch.no_grad():
                            next_obs_plan = traj[:, 1, :]
                            obs_policy = obs.clone()
                            next_obs_policy = next_obs_plan.clone()
                            
                            
                            if args.rebase_policy:
                                next_obs_policy[:, :2] -= obs_policy[:, :2]
                                obs_policy[:, :2] = 0
                            
                            act, log = policy.sample(
                                policy_prior,
                                solver=args.policy_solver,
                                n_samples=args.num_envs,
                                sample_steps=args.policy_sampling_steps,
                                condition_cfg=torch.cat([obs_policy, next_obs_policy], dim=-1), w_cfg=1.0,
                                use_ema=args.policy_use_ema, temperature=args.policy_temperature)
                            act = act.cpu().numpy()
                    else:
                        # inverse dynamic
                        with torch.no_grad():
                            act = invdyn.predict(obs, traj[:, 1, :]).cpu().numpy()
                else:
                    act = traj[:, 0, obs_dim:]
                    act = act.cpu().numpy()
                    
                # step
                obs, rew, done, info = env_eval.step(act)
                finished |= (rew == 1.0)

                t += 1
                cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                ep_reward += finished
                # print(f'[t={t}] xy: {np.around(obs[:, :2], 2)}')
                print(f'[t={t}] finished: {finished}')

            episode_rewards.append(ep_reward)

        episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
        episode_rewards = np.array(episode_rewards).reshape(-1) * 100
        mean = np.mean(episode_rewards)
        err = np.std(episode_rewards) / np.sqrt(len(episode_rewards))
        print(mean, err)

        if args.enable_wandb:
            wandb.log({'Mean Reward': mean, 'Error': err})
            wandb.finish()

    elif args.mode == "train_inv":
        invdyn = MlpInvDynamic(obs_dim, act_dim, 512, nn.Tanh(), {"lr": 2e-4}, device=args.device)
        invdyn_lr_scheduler = CosineAnnealingLR(invdyn.optim, args.invdyn_gradient_steps)
        invdyn.train()
        
        n_gradient_step = 0
        log = {"bc_loss_policy": 0,}
        pbar = tqdm(total=args.invdyn_gradient_steps/args.log_interval)
        for planner_batch, policy_batch in loop_two_dataloaders(planner_dataloader, policy_dataloader):
            
            policy_horizon_obs = policy_batch["obs"]["state"].to(args.device)
            policy_horizon_action = policy_batch["act"].to(args.device)
            policy_td_obs, policy_td_next_obs, policy_td_act = policy_horizon_obs[:,0,:], policy_horizon_obs[:,1,:], policy_horizon_action[:,0,:]
        
            if n_gradient_step <= args.invdyn_gradient_steps:
                log["bc_loss_policy"] += invdyn.update(policy_td_obs, policy_td_act, policy_td_next_obs)['loss']
                invdyn_lr_scheduler.step()
                
            # ----------- Logging ------------
            if (n_gradient_step + 1) % args.log_interval == 0:
                log["bc_loss_policy"] /= args.log_interval
                print(log)
                if args.enable_wandb:
                    wandb.log(log, step=n_gradient_step + 1)
                pbar.update(1)
                log = {"bc_loss_policy": 0,}
                
            # ----------- Saving ------------
            if (n_gradient_step + 1) % args.save_interval == 0:
                invdyn.save(save_path + f"invdyn_ckpt_{n_gradient_step + 1}.pt")
                invdyn.save(save_path + f"invdyn_ckpt_latest.pt")
                
            n_gradient_step += 1
            if n_gradient_step >= args.invdyn_gradient_steps:
                break
        
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    pipeline()
