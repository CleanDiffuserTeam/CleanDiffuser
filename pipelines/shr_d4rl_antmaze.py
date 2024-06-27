
# simple hierarchical diffusers
# shd

import os

import d4rl
import gym
import hydra
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.classifier import CumRewClassifier
from cleandiffuser.dataset.d4rl_antmaze_dataset import D4RLAntmazeDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_classifier import HalfJannerUNet1d
from cleandiffuser.nn_diffusion import JannerUNet1d
from cleandiffuser.utils import report_parameters
from utils import set_seed

@hydra.main(config_path="../configs/shd/antmaze", config_name="antmaze", version_base=None)
def pipeline(args):

    set_seed(args.seed)

    save_path = f'results/{args.pipeline_name}/{args.task.env_name}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # ---------------------- Create Dataset ----------------------
    env = gym.make(args.task.env_name)
    # hyper-param. seg_length = xxx

    # HL -- downsampled
    hl_dataset = D4RLAntmazeDataset( # change the dataset
        env.get_dataset(), horizon=args.task.horizon, discount=args.discount,
        noreaching_penalty=args.noreaching_penalty,)
    hl_dataloader = DataLoader(
        hl_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    hl_obs_dim, hl_act_dim = hl_dataset.o_dim, hl_dataset.a_dim

    # LL -- short horizon
    ll_dataset = D4RLAntmazeDataset(
        env.get_dataset(), horizon=args.task.horizon, discount=args.discount,
        noreaching_penalty=args.noreaching_penalty,)
    ll_dataloader = DataLoader(
        ll_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    ll_obs_dim, ll_act_dim = ll_dataset.o_dim, ll_dataset.a_dim

    # --------------- Network Architecture -----------------
    # HL 
    hl_nn_diffusion = JannerUNet1d(
        hl_obs_dim + hl_act_dim, model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.task.dim_mult,
        timestep_emb_type="positional", attention=False, kernel_size=5)
    
    hl_nn_classifier = HalfJannerUNet1d(
        args.task.horizon, hl_obs_dim + hl_act_dim, out_dim=1,
        model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.task.dim_mult,
        timestep_emb_type="positional", kernel_size=3)
    
    # LL
    ll_nn_diffusion = JannerUNet1d(
        ll_obs_dim + ll_act_dim, model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.task.dim_mult,
        timestep_emb_type="positional", attention=False, kernel_size=5)

    ll_nn_classifier = HalfJannerUNet1d(
        args.task.horizon, ll_obs_dim + ll_act_dim, out_dim=1,
        model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.task.dim_mult,
        timestep_emb_type="positional", kernel_size=3)
    
    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(hl_nn_diffusion)
    report_parameters(ll_nn_diffusion)

    print(f"======================= Parameter Report of Classifier =======================")
    report_parameters(hl_nn_classifier)
    report_parameters(ll_nn_classifier)

    print(f"==============================================================================")

    # --------------- Classifier Guidance --------------------
    hl_classifier = CumRewClassifier(hl_nn_classifier, device=args.device)
    ll_classifier = CumRewClassifier(ll_nn_classifier, device=args.device)

    # ----------------- Masking -------------------
    hl_fix_mask = torch.zeros((args.task.horizon, hl_obs_dim + hl_act_dim))
    hl_fix_mask[0, :hl_obs_dim] = 1.
    hl_loss_weight = torch.ones((args.task.horizon, hl_obs_dim + hl_act_dim))
    hl_loss_weight[0, hl_obs_dim:] = args.action_loss_weight

    ll_fix_mask = torch.zeros((args.task.horizon, ll_obs_dim + ll_act_dim))
    ll_fix_mask[0, :ll_obs_dim] = 1.
    ll_loss_weight = torch.ones((args.task.horizon, ll_obs_dim + ll_act_dim))
    ll_loss_weight[0, ll_obs_dim:] = args.action_loss_weight
    
    # need design

    # --------------- Diffusion Model --------------------
    hl_agent = DiscreteDiffusionSDE(
        hl_nn_diffusion, None,
        fix_mask=hl_fix_mask, loss_weight=hl_loss_weight, classifier=hl_classifier, ema_rate=args.ema_rate,
        device=args.device, diffusion_steps=args.diffusion_steps, predict_noise=args.predict_noise)
    
    ll_agent = DiscreteDiffusionSDE(
        ll_nn_diffusion, None,
        fix_mask=ll_fix_mask, loss_weight=ll_loss_weight, classifier=ll_classifier, ema_rate=args.ema_rate,
        device=args.device, diffusion_steps=args.diffusion_steps, predict_noise=args.predict_noise)
    
    # ---------------------- Training ----------------------
    if args.mode == "train":

        hl_diffusion_lr_scheduler = CosineAnnealingLR(hl_agent.optimizer, args.diffusion_gradient_steps)
        ll_diffusion_lr_scheduler = CosineAnnealingLR(ll_agent.optimizer, args.diffusion_gradient_steps)

        hl_classifier_lr_scheduler = CosineAnnealingLR(hl_agent.classifier.optim, args.classifier_gradient_steps)
        ll_classifier_lr_scheduler = CosineAnnealingLR(ll_agent.classifier.optim, args.classifier_gradient_steps)

        hl_agent.train()
        ll_agent.train()

        hl_n_gradient_step = 0
        ll_n_gradient_step = 0

        hl_log = {"hl_avg_loss_diffusion": 0., "hl_avg_loss_classifier": 0.}
        ll_log = {"ll_avg_loss_diffusion": 0., "ll_avg_loss_classifier": 0.}

        for hl_batch, ll_batch in zip(loop_dataloader(hl_dataloader), loop_dataloader(ll_dataloader)):

            hl_obs = hl_batch["obs"]["state"].to(args.device)
            hl_act = hl_batch["act"].to(args.device)
            hl_val = hl_batch["val"].to(args.device)
            hl_x = torch.cat([hl_obs, hl_act], -1)

            ll_obs = ll_batch["obs"]["state"].to(args.device)
            ll_act = ll_batch["act"].to(args.device)
            ll_val = ll_batch["val"].to(args.device)
            ll_x = torch.cat([ll_obs, ll_act], -1)

            # ----------- Gradient Step ------------
            hl_log["hl_avg_loss_diffusion"] += hl_agent.update(hl_x)['loss']
            ll_log["ll_avg_loss_diffusion"] += ll_agent.update(ll_x)['loss']

            hl_diffusion_lr_scheduler.step()
            ll_diffusion_lr_scheduler.step()

            if hl_n_gradient_step <= args.classifier_gradient_steps:
                hl_log["hl_avg_loss_classifier"] += hl_agent.update_classifier(hl_x, hl_val)['loss']
                hl_classifier_lr_scheduler.step()

            if ll_n_gradient_step <= args.classifier_gradient_steps:
                ll_log["ll_avg_loss_classifier"] += ll_agent.update_classifier(ll_x, ll_val)['loss']
                ll_classifier_lr_scheduler.step()

            # ----------- Logging ------------
            if (hl_n_gradient_step + 1) % args.log_interval == 0:
                hl_log["gradient_steps"] = hl_n_gradient_step + 1
                hl_log["hl_avg_loss_diffusion"] /= args.log_interval
                print(hl_log)
                hl_log = {"hl_avg_loss_diffusion": 0.}

            if (ll_n_gradient_step + 1) % args.log_interval == 0:
                ll_log["gradient_steps"] = ll_n_gradient_step + 1
                ll_log["ll_avg_loss_diffusion"] /= args.log_interval
                print(ll_log)
                ll_log = {"ll_avg_loss_diffusion": 0.}

            # ----------- Saving ------------
            if (hl_n_gradient_step + 1) % args.save_interval == 0:
                hl_agent.save(save_path + f"diffusion_ckpt_{hl_n_gradient_step + 1}.pt")
                hl_agent.classifier.save(save_path + f"classifier_ckpt_{hl_n_gradient_step + 1}.pt")
                hl_agent.save(save_path + f"diffusion_ckpt_latest.pt")
                hl_agent.classifier.save(save_path + f"classifier_ckpt_latest.pt")

            if (ll_n_gradient_step + 1) % args.save_interval == 0:
                ll_agent.save(save_path + f"diffusion_ckpt_{ll_n_gradient_step + 1}.pt")
                ll_agent.classifier.save(save_path + f"classifier_ckpt_{ll_n_gradient_step + 1}.pt")
                ll_agent.save(save_path + f"diffusion_ckpt_latest.pt")
                ll_agent.classifier.save(save_path + f"classifier_ckpt_latest.pt")

            hl_n_gradient_step += 1
            ll_n_gradient_step += 1

            if hl_n_gradient_step >= args.diffusion_gradient_steps:
                break

            if ll_n_gradient_step >= args.classifier_gradient_steps:
                break


if __name__ == "__main__":
    pipeline()
