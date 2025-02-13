#!/bin/bash
# This script is used to run dd_d4rl, you should run diffusion_training first, then run invdyn_training, finally run inference.
# mode: diffusion_training -> invdyn_training -> inference
pipeline_name="dd_d4rl"

# test task list: 
# halfcheetah-medium-v2 halfcheetah-medium-replay-v2 halfcheetah-medium-expert-v2
# hopper-medium-v2 hopper-medium-replay-v2 hopper-medium-expert-v2
# walker2d-medium-v2 walker2d-medium-replay-v2 walker2d-medium-expert-v2
# kitchen-partial-v0 kitchen-mixed-v0
# antmaze-medium-play-v2 antmaze-medium-diverse-v2 antmaze-large-play-v2 antmaze-large-diverse-v2

# -------------------- halfcheetah-medium-v2 --------------------
# training
env_name="halfcheetah-medium-v2"
training_steps=1000000
seed=0
mode="diffusion_training"
# CUDA_VISIBLE_DEVICES=0 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode diffusion_training_steps=$training_steps seed=$seed > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &
# invdyn_training
invdyn_training_steps=400000
mode="invdyn_training"
#CUDA_VISIBLE_DEVICES=0 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode invdyn_training_steps=$invdyn_training_steps > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &
# inference
mode="inference"
CUDA_VISIBLE_DEVICES=0 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode diffusion_ckpt=$training_steps invdyn_ckpt=$invdyn_training_steps solver=ddpm sampling_steps=20 num_envs=50 num_episodes=3 > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &

# -------------------- halfcheetah-medium-replay-v2 --------------------
# training
env_name="halfcheetah-medium-replay-v2"
training_steps=1000000
seed=0
mode="diffusion_training"
# CUDA_VISIBLE_DEVICES=1 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode diffusion_training_steps=$training_steps seed=$seed > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &
# invdyn_training
invdyn_training_steps=400000
mode="invdyn_training"
# CUDA_VISIBLE_DEVICES=1 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode invdyn_training_steps=$invdyn_training_steps > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &
# inference
mode="inference"
CUDA_VISIBLE_DEVICES=1 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode diffusion_ckpt=$training_steps invdyn_ckpt=$invdyn_training_steps solver=ddpm sampling_steps=20 num_envs=50 num_episodes=3 > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &

# -------------------- halfcheetah-medium-expert-v2 --------------------
# training
env_name="halfcheetah-medium-expert-v2"
training_steps=1000000
seed=0
mode="diffusion_training"
# CUDA_VISIBLE_DEVICES=2 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode diffusion_training_steps=$training_steps seed=$seed > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &
# invdyn_training
invdyn_training_steps=400000
mode="invdyn_training"
# CUDA_VISIBLE_DEVICES=2 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode invdyn_training_steps=$invdyn_training_steps > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &
# inference
mode="inference"
CUDA_VISIBLE_DEVICES=2 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode diffusion_ckpt=$training_steps invdyn_ckpt=$invdyn_training_steps solver=ddpm sampling_steps=20 num_envs=50 num_episodes=3 > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &

# -------------------- hopper-medium-v2 --------------------
# training
env_name="hopper-medium-v2"
training_steps=1000000
seed=0
mode="diffusion_training"
#CUDA_VISIBLE_DEVICES=3 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode diffusion_training_steps=$training_steps seed=$seed > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &
# invdyn_training
invdyn_training_steps=400000
mode="invdyn_training"
# CUDA_VISIBLE_DEVICES=3 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode invdyn_training_steps=$invdyn_training_steps > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &
# inference
mode="inference"
CUDA_VISIBLE_DEVICES=3 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode diffusion_ckpt=$training_steps invdyn_ckpt=$invdyn_training_steps solver=ddpm sampling_steps=20 num_envs=50 num_episodes=3 > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &

# -------------------- hopper-medium-replay-v2 -------------------- 
env_name="hopper-medium-replay-v2"
training_steps=1000000
seed=0
mode="diffusion_training"
# CUDA_VISIBLE_DEVICES=4 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode diffusion_training_steps=$training_steps seed=$seed > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &
# invdyn_training
invdyn_training_steps=400000
mode="invdyn_training"
# CUDA_VISIBLE_DEVICES=4 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode invdyn_training_steps=$invdyn_training_steps > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &
# inference
mode="inference"
CUDA_VISIBLE_DEVICES=4 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode diffusion_ckpt=$training_steps invdyn_ckpt=$invdyn_training_steps solver=ddpm sampling_steps=20 num_envs=50 num_episodes=3 > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &

# -------------------- hopper-medium-expert-v2 --------------------
# training
env_name="hopper-medium-expert-v2"
training_steps=1000000
seed=0
mode="diffusion_training"
# CUDA_VISIBLE_DEVICES=7 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode diffusion_training_steps=$training_steps seed=$seed > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &
# invdyn_training
invdyn_training_steps=400000
mode="invdyn_training"
# CUDA_VISIBLE_DEVICES=7 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode invdyn_training_steps=$invdyn_training_steps > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &
# inference
mode="inference"
CUDA_VISIBLE_DEVICES=7 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode diffusion_ckpt=$training_steps invdyn_ckpt=$invdyn_training_steps solver=ddpm sampling_steps=20 num_envs=50 num_episodes=3 > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &


# -------------------- walker2d-medium-v2 --------------------
# training
env_name="walker2d-medium-v2"
training_steps=1000000
seed=0
mode="diffusion_training"
# CUDA_VISIBLE_DEVICES=5 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode diffusion_training_steps=$training_steps seed=$seed > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &
# invdyn_training
invdyn_training_steps=400000
mode="invdyn_training"
# CUDA_VISIBLE_DEVICES=5 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode invdyn_training_steps=$invdyn_training_steps > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &
# inference
mode="inference"
CUDA_VISIBLE_DEVICES=5 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode diffusion_ckpt=$training_steps invdyn_ckpt=$invdyn_training_steps solver=ddpm sampling_steps=20 num_envs=50 num_episodes=3 > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &

# -------------------- walker2d-medium-expert-v2 --------------------
env_name="walker2d-medium-expert-v2"
training_steps=1000000
seed=0
mode="diffusion_training"
# CUDA_VISIBLE_DEVICES=6 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode diffusion_training_steps=$training_steps seed=$seed > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &
# invdyn_training
invdyn_training_steps=400000
mode="invdyn_training"
# CUDA_VISIBLE_DEVICES=6 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode invdyn_training_steps=$invdyn_training_steps > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &
# inference
mode="inference"
CUDA_VISIBLE_DEVICES=6 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode diffusion_ckpt=$training_steps invdyn_ckpt=$invdyn_training_steps solver=ddpm sampling_steps=20 num_envs=50 num_episodes=3 > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &

# -------------------- walker2d-medium-replay-v2 --------------------
env_name="walker2d-medium-replay-v2"
training_steps=1000000
seed=0
mode="diffusion_training"
# CUDA_VISIBLE_DEVICES=7 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode diffusion_training_steps=$training_steps seed=$seed > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &
# invdyn_training
invdyn_training_steps=400000
mode="invdyn_training"
# CUDA_VISIBLE_DEVICES=7 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode invdyn_training_steps=$invdyn_training_steps > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &
# inference
mode="inference"
CUDA_VISIBLE_DEVICES=7 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode diffusion_ckpt=$training_steps invdyn_ckpt=$invdyn_training_steps solver=ddpm sampling_steps=20 num_envs=50 num_episodes=3 > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &

# -------------------- kitchen-partial-v0 --------------------
env_name="kitchen-partial-v0"
training_steps=1000000
seed=0
mode="diffusion_training"
# CUDA_VISIBLE_DEVICES=0 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode diffusion_training_steps=$training_steps seed=$seed > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &
# invdyn_training
invdyn_training_steps=400000
mode="invdyn_training"
# CUDA_VISIBLE_DEVICES=0 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode invdyn_training_steps=$invdyn_training_steps > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &
# inference
mode="inference"
CUDA_VISIBLE_DEVICES=0 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode diffusion_ckpt=$training_steps invdyn_ckpt=$invdyn_training_steps solver=ddpm sampling_steps=20 num_envs=50 num_episodes=3 > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &

# -------------------- kitchen-mixed-v0 --------------------
env_name="kitchen-mixed-v0"
training_steps=1000000
seed=0
mode="diffusion_training"
# CUDA_VISIBLE_DEVICES=1 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode diffusion_training_steps=$training_steps seed=$seed > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &
# invdyn_training
invdyn_training_steps=400000
mode="invdyn_training"
# CUDA_VISIBLE_DEVICES=1 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode invdyn_training_steps=$invdyn_training_steps > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &
# inference
mode="inference"
CUDA_VISIBLE_DEVICES=1 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode diffusion_ckpt=$training_steps invdyn_ckpt=$invdyn_training_steps solver=ddpm sampling_steps=20 num_envs=50 num_episodes=3 > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &

# -------------------- antmaze-medium-diverse-v2 --------------------
env_name="antmaze-medium-diverse-v2"
training_steps=1000000
seed=0
mode="diffusion_training"
# CUDA_VISIBLE_DEVICES=2 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode diffusion_training_steps=$training_steps seed=$seed > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &
# invdyn_training
invdyn_training_steps=400000
mode="invdyn_training"
# CUDA_VISIBLE_DEVICES=2 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode invdyn_training_steps=$invdyn_training_steps > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &
# inference
mode="inference"
CUDA_VISIBLE_DEVICES=2 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode diffusion_ckpt=$training_steps invdyn_ckpt=$invdyn_training_steps solver=ddpm sampling_steps=20 num_envs=50 num_episodes=3 > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &

# -------------------- antmaze-medium-play-v2 --------------------
env_name="antmaze-medium-play-v2"
training_steps=1000000
seed=0
mode="diffusion_training"
# CUDA_VISIBLE_DEVICES=3 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode diffusion_training_steps=$training_steps seed=$seed > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &
# invdyn_training
invdyn_training_steps=400000
mode="invdyn_training"
# CUDA_VISIBLE_DEVICES=3 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode invdyn_training_steps=$invdyn_training_steps > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &
# inference
mode="inference"
CUDA_VISIBLE_DEVICES=3 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode diffusion_ckpt=$training_steps invdyn_ckpt=$invdyn_training_steps solver=ddpm sampling_steps=20 num_envs=50 num_episodes=3 > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &

# -------------------- antmaze-large-diverse-v2 --------------------
env_name="antmaze-large-diverse-v2"
training_steps=1000000
seed=0
mode="diffusion_training"
# CUDA_VISIBLE_DEVICES=4 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode diffusion_training_steps=$training_steps seed=$seed > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &
# invdyn_training
invdyn_training_steps=400000
mode="invdyn_training"
# CUDA_VISIBLE_DEVICES=4 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode invdyn_training_steps=$invdyn_training_steps > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &
# inference
mode="inference"
CUDA_VISIBLE_DEVICES=4 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode diffusion_ckpt=$training_steps invdyn_ckpt=$invdyn_training_steps solver=ddpm sampling_steps=20 num_envs=50 num_episodes=3 > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &

# -------------------- antmaze-large-play-v2 --------------------
env_name="antmaze-large-play-v2"
training_steps=1000000
seed=0
mode="diffusion_training"
# CUDA_VISIBLE_DEVICES=5 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode diffusion_training_steps=$training_steps seed=$seed > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &
# invdyn_training
invdyn_training_steps=400000
mode="invdyn_training"
# CUDA_VISIBLE_DEVICES=5 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode invdyn_training_steps=$invdyn_training_steps > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &
# inference
mode="inference"
CUDA_VISIBLE_DEVICES=5 nohup python -m pipelines.decision_diffuser.dd_d4rl task.env_name=$env_name mode=$mode diffusion_ckpt=$training_steps invdyn_ckpt=$invdyn_training_steps solver=ddpm sampling_steps=20 num_envs=50 num_episodes=3 > logs/${mode}_${pipeline_name}_${env_name}.log 2>&1 &


