import sys
import os
import gym
import pytest
import numpy as np

from cleandiffuser.env import kitchen
from cleandiffuser.env.wrapper import VideoRecordingWrapper, MultiStepWrapper
from cleandiffuser.env.utils import VideoRecorder
from cleandiffuser.env.kitchen.kitchen_lowdim_wrapper import KitchenLowdimWrapper


@pytest.fixture
def setup_env():
    env = gym.make("kitchen-all-v0")  # state
    render_hw = (240, 360)
    env = KitchenLowdimWrapper(env=env, init_qpos=None, init_qvel=None, render_hw=tuple(render_hw))

    fps = 12.5
    video_recorder = VideoRecorder.create_h264(
        fps=fps,
        codec='h264',
        input_pix_fmt='rgb24',
        crf=22,
        thread_type='FRAME',
        thread_count=1
    )
    steps_per_render = int(max(10 // fps, 1))
    env = VideoRecordingWrapper(env, video_recorder, file_path=None, steps_per_render=steps_per_render)
    env = MultiStepWrapper(env, n_obs_steps=2, n_action_steps=8, max_episode_steps=280)

    env.seed(1000)
    return env


def test_env_reset(setup_env):
    env = setup_env
    obs = env.reset()

    assert obs is not None


def test_env_step(setup_env):
    env = setup_env
    obs = env.reset()

    action = env.action_space.sample()

    obs, reward, terminal, info = env.step(action)

    assert obs is not None
    assert isinstance(reward, float)
    assert isinstance(info, dict)