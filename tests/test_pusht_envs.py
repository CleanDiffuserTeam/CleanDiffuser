import sys
import os
import gym
import pytest
import numpy as np

from cleandiffuser.env import pusht
from cleandiffuser.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from cleandiffuser.env.wrapper import VideoRecordingWrapper, MultiStepWrapper
from cleandiffuser.env.utils import VideoRecorder


@pytest.fixture
def setup_env(request):
    env_type = request.param
    if env_type == "pusht-keypoints-v0":
        kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
        env = gym.make(env_type, **kp_kwargs)
    else:
        env = gym.make(env_type)

    fps = 10
    video_recorder = VideoRecorder.create_h264(
        fps=fps,
        codec='h264',
        input_pix_fmt='rgb24',
        crf=22,
        thread_type='FRAME',
        thread_count=1
    )
    steps_per_render = max(10 // fps, 1)
    env = VideoRecordingWrapper(env, video_recorder, file_path=None, steps_per_render=steps_per_render)
    env = MultiStepWrapper(env, n_obs_steps=2, n_action_steps=8, max_episode_steps=300)

    env.seed(1000)
    return env


@pytest.mark.parametrize("setup_env", ["pusht-v0", "pusht-image-v0", "pusht-keypoints-v0"], indirect=True)
def test_env_reset(setup_env):
    env = setup_env

    obs = env.reset()
    assert obs is not None


@pytest.mark.parametrize("setup_env", ["pusht-v0", "pusht-image-v0", "pusht-keypoints-v0"], indirect=True)
def test_env_step(setup_env):
    env = setup_env
    obs = env.reset()

    action = env.action_space.sample()

    obs, reward, terminal, info = env.step(action)

    assert obs is not None
    assert isinstance(reward, float)
    assert isinstance(info, dict)

    with np.printoptions(precision=2, suppress=True, threshold=80):
        print("Env: ", env.spec.id)
        if isinstance(obs, np.ndarray):
            print("Obs: ", obs.shape)
        else:
            pass
        print("Action: ", action.shape)