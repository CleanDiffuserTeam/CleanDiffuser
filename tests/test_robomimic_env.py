import pytest
import sys
import os
import numpy as np

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

from cleandiffuser.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
from cleandiffuser.env.wrapper import VideoRecordingWrapper, MultiStepWrapper
from cleandiffuser.env.utils import VideoRecorder

import gym

@pytest.fixture
def create_env():
    dataset_path = 'dev/robomimic/datasets/lift/mh/low_dim.hdf5'
    dataset_path = os.path.expanduser(dataset_path)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    abs_action = False
    if abs_action:
        env_meta['env_kwargs']['controller_configs']['control_delta'] = False
    obs_keys = ['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']
    ObsUtils.initialize_obs_modality_mapping_from_dict(
        {'low_dim': obs_keys})
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        # only way to not show collision geometry
        # is to enable render_offscreen
        # which uses a lot of RAM.
        render_offscreen=False,
        use_image_obs=False,
    )
    return env


def test_create_env(create_env):
    env = create_env

    assert env is not None

    env = RobomimicLowdimWrapper(
        env=env,
        obs_keys=['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
        init_state=None,
        render_hw=(256, 256),
        render_camera_name='agentview'
    )

    video_recoder = VideoRecorder.create_h264(
        fps=10,
        codec='h264',
        input_pix_fmt='rgb24',
        crf=22,
        thread_type='FRAME',
        thread_count=1
    )
    env = VideoRecordingWrapper(env, video_recoder, file_path=None, steps_per_render=2)
    env = MultiStepWrapper(env, n_obs_steps=2, n_action_steps=8, max_episode_steps=300)

    env.seed(1000)
    obs = env.reset()
    action = env.action_space.sample()
    obs, reward, terminal, info = env.step(action)

    with np.printoptions(precision=4, suppress=True, threshold=5):
        print("Obs Shape: ", obs.shape)
        print("Obs: ", repr(obs))
        print("Action Shape: ", action.shape)
        print("Action: ", repr(action))

