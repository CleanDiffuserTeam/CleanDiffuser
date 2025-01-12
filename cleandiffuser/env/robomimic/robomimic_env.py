from pathlib import Path

import gym
import numpy as np
from gym import spaces

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils


class RobomimicEnv(gym.Env):
    def __init__(
        self,
        dataset_path: str = None,
        abs_action: bool = True,
        enable_render: bool = True,
        use_image_obs: bool = True,
    ):
        super().__init__()
        self._third_person_camera_name = "agentview"
        if "transport" in str(dataset_path):
            self._third_person_camera_name = "shouldercamera0"
        elif "tool_hang" in str(dataset_path):
            self._third_person_camera_name = "sideview"

        self.max_episode_steps = 1100
        if (
            "lift" in str(dataset_path)
            or "can" in str(dataset_path)
            or "square" in str(dataset_path)
        ):
            self.max_episode_steps = 500
        self.episode_steps = 0

        if dataset_path is None:
            dataset_path = (
                Path(__file__).parents[3] / "dev/robomimic/datasets/can/ph/low_dim_abs.hdf5"
            )

        is_dual_arm = "transport" in str(dataset_path)
        self.is_dual_arm = is_dual_arm
        self.use_image_obs = use_image_obs

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        if abs_action:
            env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False

        obs_dict = {
            "low_dim": ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
        }
        if is_dual_arm:
            obs_dict["low_dim"].extend(["robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos"])
        if use_image_obs:
            obs_dict["rgb"] = [self._third_person_camera_name + "_image", "robot0_eye_in_hand_image"]
            if is_dual_arm:
                obs_dict["rgb"].extend(["robot1_eye_in_hand_image"])

        ObsUtils.initialize_obs_modality_mapping_from_dict(obs_dict)
        self.env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=enable_render,
            use_image_obs=use_image_obs,
        )

        # setup spaces
        low = np.full(self.env.action_dimension, fill_value=-1)
        high = np.full(self.env.action_dimension, fill_value=1)
        self.action_space = spaces.Box(low=low, high=high, shape=low.shape, dtype=np.float32)

        observation_space = spaces.Dict()
        obs_example = self.get_observation()
        low = np.full_like(obs_example["lowdim"], fill_value=-1)
        high = np.full_like(obs_example["lowdim"], fill_value=1)
        observation_space["lowdim"] = spaces.Box(
            low=low, high=high, shape=low.shape, dtype=np.float32
        )
        observation_space["image"] = (
            spaces.Box(low=0, high=255, shape=obs_example["image"].shape, dtype=np.uint8)
            if use_image_obs
            else spaces.Box(low=0, high=255, shape=(2, 3, 84, 84), dtype=np.uint8)
        )
        self.observation_space = observation_space

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed

    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env.get_observation()

        lowdim_list = [
            raw_obs["object"],
            raw_obs["robot0_eef_pos"],
            raw_obs["robot0_eef_quat"],
            raw_obs["robot0_gripper_qpos"],
        ]
        if self.is_dual_arm:
            lowdim_list.extend(
                [
                    raw_obs["robot1_eef_pos"],
                    raw_obs["robot1_eef_quat"],
                    raw_obs["robot1_gripper_qpos"],
                ]
            )

        lowdim = np.concatenate(lowdim_list, axis=0)

        if self.use_image_obs:
            image_list = [
                raw_obs[self._third_person_camera_name + "_image"],
                raw_obs["robot0_eye_in_hand_image"],
            ]
            if self.is_dual_arm:
                image_list.extend([raw_obs["robot1_eye_in_hand_image"]])
            image = np.stack(image_list, axis=0)

            # transform [0,1] to [0,255] since the robomimic dataset uses uint8 image
            image = (image * 255 + 1e-8).astype(np.uint8)
        else:
            image = None

        return {
            "lowdim": lowdim,
            "image": image,
        }

    def reset(self):
        raw_obs = self.env.reset()
        self.episode_steps = 0
        return self.get_observation(raw_obs)

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        self.episode_steps += 1
        done = done or reward or self.episode_steps >= self.max_episode_steps
        return self.get_observation(raw_obs), reward, done, info

    def render(
        self,
        mode: str = "rgb_array",
        height: int = 256,
        width: int = 256,
        render_camera_name: str = None,
    ):
        if render_camera_name is None:
            render_camera_name = self._third_person_camera_name
        return self.env.render(
            mode=mode, height=height, width=width, camera_name=render_camera_name
        )
