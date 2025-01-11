import os
from pathlib import Path
from typing import Optional

import gym
import numpy as np
from gym import spaces
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_real_depth_map

import libero
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv


class LiberoEnv(gym.Env):
    TASK_SUITE_NAME = (
        ""  # can be "libero_goal", "libero_spatial", "libero_object", "libero_10", "libero_90"
    )

    def __init__(
        self,
        task_id: int = 0,  # from 0 to 9
        image_size: int = 224,
        require_depth: bool = True,
        require_point_cloud: bool = False,
        num_points: int = 8192,
        camera_names: list = ["agentview", "robot0_eye_in_hand"],
        seed: int = 0,
    ):
        super().__init__()
        self._require_point_cloud = require_point_cloud
        self._require_depth = require_depth
        assert not require_point_cloud or require_depth, "Require depth if require point cloud!"

        self._image_size = image_size
        self._camera_names = camera_names
        self._num_points = num_points

        root_path = Path(os.path.dirname(libero.libero.__file__))

        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.TASK_SUITE_NAME]()
        task = task_suite.get_task(task_id)

        task_description = task.language
        self.task_description = task_description
        task_bddl_file = root_path / "bddl_files" / task.problem_folder / task.bddl_file

        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": image_size,
            "camera_widths": image_size,
            "camera_depths": require_depth,
            "camera_names": camera_names,
        }
        self.env = OffScreenRenderEnv(**env_args)
        self.env.seed(seed)

        if require_point_cloud:
            import open3d as o3d

            def cammat2o3d(cam_mat, width, height):
                cx = cam_mat[0, 2]
                fx = cam_mat[0, 0]
                cy = cam_mat[1, 2]
                fy = cam_mat[1, 1]

                return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

            self._cam_intrinsics = dict()
            for camera_name in camera_names:
                cammat = get_camera_intrinsic_matrix(self.sim, camera_name, image_size, image_size)
                self._cam_intrinsics[camera_name] = cammat2o3d(cammat, image_size, image_size)

            self._pcd_bb = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=(-1.0, -1.0, 0.0), max_bound=(1.0, 1.0, 1.6)
            )

        self._init_states = task_suite.get_task_init_states(task_id)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Dict()
        example_obs = self.reset()
        for key, value in example_obs.items():
            if "image" in key:
                self.observation_space[key] = spaces.Box(
                    low=0, high=255, shape=value.shape, dtype=np.uint8
                )
            elif "depth" in key:
                self.observation_space[key] = spaces.Box(
                    low=0, high=2.0, shape=value.shape, dtype=value.dtype
                )
            elif "pointcloud" in key:
                self.observation_space[key] = spaces.Box(
                    low=-1.0, high=1.6, shape=value.shape, dtype=value.dtype
                )
            else:
                self.observation_space[key] = spaces.Box(
                    low=-np.inf, high=np.inf, shape=value.shape, dtype=value.dtype
                )

    @property
    def num_init_states(self):
        return len(self._init_states)

    def check_success(self):
        return self.env.env._check_success()

    @property
    def _visualizations(self):
        return self.env.env._visualizations

    @property
    def robots(self):
        return self.env.env.robots

    @property
    def sim(self):
        return self.env.env.sim

    def get_sim_state(self):
        return self.env.env.sim.get_state().flatten()

    def _post_process(self):
        return self.env.env._post_process()

    def _update_observables(self, force=False):
        self.env._update_observables(force=force)

    def set_state(self, mujoco_state):
        self.env.env.sim.set_state_from_flattened(mujoco_state)

    def reset_from_xml_string(self, xml_string):
        self.env.env.reset_from_xml_string(xml_string)

    def seed(self, seed):
        self.env.env.seed(seed)

    def set_init_state(self, init_state):
        return self.regenerate_obs_from_state(init_state)

    def regenerate_obs_from_state(self, mujoco_state):
        self.set_state(mujoco_state)
        self.env.env.sim.forward()
        self.check_success()
        self._post_process()
        self._update_observables(force=True)
        return self.get_observations()

    def get_observations(self):
        raw_obs = self.env.env._get_observations()

        for camera_name in self._camera_names:
            # flip to the correct orientation and transpose to (C, H, W)
            raw_obs[f"{camera_name}_image"] = raw_obs[f"{camera_name}_image"][::-1].transpose(
                2, 0, 1
            )

        if self._require_depth:
            for camera_name in self._camera_names:
                # convert to metric depth, flip to the correct orientation and reshape to (H, W)
                depth = raw_obs[f"{camera_name}_depth"]
                metric_depth = get_real_depth_map(self.sim, depth)
                metric_depth = np.clip(metric_depth, 0.0, 2.0)[::-1][:, :, 0].astype(np.float32)
                raw_obs[f"{camera_name}_depth"] = metric_depth

        if self._require_point_cloud:
            import open3d as o3d

            for camera_name in self._camera_names:
                voxel_size = 0.003 if "eye_in_hand" in camera_name else 0.005

                o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(
                    o3d.geometry.Image(np.copy(raw_obs[f"{camera_name}_depth"][:, :, None])),
                    self._cam_intrinsics[camera_name],
                )
                o3d_cloud = o3d_cloud.crop(self._pcd_bb)
                o3d_cloud = o3d_cloud.voxel_down_sample(voxel_size=voxel_size)
                o3d_cloud = o3d_cloud.farthest_point_down_sample(num_samples=self._num_points)

                pointcloud = np.asarray(o3d_cloud.points)
                if pointcloud.shape[0] < self._num_points:
                    pointcloud = np.concatenate(
                        [
                            pointcloud,
                            pointcloud[
                                np.random.choice(
                                    pointcloud.shape[0],
                                    self._num_points - pointcloud.shape[0],
                                    replace=False,
                                )
                            ],
                        ],
                        axis=0,
                    )

                raw_obs[f"{camera_name}_pointcloud"] = pointcloud

        return raw_obs

    def close(self):
        self.env.env.close()
        del self.env.env

    def reset(self, init_state_id: Optional[int] = None):
        if init_state_id is None:
            init_state_id = np.random.randint(self.num_init_states)
        self.env.reset()
        obs = self.set_init_state(self._init_states[init_state_id])
        return obs

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        obs = self.get_observations()
        info["task_description"] = self.task_description
        return obs, reward, done, info
