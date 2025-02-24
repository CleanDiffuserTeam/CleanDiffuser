# CleanDiffuser LIBERO Benchmark Support

> Note: The CleanDiffuser LIBERO benchmark support is currently in beta. Please report any issues to me [here](zibindong@outlook.com).

CleanDiffuser is fully decoupled from any benchmark or dataset, meaning you are not required to use the CleanDiffuser LIBERO Benchmark Support for conducting LIBERO experiments. It is simply a streamlined wrapper for using LIBERO, designed to facilitate quick algorithm development and validation. Compared to directly using LIBERO, the support we provide includes the following features:  

- A script for regenerating the dataset, which expands the original LIBERO dataset into 3D data (including metric depth and point clouds). The generated dataset is stored in the Zarr format, enabling more efficient data reading. CleanDiffuser also provides a PyTorch Dataset implementation for direct use of this dataset.  
- A gym environment wrapper that allows users to create a `gym.Env` using `gym.make("libero-goal-v0")`. By passing parameters, users can specify whether the observation includes extra 3D data, such as metric depth or point clouds.  
- A diffusion policy pipeline for the LIBERO Benchmark, provided as a reference for users working with LIBERO.  

Of course, simpler wrappers come with lower customizability. For users who require deeply customized environments, we still recommend using the original LIBERO.

## 1. 3D Dataset Support

### 1.1. Regenerating the dataset

LIBERO datasets only provide RGB image observations. To support the validation of 3D embodied policy algorithms, we can replay the datasets in the environment to obtain metric depth and point clouds. Users can achieve this by:
- step 1: Downloading the LIBERO dataset from the [LIBERO dataset website](https://libero-project.github.io/datasets), unzipping it to `/path_to_LIBERO/libero/datasets/`.
- step 2: Running `regenerate_dataset.py`. In this script, users have several options to modify the dataset generation process, such as `DEVICE`, `PATH_TO_T5` (default to use T5-base to embed the text), `IMAGE_SIZE` (image size for the generated dataset), `BOUNDING_BOX` (point cloud cropping box), etc. The generated dataset will be stored in `/path_to_CleanDiffuser/dev/libero/` in the Zarr format.

Take `libero_goal` as an example, the generated dataset will have the following structure:
```
/
 ├── data
 │   ├── actions (63728, 7) float32
 │   ├── color (63728, 3, 224, 224) uint8
 │   ├── color_ego (63728, 3, 224, 224) uint8
 │   ├── depth (63728, 224, 224) uint16
 │   ├── depth_ego (63728, 224, 224) uint16
 │   ├── eef_states (63728, 7) float32
 │   ├── gripper_states (63728, 2) float32
 │   ├── joint_states (63728, 7) float32
 │   ├── pointcloud (63728, 8192, 3) float32
 │   ├── pointcloud_ego (63728, 8192, 3) float32
 │   ├── rewards (63728,) float32
 │   └── states (63728, 79) float32
 └── meta
     ├── episode_ends (500,) uint32
     ├── language_embeddings (500, 32, 768) float32
     ├── language_masks (500, 32) float32
     └── task_id (500,) uint16
```

### 1.2 Wrapper for PyTorch Dataset
You can wrapper this Zarr dataset by your own PyTorch Dataset implementation or use the provided `LiberoDataset` in CleanDiffuser.

```python
from cleandiffuser.dataset.libero_dataset import LiberoDataset

dataset = LiberoDataset(
    data_path="/path_to_CleanDiffuser/dev/libero/libero_goal.zarr",
    observation_meta=["color", "depth", "pointcloud"],  # choose from ["color", "color_ego", "depth", "depth_ego", "pointcloud", "pointcloud_ego", "eef_states", "gripper_states", "joint_states", "states"], default to use all.
    To=2,  # observation horizon, 1 + history
    Ta=16  # action horizon, 1 + future
)

item = dataset[0]
# `item` contains:
# observation: (each from the observation_meta)
# action: (7,)
# language_embedding: (32, 768) t5-base embedding
# language_mask: (32,) padding mask
```
> Note: Like other CleanDiffuser-provided datasets, the `actions` are normalized to [-1, 1] in dataset. Don't forget to denormalize them before using by `dataset.normalizers["actions"].unnormalize(pred_action)`.

## 2. Gym Environment Wrapper

You can use the provided LIBERO gym environment by:
```python
>>> import gym
>>> from cleandiffuser.env import libero
>>> env = gym.make(
    "libero-goal-v0",  # from ["libero-goal-v0", "libero-object-v0", "libero-spatial-v0", "libero-10-v0", "libero-90-v0"],
    task_id=0,  # task id from 0 to 9
    image_size=224,  # image size (height, width)
    require_depth=True,  # require metric depth
    require_point_cloud=True,  # require point cloud
    num_points=8192,  # number of points in point cloud
    camera_names=["agentview", "robot0_eye_in_hand"],  # camera names
    seed=0  # random seed
)
>>> o = env.reset()
>>> for k, v in o.items():
        print(k, v.shape, v.dtype)
robot0_joint_pos (7,) float64
robot0_joint_pos_cos (7,) float64
robot0_joint_pos_sin (7,) float64
robot0_joint_vel (7,) float64
robot0_eef_pos (3,) float64
robot0_eef_quat (4,) float64
robot0_gripper_qpos (2,) float64
robot0_gripper_qvel (2,) float64
agentview_image (3, 224, 224) uint8
agentview_depth (224, 224) float32
robot0_eye_in_hand_image (3, 224, 224) uint8
...
```

LIBERO defines a set of `init_states` for benchmarking. You can call `env.num_init_states` to get the number of `init_states` and `env.reset(init_state_id=i)` to reset the environment with the i-th `init_state`. If not specified, the environment will randomly choose an `init_state`.

## 3. Diffusion Policy Pipeline

See `/path_to_CleanDiffuser/pipelines/diffusion_policy/dp_libero.py`. Make sure you have regenerated the dataset as described in Section 1.1 before running the pipeline.