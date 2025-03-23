# CleanDiffuser Gym Environment Support

> > Note: The CleanDiffuser Gym Environment Support is currently in beta. Please report any issues to me [here](zibindong@outlook.com).

To simplify the usage of commonly used benchmarks, CleanDiffuser has created several gym environments (Please use `gym>=0.13,<0.24`). The supported benchmarks are listed below.

## 1. [Robomimic](https://robomimic.github.io/)
### 1.1 Requirements
You need to install robomimic following the [installation instructions](https://robomimic.github.io/docs/introduction/installation.html), and download the datasets from [here](https://diffusion-policy.cs.columbia.edu/data/training/). The environment creation requires these datasets.

### 1.2 Usage
```python
import gym
from cleandiffuser.env import robomimic

env = gym.make(
    dataset_path="/path/to/dataset.hdf5",
    abs_action=True,  # Whether to position control
    enable_render=True,  # Whether to enable rendering
    use_image_obs=True,  # Whether to use image observations
)
```
see [this file](./robomimic/robomimic_env.py) for more details.

## 2. PushT
### 2.1 Requirements
You need to install the following packages:
```bash
pygame
pymunk
shapely<2.0.0
scikit-image<0.23.0
opencv-python
```
and download the datasets from [here](https://diffusion-policy.cs.columbia.edu/data/training/). The environment creation doesn't require these datasets, and they are just for your training.

### 2.2 Usage
```python
import gym
from cleandiffuser.env import pusht

env0 = gym.make("pusht-v0")
env1 = gym.make("pusht-keypoints-v0")
env2 = gym.make("pusht-image-v0")
```
see [this file](./pusht/pusht_env.py) for more details.

## 3. RelayKitchen
Unlike FrankaKitchen in D4RL, RelayKitchen is a imitation learning benchmark containing human demonstrations.

### 3.1 Requirements
You need to install the following packages:
```python
mujoco_py>=2.0,<=3.1.6
dm_control>=1.0.3,<=1.0.20
```
and download the datasets from [here](https://diffusion-policy.cs.columbia.edu/data/training/). The environment creation doesn't require these datasets, and they are just for your training.

### 3.2 Usage
```python
import gym
from cleandiffuser.env import kitchen

env = gym.make("kitchen-all-v0")
```
see [this file](./kitchen/base.py) for more details.

## 4. [Libero](https://libero-project.github.io/main.html)
### 4.1 Requirements
You need to install libero following the [installation instructions](https://lifelong-robot-learning.github.io/LIBERO/html/getting_started/installation.html). If you need to access 3d information, e.g. depth/pointcloud, you also need to install [pytorch3d](https://github.com/facebookresearch/pytorch3d) and [open3d](https://www.open3d.org/). For anyone who has difficulty installing pytorch3d, you may try this [guide](https://github.com/facebookresearch/pytorch3d/discussions/1752).

### 4.2 Usage
Please refer to [this file](./libero/README.md). This README file contains all instructions from dataset generation to environment creation.