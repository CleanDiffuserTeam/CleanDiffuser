# Update Log 2025-03-10

## 1. Lightning Style Pipeline Files

We have reimplemented the pipeline files based on the following principles:  

- Adopted the [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) style to enable users to leverage advanced deep learning techniques provided by Lightning, such as mixed precision and multi-GPU/multi-node training.  
- Removed the `configs` and replaced them with `argparser` for passing arguments, allowing users to understand all algorithm details within a **single file**.  
- Eliminated numerous unimportant arguments to maintain readability and avoid unnecessary complexity.  
- Avoided excessive hyperparameter tuning to reflect the algorithm's performance in general scenarios.  

The new version of the pipeline files is significantly improved in terms of simplicity and readability compared to the previous version, especially for imitation learning algorithms such as DBC and Diffusion Policy. Due to limited capacity, not all algorithms have been restructured yet. The algorithms for those pending updates are marked in files.  

For the restructured algorithms, we provide pre-trained checkpoints, which you can download **[here](https://1drv.ms/f/c/ba682474b24f6989/EkquzdUmBPhGmSXtRpskv1MB25ljjldiJB3z4UDD-FbJKQ?e=cXgUbI)** and extract to `"/path/to/CleanDiffuser/results/"`.

## 2. Support for Popular Embodied AI Benchmarks

We have added support for several popular embodied AI benchmarks, including Robomimic, PushT, Kitchen, and Libero. We provide Gym-like environment wrappers for these benchmarks, along with their datasets. The Gym-like wrappers are not only simplify the usage of these benchmarks but also provide some useful tools, **e.g. Depth/PointCloud observations in Libero**.

Please see the [environment support documentation](../cleandiffuser/env/README.md) for more details.

## 3. Pretrained Inverse Dynamics Models & IQL Models

We have notices that training inverse dynamics models and IQL models is a repetitive and tedious process. To help users bypass this process and accelerate algorithm validation, we provide a set of pre-trained inverse dynamics models and IQL models.

Please see the [inverse dynamics models documentation](../cleandiffuser/invdynamic/README.md) and [IQL models documentation](../cleandiffuser/utils/valuefuncs/README.md) for more details.