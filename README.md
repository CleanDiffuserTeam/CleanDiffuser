<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<p align="center">
    <br>
    <img src="assets/github_logo.jpg" width="300"/>
    <br>
<p>

# CleanDiffuser: An Easy-to-use Modularized Library for Diffusion Models in Decision Making

<p align="center">
·
<a href="https://arxiv.org/abs/2406.09509">ArXiv</a>
·
<a href="assets/CleanDiffuser.pdf">Paper</a>
·
<a href="https://cleandiffuserteam.github.io/CleanDiffuserDocs/">Documentation</a>
·
</p>

> **Note:** This is `lightning` branch. We've completely rewritten the codebase using *PyTorch Lightning*, restructuring all the classes as `LightningModule`. This lets us train models with `Trainer` and tap into advanced deep learning techniques like parallel training and mixed precision - all with just a few lines of code.
We've also optimized the codebase structure, squashed some bugs, and made the code more readable and user-friendly. **We strongly recommend using this branch for a better experience.** However, this branch is still a work in progress. Many algorithms in `pipelines` haven't been migrated yet, and there might be some bugs lurking in corner cases. If you spot any issues, please open an issue or submit a pull request. We're working hard to polish up this branch and merge it into main as soon as possible!

**CleanDiffuser** is an easy-to-use modularized Diffusion Model library designed for decision-making, which comprehensively integrates different types of diffusion algorithmic branches. CleanDiffuser offers a variety of advanced *diffusion models*, *network structures*, diverse *conditions*, and *algorithm pipelines* in a simple and user-friendly manner. Inheriting the design philosophy of [CleanRL](https://github.com/vwxyzjn/cleanrl) and [Diffusers](https://github.com/huggingface/diffusers), CleanDiffuser emphasizes **usability, simplicity, and customizability**. We hope that CleanDiffuser will serve as a foundational tool library, providing long-term support for Diffusion Model research in the decision-making community, facilitating the application of research for scientists and practitioners alike. The highlight features of CleanDiffuser are:

- 🚀 Amazing features specially designed for decision-making tasks
- 🍧 Support for multiple advanced diffusion models and network architectures
- 🧩 Build decoupled modules into integrated pipelines easily like building blocks
- 📈 Wandb logging and Hydra configuration
- 🌏 Unified environmental interface and efficient dataloader

We strongly recommend reading [papers](https://arxiv.org/abs/2406.09509) and [documents](https://cleandiffuserteam.github.io/CleanDiffuserDocs/) to learn more about CleanDiffuser and its design philosophy.

<p align="center">
    <br>
    <img src="assets/framework.png" width="700"/>
    <br>
<p>

<!-- NEWS -->
## 🔥 News and Change Log
- [**2025-03-10**] 🎉 TODO
- [**2025-02-15**] 🥳 We have added a diffusion planner based on empirical studies using CleanDiffuser, [Diffusion Veteran](https://openreview.net/forum?id=7BQkXXM8Fy).
- [**2024-09-26**] 🎁 Our paper [CleanDiffuser](https://arxiv.org/abs/2406.09509), has been accepted by **NeurIPS 2024 Datasets and Benchmark Track**!
- [**2024-08-27**] 🥳 We have added a lightning-fast diffusion planner, [DiffuserLite](https://arxiv.org/pdf/2401.15443), and two popular diffusion policies, [SfBC](https://arxiv.org/abs/2209.14548) and [QGPO](https://arxiv.org/abs/2304.12824), to the pipeline. Additionally, we have updated some unit tests and [API documentation](https://cleandiffuserteam.github.io/CleanDiffuserDocs/).
- [**2024-07-03**] 💫 We provided a CleanDiffuser-based replication of ACT ([action chunking with transformers](https://arxiv.org/abs/2304.13705)) in the [act branch](https://github.com/CleanDiffuserTeam/CleanDiffuser/tree/act).
- [**2024-06-24**] 🥰 We have added Consistency Models into CleanDifuser. With one model, you can do both Consistency Distillation and Consistency Training! Check out an example in `tutorials/sp_consistency_policy.py` ! (Note: Our consistency training implementation refers to the improved version, see https://arxiv.org/abs/2310.14189.)
- [**2024-06-17**] 🔥 We released arxiv version of [**CleanDiffuser: An Easy-to-use Modularized Library for Diffusion Models in Decision Making**](https://arxiv.org/abs/2406.09509). 

<!-- GETTING STARTED -->
## 🛠️ Getting Started

#### 1. Create and activate conda environment with your preferred Python version.
```bash
$ conda create -n cleandiffuser python==3.10
$ conda activate cleandiffuser
```
#### 2. Install PyTorch
Install `torch>1.0.0` that is compatible with your CUDA version. For example, `PyTorch 2.2.2` with `CUDA 12.1`:
```bash
$ conda install pytorch==2.2.2 torchvision==0.17.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```
#### 3. Install CleanDiffuser from source
```bash
$ git clone https://github.com/CleanDiffuserTeam/CleanDiffuser.git
$ git checkout lightning
$ cd CleanDiffuser
$ pip install -e .
```
#### 4. Optional Dependencies
**D4RL:** Most of our RL pipeline files run on D4RL benchmarks. Please install `mujoco` and `d4rl` following [this](https://github.com/Farama-Foundation/D4RL).

**Robomimic, PushT, Kitchen, Libero:** Most of our IL pipeline files run on these embodied AI benchmarks. Please install following instructions [here](/cleandiffuser/env/README.md).

<!-- TUTORIALS -->
## 🍷 Tutorials

After refactoring with *PyTorch Lightning*, we can now train models and use cutting-edge deep learning techniques like parallel training and mixed precision in a much more streamlined way. To help you get started, we've put together some notebook tutorials in `notebooks` folder [here](/notebooks/1_dbc_for_frankakitchen.ipynb).

<!-- USAGE EXAMPLES -->
## 💻 Pipelines

We're sorry that not all the algorithms in `pipelines` have been fully migrated yet. Some have been moved over, but they haven't been properly tested. If you need to use these algorithms, we'd suggest sticking with the main branch for now.

<!-- CONTRIBUTING -->
## 🙏 Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

<!-- LICENSE -->
## 🏷️ License

Distributed under the Apache License 2.0. See `LICENSE.txt` for more information.

<!-- CONTACT -->
## ✉️ Contact

For any questions, please feel free to email `zibindong@outlook.com` and `yuanyf@tju.edu.cn`.

<!-- CITATION -->
## 📝 Citation

If you find our work useful, please consider citing:
```
@inproceedings{dong2024cleandiffuser,
title={CleanDiffuser: An Easy-to-use Modularized Library for Diffusion Models in Decision Making},
author={Zibin Dong and Yifu Yuan and Jianye HAO and Fei Ni and Yi Ma and Pengyi Li and YAN ZHENG},
booktitle={The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, {NeurIPS}},
year={2024},
url={https://openreview.net/forum?id=7ey2ugXs36}
}
```
 