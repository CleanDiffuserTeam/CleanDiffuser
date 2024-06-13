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

# CleanDiffuser: 易于使用的模块化决策扩散模型库

<p align="center">
·
<a href="">预印本</a>
·
<a href="https://cleandiffuserteam.github.io/CleanDiffuserDocs/">文档</a>
·
<a href="README_ZH.md">中文版</a>
·
</p>

**CleanDiffuser** 是一个易用的模块化扩散模型库，专为决策任务定制，综合集成了不同类型的扩散算法分支。CleanDiffuser 以简单且用户友好的方式提供多种高级的 扩散模型、网络结构、多样的条件嵌入和算法管道。继承了 [CleanRL](https://github.com/vwxyzjn/cleanrl) 和 [Diffusers](https://github.com/huggingface/diffusers) 的核心设计理念，CleanDiffuser 强调可用性、简单性和可定制性。我们希望 CleanDiffuser 能成为一个基础工具库，为决策领域的扩散模型研究提供长期支持，促进科研人员和实践者的应用。CleanDiffuser 的亮点功能包括：

- 🚀 专为决策制定任务设计的惊人特性  
- 🍧 支持多种高级扩散模型和网络结构  
- 🧩 像搭积木一样轻松将解耦的模块构建成集成的管道
- 📈 Wandb 日志记录和 Hydra 配置  
- 🌏 统一的环境接口和高效的数据加载器  

我们强烈推荐阅读 [papers]() 和 [documents](https://cleandiffuserteam.github.io/CleanDiffuserDocs/) 以了解更多关于 CleanDiffuser 及其设计理念的信息。

<p align="center">
    <br>
    <img src="assets/framework.png" width="700"/>
    <br>
<p>

## 🛠️ 快速启动

我们推荐通过 Conda 虚拟环境安装和体验 CleanDiffuser。

首先，安装与 mujoco-py 环境相关的依赖。详情见 https://github.com/openai/mujoco-py#install-mujoco

```bash
apt-get install libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```

下载 CleanDiffuser 并将该文件夹添加到你的 PYTHONPATH。你也可以将其添加到 .bashrc 中以方便使用：
```bash
git clone https://github.com/CleanDiffuserTeam/CleanDiffuser.git
export PYTHONPATH=$PYTHONPATH:/path/to/CleanDiffuser
```

安装 Conda 虚拟环境和 PyTorch：  
```bash
conda create -n cleandiffuser python==3.9
conda activate cleandiffuser
# pytorch
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
# pytorch3d
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1121/download.html
```

安装剩余依赖：
```bash
pip install -r requirements.txt

# If you need to run D4RL-related environments, install D4RL additionally:
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
```

如果需要复现模仿学习相关环境（PushT、Kitchen、Robomimic），需要额外下载数据集。我们推荐从 [Datasets](https://diffusion-policy.cs.columbia.edu/data/training/) 下载对应的压缩文件。如下面文件结构所示，我们提供默认的数据集路径为 `dev/`：

```bash
dev/
.
├── kitchen
├── pusht_cchi_v7_replay.zarr
├── robomimic
```

<!-- USAGE EXAMPLES -->
## 💻 如何使用CleanDiffuser

`cleandiffuser` 文件夹包含了 CleanDiffuser 代码库的核心组件，包括 扩散模型 (`Diffusion Models`)、网络架构 (`Network Architectures`) 和 引导采样 (`Guided Sampling`)。它还提供了统一的 环境和数据集接口 (`Env and Dataset Interfaces`)。

在 `tutorials` 文件夹中，我们提供了最简单的可运行教程和算法，可以与文档结合理解。

在 CleanDiffuser 中，我们可以像搭积木一样将独立模块组合成算法管道 (algorithmic pipeline)。 `pipelines` 文件夹中提供了 CleanDiffuser 中目前实现的所有算法。通过链接 `configs` 文件夹中的 Hydra 配置，可以复现论文中展示的结果：

- 你可以简单地使用默认环境和配置运行每个算法最简单的感受算法实现细节，无需额外设置，例如：

```bash
# DiffusionPolicy with Chi_UNet in lift-ph
python pipelines/dp_pusht.py
# Diffuser in halfcheetah-medium-expert-v2
python pipelines/diffuser_d4rl_mujoco.py
```

- 得益于 Hydra，CleanDiffuser 还支持通过 CLI 或直接修改相应的配置文件灵活运行算法。我们提供了一些示例：  

```bash
# Load PushT config
python pipelines/dp_pusht.py --config-path=../configs/dp/pusht/dit --config-name=pusht
# Load PushT config and overwrite some hyperparameters
python pipelines/dp_pusht.py --config-path=../configs/dp/pusht/dit --config-name=pusht dataset_path=path/to/dataset seed=42 device=cuda:1
# Train Diffuser in hopper-medium-v2 task
python pipelines/diffuser_d4rl_mujoco.py task=hopper-medium-v2 
```

- 在 CleanDiffuser 中，我们提供了模式 (`mode`) 选项以在模型的 **训练** (`mode=train`) 或 **推理** (`mode=inference`) 之间切换：

```bash
# Imitation learning environment
python pipelines/dp_pusht.py mode=inference model_path=path/to/checkpoint
# Reinforcement learning environment
python pipelines/diffuser_d4rl_mujoco.py mode=inference ckpt=latest
```

<!-- ## 💫 Feature -->



<!-- Implemented Components -->
## 🎁 已实现的子模块和相关算法


| **Category**                | **Items**                      |
|-----------------------------|--------------------------------|
| **SDE/ODE with Solvers**    |                                |
| *Diffusion SDE*             | DDPM                           |
|                             | DDIM                           |
|                             | DPM-Solver                     |
|                             | DPM-Solver++                   |
| *EDM*                       | Eular                          |
|                             | 2nd Order Heun                 |
| *Recitified Flow*           | Euler                          |
| **Network Architectures**   | Pearce_MLP                     |
|                             | Chi_UNet1d                     |
|                             | Pearce_Transformer             |
|                             | LNResnet                       |
|                             | Chi_Transformer                |
|                             | DQL_MLP                        |
|                             | Janner_UNet1d                  |
|                             | DiT1d                          |
|                             |                                |
| **Guided Sampling Methods** | Classifier Guidance            |
|                             | Classifier-free Guidance       |
|                             |                                |
| **Pipelines**               |                                |
| *Planners*                  | Diffuser                       |
|                             | Decision Diffuser              |
|                             | AdaptDiffuser                  |
| *Policies*                  | DQL                            |
|                             | EDP                            |
|                             | IDQL                           |
|                             | Diffusion Policy               |
|                             | DiffusionBC                    |
| *Data Synthesizers*         | SynthER                        |
|                             |                                |


<!-- LICENSE -->
## 🏷️ 许可证

使用Apache License 2.0许可证，查看`LICENSE.txt`或许详细许可证信息。

<!-- CONTACT -->
## ✉️ 联系

如有任何问题，请随时发送邮件至`zibindong@outlook.com` (董子斌)或者 `yuanyf@tju.edu.cn` (袁逸夫)。

## 📝 引用

如果你发现这个库能帮到你，请考虑引用：  
```
@misc{cleandiffuser,
  author = {CleanDiffuserTeam},
  title = {CleanDiffuser},
  year = {2024},
  howpublished = {\url{https://github.com/CleanDiffuserTeam/CleanDiffuser}},
}
```

