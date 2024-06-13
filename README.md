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
<a href="">ArXiv</a>
·
<a href="https://cleandiffuserteam.github.io/CleanDiffuserDocs/">Documentation</a>
·
<a href="README_ZH.md">中文版</a>
·
</p>

**CleanDiffuser** is an easy-to-use modularized Diffusion Model library tailored for decision-making, which comprehensively integrates different types of diffusion algorithmic branches. CleanDiffuser offers a variety of advanced *diffusion models*, *network structures*, diverse *conditions*, and *algorithm pipelines* in a simple and user-friendly manner. Inheriting the design philosophy of [CleanRL](https://github.com/vwxyzjn/cleanrl) and [Diffusers](https://github.com/huggingface/diffusers), CleanDiffuser emphasizes **usability, simplicity, and customizability**. We hope that CleanDiffuser will serve as a foundational tool library, providing long-term support for Diffusion Model research in the decision-making community, facilitating the application of research for scientists and practitioners alike. The highlight features of CleanDiffuser are:

- 🚀 Amazing features specially tailored for decision-making tasks
- 🍧 Support for multiple advanced diffusion models and network architectures
- 🧩 Build decoupled modules into integrated pipelines easily like building blocks
- 📈 Wandb logging and Hydra configuration
- 🌏 Unified environmental interface and efficient dataloader

We strongly recommend reading [papers]() and [documents](https://cleandiffuserteam.github.io/CleanDiffuserDocs/) to learn more about CleanDiffuser and its design philosophy.

<p align="center">
    <br>
    <img src="assets/framework.png" width="700"/>
    <br>
<p>
    
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
<ol>
<li><a href="#getting-started">Getting Started</a></li>
<li><a href="#usage">Usage</a></li>
<li><a href="#feature">Feature</a></li>
<li><a href="#algorithm">Implemented Components</a></li>
<li><a href="#roadmap">Roadmap</a></li>
<li><a href="#contributing">Contributing</a></li>
<li><a href="#license">License</a></li>
<li><a href="#contact">Contact</a></li>
<li><a href="#acknowledgments">Acknowledgments</a></li>
</ol>
</details>

<!-- GETTING STARTED -->
## 🛠️ Getting Started

We recommend installing and experiencing CleanDiffuser through a Conda virtual environment.

First, install the dependencies related to the mujoco-py environment. For more details, see https://github.com/openai/mujoco-py#install-mujoco

```bash
apt-get install libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```

Download CleanDiffuser and add this folder to your PYTHONPATH. You can also add it to .bashrc for convenience:
```bash
git clone https://github.com/CleanDiffuserTeam/CleanDiffuser.git
export PYTHONPATH=$PYTHONPATH:/path/to/CleanDiffuser
```

Install the Conda virtual environment and PyTorch:
```bash
conda create -n cleandiffuser python==3.9
conda activate cleandiffuser
# pytorch
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
# pytorch3d
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1121/download.html
```

Install the remaining dependencies:
```bash
pip install -r requirements.txt

# If you need to run D4RL-related environments, install D4RL additionally:
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
```

If you need to reproduce imitation learning related environments (PushT, Kitchen, Robomimic), you need to download the datasets additionally. We recommend downloading the corresponding compressed files from [Datasets](https://diffusion-policy.cs.columbia.edu/data/training/). We provide the default dataset path as `dev/`:

```bash
dev/
.
├── kitchen
├── pusht_cchi_v7_replay.zarr
├── robomimic
```

<!-- USAGE EXAMPLES -->
## 💻 Usage

The `cleandiffuser` folder contains the core components of the CleanDiffuser codebase, including `Diffusion Models`, `Network Architectures`, and `Guided Sampling`. It also provides unified `Env and Dataset Interfaces`.

In the `tutorials` folder, we provide the simplest runnable tutorials and algorithms, which can be understood in conjunction with the documentation.

<!-- ```bash
# Build the DiffusionBC algorithm with minimal code
python tutorials/1_a_minimal_DBC_implementation.py
# Construct a Diffusion Model without guidance
python tutorials/2_classifier-free_guidance.py
# Construct a Diffusion Model with guidance
python tutorials/3_classifier_guidance
# Apply your customized Diffusion Model
python tutorials/4_customize_your_diffusion_network_backbone
``` -->

In CleanDiffuser, we can combine independent modules to algorithms pipelines like building blocks. In the `pipelines` folder, we provide all the algorithms currently implemented in CleanDiffuser. By linking with the Hydra configurations in the `configs` folder, you can reproduce the results presented in the papers:

You can simply run each algorithm with the default environment and configuration without any additional setup, for example:

```bash
# DiffusionPolicy with Chi_UNet in lift-ph
python pipelines/dp_pusht.py
# Diffuser in halfcheetah-medium-expert-v2
python pipelines/diffuser_d4rl_mujoco.py
```

Thanks to Hydra, CleanDiffuser also supports flexible running of algorithms through CLI or directly modifying the corresponding configuration files. We provide some examples:

```bash
# Load PushT config
python pipelines/dp_pusht.py --config-path=../configs/dp/pusht/dit --config-name=pusht
# Load PushT config and overwrite some hyperparameters
python pipelines/dp_pusht.py --config-path=../configs/dp/pusht/dit --config-name=pusht dataset_path=path/to/dataset seed=42 device=cuda:1
# Train Diffuser in hopper-medium-v2 task
python pipelines/diffuser_d4rl_mujoco.py task=hopper-medium-v2 
```

In CleanDiffuser, we provide a mode option to switch between **training** `(mode=train)` or **inference** `(mode=inference)` of the model:

```bash
# Imitation learning environment
python pipelines/dp_pusht.py mode=inference model_path=path/to/checkpoint
# Reinforcement learning environment
python pipelines/diffuser_d4rl_mujoco.py mode=inference ckpt=latest
```

<!-- ## 💫 Feature -->



<!-- Implemented Components -->
## 🎁 Implemented Components


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



<!-- ROADMAP -->
## 🧭 Roadmap

- [ ] Updating the reproduced [ACT](https://arxiv.org/abs/2304.13705) and [BC-RNN](https://arxiv.org/abs/2108.03298) for comparison
- [ ] Unifying some old APIs into a new unified version

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## 🙏 Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## 🏷️ License

Distributed under the Apache License 2.0. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## ✉️ Contact

For any questions, please feel free to email `zibindong@outlook.com` and `yuanyf@tju.edu.cn`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 📝 Citation

If you find our work useful, please consider citing:
```
@misc{cleandiffuser,
  author = {CleanDiffuserTeam},
  title = {CleanDiffuser},
  year = {2024},
  howpublished = {\url{https://github.com/CleanDiffuserTeam/CleanDiffuser}},
}
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>
