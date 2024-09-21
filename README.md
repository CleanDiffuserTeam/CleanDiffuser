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

<!-- - [**2024-07-16**] 🔥 We have open-sourced the exciting Diffusion Planning algorithm, [DiffuserLite](https://arxiv.org/pdf/2401.15443)! DiffuserLite achieves a decision frequency of **122Hz**, which is **112.7 times** of previous Diffusion Planning frameworks. -->
- [**2024-07-03**] 💫 We provided a CleanDiffuser-based replication of ACT ([action chunking with transformers](https://arxiv.org/abs/2304.13705)) in the [act branch](https://github.com/CleanDiffuserTeam/CleanDiffuser/tree/act).
- [**2024-06-24**] 🥰 We have added Consistency Models into CleanDifuser. With one model, you can do both Consistency Distillation and Consistency Training! Check out an example in `tutorials/sp_consistency_policy.py` ! (Note: Our consistency training implementation refers to the improved version, see https://arxiv.org/abs/2310.14189.)
- [**2024-06-17**] 🔥 We released arxiv version of [**CleanDiffuser: An Easy-to-use Modularized Library for Diffusion Models in Decision Making**](https://arxiv.org/abs/2406.09509). 

<!-- GETTING STARTED -->
## 🛠️ Getting Started

#### 1. Create and activate conda environment
```bash
$ conda create -n cleandiffuser python==3.9
$ conda activate cleandiffuser
```
#### 2. Install PyTorch
Install `torch>1.0.0,<2.3.0` that is compatible with your CUDA version. For example, `PyTorch 2.2.2` with `CUDA 12.1`:
```bash
$ conda install pytorch==2.2.2 torchvision==0.17.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```
#### 3. Install CleanDiffuser from source
```bash
$ git clone https://github.com/CleanDiffuserTeam/CleanDiffuser.git
$ cd CleanDiffuser
$ pip install -e .
```
#### 4. Additional installations
For users who need to run `pipelines` and reproduce the results of the paper, they will need to install RL simulators.

First, install the dependencies related to the mujoco-py environment. For more details, see https://github.com/openai/mujoco-py#install-mujoco

```bash
$ sudo apt-get install libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev patchelf
```
```bash
# Install D4RL from source (recommended)
$ cd <PATH_TO_D4RL_INSTALL_DIR>
$ git clone https://github.com/Farama-Foundation/D4RL.git
$ cd D4RL
$ pip install -e .
# Install Robomimic from source (recommended)
$ cd <PATH_TO_ROBOMIMIC_INSTALL_DIR>
$ git clone https://github.com/ARISE-Initiative/robomimic.git
$ cd robomimic
$ pip install -e .
$ cd <PATH_TO_ROBOSUITE_INSTALL_DIR>
$ git clone https://github.com/ARISE-Initiative/robosuite.git
$ cd robosuite
$ pip install -e .
```

> **Note:** The latest version of dependencies running the `robomimic image` still has compatibility issues, and we are actively working on a fix. The temporary solution is to downgrade the `gym` version to `0.21.0`: pip install setuptools==65.5.0 pip==21, pip install gym==0.21.0

Try it now!   
```bash
# Tutorial
$ python tutorials/1_a_minimal_DBC_implementation.py
# Reinforcement Learning
$ python pipelines/diffuser_d4rl_mujoco.py
# Imitation Learning (need to download the dataset, see below)
$ python pipelines/dp_pusht.py
```
If you need to reproduce Imitation Learning environments (`pusht`, `kitchen`, `robomimic`), you need to download the datasets additionally. We recommend downloading the corresponding compressed files from [Datasets](https://diffusion-policy.cs.columbia.edu/data/training/). We provide the default dataset path as `dev/`:
```bash
dev/
.
├── kitchen
├── pusht
├── robomimic
```

<!-- TUTORIALS -->
## 🍷 Tutorials

After refactoring with *PyTorch Lightning*, we can now train models and use cutting-edge deep learning techniques like parallel training and mixed precision in a much more streamlined way. To help you get started, we've put together some notebook tutorials in `notebooks` folder.

<!-- USAGE EXAMPLES -->
## 💻 Pipelines

We're sorry that not all the algorithms in `pipelines` have been fully migrated yet. Some have been moved over, but they haven't been properly tested. If you need to use these algorithms, we'd suggest sticking with the main branch for now.

<!-- ## 💫 Feature -->


<!-- Implemented Components -->
## 🎁 Implemented Components


| **Category**                | **Items**                      | **Paper**                      |
|-----------------------------|--------------------------------|--------------------------------|
| **SDE/ODE with Solvers**    |                                |                                |
| *Diffusion SDE*             | DDPM                           |✅[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)|
|                             | DDIM                           |✅[Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)|
|                             | DPM-Solver                     |✅[DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://arxiv.org/abs/2206.00927)|
|                             | DPM-Solver++                   |✅[DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models](https://arxiv.org/abs/2211.01095)|
| *EDM*                       | Eular                          |✅[Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)|
|                             | 2nd Order Heun                 |                                |
| *Recitified Flow*           | Euler                          |✅[Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)|
| *Consistency Models*        |                                |✅[Consistency Models](https://arxiv.org/abs/2303.01469)|
|                             |                                |                                |
| **Network Architectures**   |                                |                                |
|                             | Pearce_MLP                     |✅[Imitating Human Behaviour with Diffusion Models](https://arxiv.org/abs/2301.10677)|                                |
|                             | Pearce_Transformer             |                                |
|                             | Chi_UNet1d                     |✅[Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)|                                |
|                             | Chi_Transformer                |                                |
|                             | LNResnet                       |✅[IDQL: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies](https://arxiv.org/abs/2304.10573)|                                
|                             | DQL_MLP                        |✅[Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning](https://arxiv.org/abs/2208.06193)|                                
|                             | Janner_UNet1d                  |✅[Planning with Diffusion for Flexible Behavior Synthesis](https://arxiv.org/abs/2205.09991)|                       
|                             | DiT1d                          |✅[AlignDiff: Aligning Diverse Human Preferences via Behavior-Customisable Diffusion Model](https://arxiv.org/abs/2310.02054)|                       
|                             |                                |                                |
| **Guided Sampling Methods** |                                |                                |
|                             | Classifier Guidance            |✅[Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)|                                 
|                             | Classifier-free Guidance       |✅[Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)|                                                                 
|                             |                                |                                |
| **Pipelines**               |                                |                                |
| *Planners*                  | Diffuser                       |✅[Planning with Diffusion for Flexible Behavior Synthesis](https://arxiv.org/abs/2205.09991)|
|                             | Decision Diffuser              |✅[Is Conditional Generative Modeling all you need for Decision-Making?](https://arxiv.org/abs/2211.15657)|
|                             | AdaptDiffuser                  |✅[AdaptDiffuser: Diffusion Models as Adaptive Self-evolving Planners](https://arxiv.org/abs/2302.01877)|
|                             | DiffuserLite (*New!*)🔥        |✅[DiffuserLite: Towards Real-time Diffusion Planning](https://arxiv.org/abs/2401.15443)|
| *Policies*                  | DQL                            |✅[Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning](https://arxiv.org/abs/2208.06193)| 
|                             | EDP                            |✅[Efficient Diffusion Policies for Offline Reinforcement Learning](https://arxiv.org/abs/2305.20081)| 
|                             | IDQL                           |✅[IDQL: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies](https://arxiv.org/abs/2304.10573)|
|                             | Diffusion Policy               |✅[Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)|                                
|                             | DiffusionBC                    |✅[Imitating Human Behaviour with Diffusion Models](https://arxiv.org/abs/2301.10677)|                                
| *Data Synthesizers*         | SynthER                        |✅[Synthetic Experience Replay](https://arxiv.org/abs/2303.06614)|                                
|                             |                                |                                |

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

<!-- LICENSE -->
## 🏷️ License

Distributed under the Apache License 2.0. See `LICENSE.txt` for more information.

<!-- ACKNOWLEDGEMENT -->
## 💓 Acknowledgement

- [huggingface diffusers](https://github.com/huggingface/diffusers)  
- [diffuser](https://github.com/jannerm/diffuser)  
- [diffusion policy](https://github.com/real-stanford/diffusion_policy)  
- [robomimic](https://github.com/ARISE-Initiative/robomimic)

<!-- CONTACT -->
## ✉️ Contact

For any questions, please feel free to email `zibindong@outlook.com` and `yuanyf@tju.edu.cn`.

<!-- CITATION -->
## 📝 Citation

If you find our work useful, please consider citing:
```
@article{cleandiffuser,
  author = {Zibin Dong and Yifu Yuan and Jianye Hao and Fei Ni and Yi Ma and Pengyi Li and Yan Zheng},
  title = {CleanDiffuser: An Easy-to-use Modularized Library for Diffusion Models in Decision Making},
  journal = {arXiv preprint arXiv:2406.09509},
  year = {2024},
  url = {https://arxiv.org/abs/2406.09509},
}
```
