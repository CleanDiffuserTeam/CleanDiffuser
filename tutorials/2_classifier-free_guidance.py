import os

import d4rl
import gym
import numpy as np
import torch
import torch.nn as nn

from cleandiffuser.diffusion.diffusionsde import DiscreteDiffusionSDE
from cleandiffuser.nn_condition import BaseNNCondition, get_mask, MLPCondition
from cleandiffuser.nn_diffusion import PearceMlp

"""
In this tutorial, we will review the applications of classifier-free guidance (CFG) and how to customize it for a diffusion model. 

In Appendix A.2 of the paper, we provide a detailed explanation of CFG. But in simple terms, we achieve CFG by learning a conditional diffusion model \epsilon(x_t,t,c) 
and an unconditional one \epsilon(x_t, t), using their weighted sum w*\epsilon(x_t,t,c)+(1-w)*\epsilon(x_t, t) as the predicted noise/data. 
The weight w influences the guidance strength.
The selection w=0 and w=1 are two special cases, where the former is equivalent to sampling from p(x) and the latter is equivalent to sampling from p(x|c). 
In these cases, CFG actually does not play a role.

In practice, we do not actually train two diffusion models; instead, we can treat the unconditional model as a special case of the conditional model 
by introducing a specific condition variable c=\Phi to represent the unconditional sampling. 
Following common practices, we encode the condition variable into features using `nn_condition`, and use an all-zero feature indicates no condition \Phi,
i.e., conditional_feature = nn_condition(c) and unconditional_feature = zeros(*feature_shape).

This requires label dropout during training, where the condition feature is set to zero with a certain probability.
The `dropout` parameter in `nn_condition` determines the probability, which is typically around `dropout=0.25`.

In tutorial 1, we simply set dropout=0 because we directly fit the diffusion model to p(a|s), equivalent to w=1, and do not require CFG. 
In this tutorial, we attempt to customize a CFG version.
"""

"""
Let's begin by customizing a `nn_condition`, which aims to encode the condition variable into features for `nn_diffusion` to use. 
Imaging a complex scenario where our condition variable may include 3D point clouds, language instructions, images, and so on, 
we might require a very large `nn_condition`. Designing a decoupled `nn_condition` module will make development very convenient. 
In this tutorial, our condition variable is just a simple low-dim state.

Our `nn_condition` needs to meet the following requirements:
1. The parent class is `nn.Module`, and the input parameters include `dropout`.
2. The `forward` method should accept two arguments: `condition` and `mask`, and be able to encode the `condition` into `feature`.
3. If `self.training=True`, generate a mask with dropout probability and set the corresponding parts of `feature` to 0. 
    If `mask` is not None, prioritize setting the corresponding parts of `feature` to 0 based on the input mask.
4. The output shape of `feature` should align with the requirements of `nn_diffusion`.

Now we want to customize a `MyObsNNCondition` that interfaces with `PearceMlp`. 
`PearceMlp` requires the shape of the condition features to be (b, To*emb_dim), and since we are using `To=1`, it becomes (b, emb_dim). 
Therefore, we intend to use a two-layer MLP to encode the low-dim state, i.e., (b, state_dim) -> MLP -> (b, emb_dim), 
and during training, generate a mask of shape (b, 1) with `dropout` probability, and multiply it with the feature.

The remaining works are the same as tutorial 1. We just need to replace the original `nn_condition` with `MyObsNNCondition`.
"""


class MyObsNNCondition(BaseNNCondition):
    def __init__(self, obs_dim, emb_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, emb_dim))
        self.dropout = dropout

    def forward(self, condition: torch.Tensor, mask: torch.Tensor = None):
        mask = get_mask(mask, (condition.shape[0], 1), self.dropout, self.training, condition.device)
        return self.mlp(condition) * mask


if __name__ == "__main__":

    device = "cuda:0"
    use_customized_nn_condition = True

    # --------------- Create Environment ---------------
    env = gym.make("kitchen-complete-v0")
    dataset = d4rl.qlearning_dataset(env)
    obs_dim, act_dim = dataset['observations'].shape[-1], dataset['actions'].shape[-1]
    size = len(dataset['observations'])

    # --------------- Network Architecture -----------------
    nn_diffusion = PearceMlp(act_dim, To=1, emb_dim=64, hidden_dim=256, timestep_emb_type="positional")

    """
    The `MyObsNNCondition` we defined is essentially an MLP encoder, a design that is simple and commonly used. 
    We certainly do not want to rewrite it from scratch every time we need to use it. 
    Fortunately, CleanDiffuser has already implemented an `MLPCondition`, which can easily achieve the same effect. 
    The following two `nn_condition` are equivalent.
    """

    if use_customized_nn_condition:
        nn_condition = MyObsNNCondition(obs_dim, emb_dim=64, hidden_dim=64, dropout=0.2)
    else:
        nn_condition = MLPCondition(
            in_dim=obs_dim, out_dim=64, hidden_dims=[64, ], act=nn.SiLU(), dropout=0.2)

    # --------------- Diffusion Model Actor --------------------
    actor = DiscreteDiffusionSDE(
        nn_diffusion, nn_condition, predict_noise=False, optim_params={"lr": 3e-4},
        x_max=+1. * torch.ones((1, act_dim)),
        x_min=-1. * torch.ones((1, act_dim)),
        diffusion_steps=5, ema_rate=0.9999, device=device)

    # --------------- Training -------------------
    actor.train()

    avg_loss = 0.
    for t in range(100000):

        idx = np.random.randint(0, size, (256,))
        obs = torch.tensor(dataset['observations'][idx], device=device).float()
        act = torch.tensor(dataset['actions'][idx], device=device).float()

        avg_loss += actor.update(act, obs)["loss"]

        if (t + 1) % 1000 == 0:
            print(f'[t={t + 1}] {avg_loss / 1000}')
            avg_loss = 0.

    savepath = "tutorials/results/2_classifier-free_guidance/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    actor.save(savepath + "diffusion.pt")

    # -------------- Inference -----------------
    savepath = "tutorials/results/2_classifier-free_guidance/"
    actor.load(savepath + "diffusion.pt")
    actor.eval()

    env_eval = gym.vector.make("kitchen-complete-v0", num_envs=50)

    obs, cum_done, cum_rew = env_eval.reset(), 0., 0.
    prior = torch.zeros((50, act_dim), device=device)
    for t in range(280):

        """
        Set `1 > w_cfg > 0` or `w_cfg > 1` to enable CFG.
        """
        act, log = actor.sample(
            prior, solver="ddpm", n_samples=50, sample_steps=5,
            temperature=0.5, w_cfg=1.2,
            condition_cfg=torch.tensor(obs, device=device, dtype=torch.float32))
        act = act.cpu().numpy()

        obs, rew, done, info = env_eval.step(act)
        cum_done = np.logical_or(cum_done, done)
        cum_rew += rew

        print(f'[t={t}] cum_rew: {cum_rew}')

        if cum_done.all():
            break

    print(f'Mean score: {np.clip(cum_rew, 0., 4.).mean() * 25.}')
    env_eval.close()

"""
The implementation in tutorial 1 is actually a special case of tutorial 2 with `w_cfg=1.0`. 
Mathematically speaking, guidance expands `p(x|c)` into `p(x)*(p(c|x))**w`. Adjusting `w` can make the distribution of the condition smoother or sharper. 
In text-to-image applications, a larger `w` can make the generated images better match the text. 
However, in policy modeling, what is the benefit of making the generated actions match the states more closely? I don't know.
Let's run multiple inferences to test it out! 
"""
