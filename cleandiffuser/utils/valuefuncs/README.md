# CleanDiffuser Pretrained IQL

> Note: The CleanDiffuser pretrained IQL models are currently in beta. Please report any issues to me [here](zibindong@outlook.com).

IQL is an offline RL algorithm that estimates the value function of the optimal policy. It is also widely used in diffusion-based decision-making models, such as IDQL and DiffuserLite. Generally, diffusion models are employed to generate a large number of candidates, after which IQL is used for value evaluation to select the best one. Since IQL training is policy-agnostic, it can be treated as an independent and decoupled component. To avoid tedious repeated training, we have pre-trained a batch of IQL models for direct use, facilitating the development and validation of algorithms.

Our implementation uses a three-layer MLP with GELU activation and LayerNorm for both Q and V functions. The hidden size is set to 512. Other hyperparameters are consistent with the original IQL paper.

Due to computational constraints and my personal capacity, pre-training has been conducted only on the three most mainstream offline RL benchmarks in D4RL: Mujoco, Kitchen, and AntMaze. Users are encouraged to pretrain on other benchmarks or develop higher-performance IQL models and submit pull requests! Thank you for all possible contributions!

## 1. Iql

`Iql` is a `LightningModule` implementation and it's our default pretrained model.
```python
from cleandiffuser.utils.valuefuncs import Iql
```
It has the following arguments:
- obs_dim (int): Observation dimension.
- act_dim (int): Action dimension.
- tau (float): IQL quantile level. 0.9 for antmaze and 0.7 for others.
- discount (float): Discount factor. Default is 0.99.
- hidden_dim (int): Hidden dimension. Default is 512.
- q_ensembles (int): Number of Q ensembles. Default is 2.
- v_ensembles (int): Number of V ensembles. Default is 1.
- ema_ratio (float): Q target EMA ratio. Default is 0.99.
- lr (float): Learning rate. Default is 3e-4.

### 1.1 Training
`Iql` requires the batch to be a dictionary with the following keys:
- obs (torch.Tensor) with shape (..., obs_dim)
- act (torch.Tensor) with shape (..., act_dim)
- next_obs (torch.Tensor) with shape (..., obs_dim)
- rew (torch.Tensor) with shape (..., 1)
- done (torch.Tensor) with shape (..., 1)

Then you can train it with a Lightning Trainer.

### 1.2 Inference
Use the following two methods to get Q and V values:

**Iql.forward_q(self, obs: torch.Tensor, act: torch.Tensor, use_ema: bool=False, requires_grad: bool=False)**
> **Args:**
> - obs (torch.Tensor): Observation tensor with shape (..., obs_dim).
> - act (torch.Tensor): Action tensor with shape (..., act_dim).
> - use_ema (bool): Whether to use EMA target. Default is False.
> - requires_grad (bool): Whether to require gradient. Default is False.
>
> **Returns:**
> - torch.Tensor: Q value tensor with shape (..., 1). It is the minimum Q value among all Q ensembles.

**Iql.forward_v(self, obs: torch.Tensor, requires_grad: bool=False)**
> **Args:**
> - obs (torch.Tensor): Observation tensor with shape (..., obs_dim).
> - requires_grad (bool): Whether to require gradient. Default is False.
>
> **Returns:**
> - torch.Tensor: V value tensor with shape (..., 1).

### 1.3 Load from pretrained checkpoint
Use `from_pretrained` method to create a pretrained model. It is a `classmethod` so you can use it like this:
```python
from cleandiffuser.utils.valuefuncs import Iql
>>> iql, params = Iql.from_pretrained("halfcheetah-medium-v2")
>>> iql.eval()
```
`params` is a dictionary containing the hyperparameters used to train the model. **Note that the training data is Gaussian normalized to improve the model performance. Related hyperparameters are also included in this dictionary.** If you are using CleanDiffuser-provided dataset, these normalizations are compatible with the default dataset setting. Otherwise, you need to manually normalize the dataset before using the pretrained model.

Checkpoints can be downloaded from [here](https://1drv.ms/u/c/ba682474b24f6989/EVgO7dzZXIJNr-pZfk511GMBX2cAIMr30S7uz9XkML_fyA?e=y2GSHI). Please download this zip file and extract it to `~/`.

Our `idql` pipeline also uses these pretrained models to infer the actions. See `CleanDiffuser/pipelines/idql/idql_d4rl.py` and take it as an example to use pretrained IQL models!
