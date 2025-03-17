# CleanDiffuser Pretrained Inverse Dynamics Models

> Note: The CleanDiffuser pretrained inverse dynamics models are currently in beta. Please report any issues to me [here](zibindong@outlook.com).

A series of algorithms extended from Decision Diffuser (DD) rely on an inverse dynamic model $a_t=\mathcal I_\theta(s_t, s_{t+1})$ to infer the actions required to achieve the planned states. Training an inverse dynamic model is an essential step in the training process of DD-like models, which is repetitive and tedious. To help users bypass this process and accelerate algorithm validation, CleanDiffuser provides a set of pre-trained inverse dynamic models.

The default pretrained model architecture is a three-layer MLP, which employs GELU activation and LayerNorm, along with a Dropout layer to mitigate overfitting. The hidden size is set to 512. During training, 20% of the dataset is reserved as the validation dataset, and an early stopping strategy is applied to prevent overfitting. As a result, the number of training steps varies for each model. Further details can be found in `CleanDiffuser/cleandiffuser/invdynamic/pretrain_on_d4rl.py`.

Due to computational constraints and my personal capacity, pre-training has been conducted only on the three most mainstream offline RL benchmarks in D4RL: Mujoco, Kitchen, and AntMaze. Users are encouraged to pretrain on other benchmarks or develop higher-performance inverse dynamic models and submit pull requests! Thank you for all possible contributions!

## 1. FancyMlpInvDynamic

`FancyMlpInvDynamic` is a `LightningModule` implementation and it's our default pretrained model.
```python
from cleandiffuser.invdynamic import FancyMlpInvDynamic
```
It has the following arguments:
- obs_dim (int): Observation dimension.
- act_dim (int): Action dimension.
- hidden_dim (int): Hidden dimension of the MLP.
- tanh_out_activation (bool): Whether to apply a tanh activation to the output.
- action_scale (float): Scale the action output by this factor when using tanh.
- add_norm (bool): Whether to add LayerNorm.
- add_dropout (bool): Whether to add Dropout.

### 1.1 Training
`FancyMlpInvDynamic` requires the batch to be a dictionary with the following keys:
- obs (torch.Tensor) with shape (..., obs_dim)
- act (torch.Tensor) with shape (..., act_dim)
- next_obs (torch.Tensor) with shape (..., obs_dim)

Then you can train it with a Lightning Trainer.

### 1.2 Inference
See the following example:
```python
>>> inv_dynamic = MlpInvDynamic(5, 3)
>>> obs, next_obs = torch.randn((2, 4, 5)), torch.randn((2, 4, 5))
>>> inv_dynamic.predict(obs, next_obs).shape
torch.Size([2, 4, 3])
```

### 1.3 Load from pretrained checkpoint
Use `from_pretrained` method to create a pretrained model. It is a `classmethod` so you can use it like this:
```python
>>> from cleandiffuser.invdynamic import FancyMlpInvDynamic
>>> inv_dynamic, params = FancyMlpInvDynamic.from_pretrained("halfcheetah-medium-v2")
>>> inv_dynamic.eval()
```
`params` is a dictionary containing the hyperparameters used to train the model. **Note that the training data is Gaussian normalized to improve the model performance. Related hyperparameters are also included in this dictionary.** If you are using CleanDiffuser-provided dataset, these normalizations are compatible with the default dataset setting. Otherwise, you need to manually normalize the dataset before using the pretrained model.

Checkpoints can be downloaded from [here](https://1drv.ms/u/c/ba682474b24f6989/EVgO7dzZXIJNr-pZfk511GMBX2cAIMr30S7uz9XkML_fyA?e=y2GSHI). Please download this zip file and extract it to `~/`.

Our `decision_diffuser` pipeline also uses these pretrained models to infer the actions. See `CleanDiffuser/pipelines/decision_diffuser/dd_d4rl.py` and take it as an example to use pretrained inverse dynamic models!
