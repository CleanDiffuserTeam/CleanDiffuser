from typing import Optional

import torch
import torch.nn as nn

from cleandiffuser.utils import PositionalEmbedding, FourierEmbedding


class BaseNNDiffusion(nn.Module):
    """
    The neural network backbone for the Diffusion model used for score matching
     (or training a noise predictor) should take in three inputs.
     The first input is the noisy data.
     The second input is the denoising time step, which can be either as a discrete variable
     or a continuous variable, specified by the parameter `discrete_t`.
     The third input is the condition embedding that has been processed through the `nn_condition`.
     In the general case, we assume that there may be multiple conditions,
     which are inputted as a tensor dictionary, or a single condition, directly inputted as a tensor.
    """

    def __init__(self, emb_dim: int, timestep_emb_type: str = "positional"):
        assert timestep_emb_type in ["positional", "fourier"]
        super().__init__()
        self.map_noise = PositionalEmbedding(emb_dim, endpoint=True) if timestep_emb_type == "positional" \
            else FourierEmbedding(emb_dim)

    def forward(self,
                x: torch.Tensor, noise: torch.Tensor,
                condition: Optional[torch.Tensor] = None):
        """
        Input:
            x:          (b, horizon, in_dim)
            noise:      (b, )
            condition:  (b, emb_dim) or None / No condition indicates zeros((b, emb_dim))

        Output:
            y:          (b, horizon, in_dim)
        """
        raise NotImplementedError

