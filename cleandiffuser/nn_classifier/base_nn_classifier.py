import torch
import torch.nn as nn

from cleandiffuser.utils import PositionalEmbedding, FourierEmbedding


class BaseNNClassifier(nn.Module):
    """
    Base Neural Network (NN) for Classifiers.

    This NN is supposed to implement the mapping: (x, t, y) -> log p(y | x, t) + C, where C is a constant.
    From a coding perspective, the output of the NN should be a real number with dimension=1.

    Parameters:
        - emb_dim: int
            Dimension of the embedding for the time variable t.
        - timestep_emb_type: str
            Type of embedding for the time variable t. Options are: "positional" or "fourier".
    """
    def __init__(self, emb_dim: int, timestep_emb_type: str = "positional"):
        assert timestep_emb_type in ["positional", "fourier"]
        super().__init__()
        self.map_noise = PositionalEmbedding(emb_dim, endpoint=True) if timestep_emb_type == "positional" \
            else FourierEmbedding(emb_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Input:
            x:  (b, *x_shape)
            t:  (b, )
            y:  (b, *y_shape)

        Output:
            logp: (b, 1)
        """
        raise NotImplementedError

