from typing import Dict, Union
import torch

TensorDict = Dict[str, Union['TensorDict', torch.Tensor]]
