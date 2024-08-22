from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.nn_diffusion import IDQLMlp
from cleandiffuser.nn_condition import MLPCondition


from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
from cleandiffuser.utils import IQL
import hydra

@hydra.main(config_path="../configs/idql/mujoco", config_name="mujoco", version_base=None)
def pipeline(args):

    pass