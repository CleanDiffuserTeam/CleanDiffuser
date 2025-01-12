import h5py
import numpy as np
import torch
import tqdm

from cleandiffuser.dataset.dataset_utils import RotationTransformer
from cleandiffuser.utils import MinMaxNormalizer, create_indices


class ActionConverter:
    def __init__(self, abs_action: bool = True, rotation_rep: str = "rotation_6d"):
        self.rotation_tf = RotationTransformer(from_rep="axis_angle", to_rep=rotation_rep)
        self.abs_action = abs_action

    def transform(self, action: np.ndarray):
        leading_dim = action.shape[:-1]
        if self.abs_action:
            is_dual_arm = action.shape[-1] == 14
            if is_dual_arm:
                action = action.reshape(*leading_dim, 2, 7)

            pos = action[..., :3]
            rot = action[..., 3:6]
            gripper = action[..., 6:]
            rot = self.rotation_tf.forward(rot)
            action = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)

            if is_dual_arm:
                action = action.reshape(*leading_dim, 20)
        return action

    def inverse_transform(self, action: np.ndarray):
        leading_dim = action.shape[:-1]
        if self.abs_action:
            is_dual_arm = action.shape[-1] == 20
            if is_dual_arm:
                action = action.reshape(*leading_dim, 2, 10)

            d_rot = action.shape[-1] - 4
            pos = action[..., :3]
            rot = action[..., 3 : 3 + d_rot]
            gripper = action[..., [-1]]
            rot = self.rotation_tf.inverse(rot)
            action = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)

            if is_dual_arm:
                action = action.reshape(*leading_dim, 14)
        return action


class RobomimicDataset(torch.utils.data.Dataset):
    """Robomimic imitation learning dataset.

    The dataset chunks the demonstrations into sequences of length `To + Ta - 1`.
    It uses `MinMaxNormalizer` to normalize the lowdim observations and actions to [-1, 1] as default.
    Each batch contains:
    - batch['action'], action of shape (batch_size, Ta, act_dim)
    - batch['lowdim'], lowdim observation of shape (batch_size, To, lowdim_dim)
    - batch['image'], image observation of shape (batch_size, To, n_views, 3, 84, 84) if image observation is available

    Below is the structure of each component:
        - action: (7 or 10,) ((14 or 20,) for dual arm)
            - actions: (7 or 10,) ((14 or 20,) for dual arm)
        - lowdim: (9 + n,) ((18 + n,) for dual arm)
            - object: (n,)
            - robot0_eef_pos: (3,)
            - robot0_eef_quat: (4,)
            - robot0_gripper_qpos: (2,)
            - robot1_eef_pos: (3,) (Only for dual arm)
            - robot1_eef_quat: (4,) (Only for dual arm)
            - robot1_gripper_qpos: (2,) (Only for dual arm)
        - image: (2, 3, 84, 84) ((3, 3, 84, 84) for dual arm)
            - agentview_image: (3, 84, 84)
            - robot0_eye_in_hand_image: (3, 84, 84)
            - robot1_eye_in_hand_image: (3, 84, 84) (Only for dual arm)

    Args:
        dataset_dir (str):
            Path to the dataset directory. Please download from
            https://diffusion-policy.cs.columbia.edu/data/training/robomimic_image.zip or
            https://diffusion-policy.cs.columbia.edu/data/training/robomimic_lowdim.zip
        To (int):
            The length of the observation sequence.
        Ta (int):
            The length of the action sequence.
        abs_action (bool):
            Whether to use absolute action (Position control).

    Examples:
        >>> dataset = RobomimicDataset(
                dataset_dir='dev/robomimic/datasets/square/mh/image_abs.hdf5', To=2, Ta=16, abs_action=True
            )
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> batch = next(iter(dataloader))
        >>> image = batch["image"]  # (32, 2, 2, 3, 84, 84) (b, To, n_views, 3, 84, 84) uint8
        >>> lowdim = batch["lowdim"]  # (32, 2, 9) (b, To, lowdim_dim)
        >>> action = batch["action"]  # (32, 16, 10) (b, Ta, act_dim)
    """

    def __init__(
        self,
        dataset_dir: str = "dev/robomimic/datasets/square/mh/image_abs.hdf5",
        To: int = 2,
        Ta: int = 16,
        abs_action: bool = False,
    ):
        super().__init__()
        third_person_camera_name = "agentview_image"
        if "transport" in str(dataset_dir):
            third_person_camera_name = "shouldercamera0_image"
        elif "tool_hang" in str(dataset_dir):
            third_person_camera_name = "sideview_image"

        dataset_dir = str(dataset_dir)
        self.action_converter = ActionConverter(abs_action=abs_action)
        self.To, self.Ta = To, Ta

        is_dual_arm = "transport" in dataset_dir
        is_image_obs = "image" in dataset_dir

        meta = {"episode_ends": list()}
        with h5py.File(dataset_dir, "r") as f:
            # Get episode ends
            demo_list = list(f["data"].keys())
            for demo in demo_list:
                meta["episode_ends"].append(f[f"data/{demo}/actions"].shape[0])
            meta["episode_ends"] = np.cumsum(meta["episode_ends"])
            size = meta["episode_ends"][-1]

            action_dim = f[f"data/{demo}/actions"].shape[-1]
            lowdim_dim = (
                f[f"data/{demo}/obs/object"].shape[-1]
                + f[f"data/{demo}/obs/robot0_eef_pos"].shape[-1]
                + f[f"data/{demo}/obs/robot0_eef_quat"].shape[-1]
                + f[f"data/{demo}/obs/robot0_gripper_qpos"].shape[-1]
            )
            if is_dual_arm:
                lowdim_dim += (
                    f[f"data/{demo}/obs/robot1_eef_pos"].shape[-1]
                    + f[f"data/{demo}/obs/robot1_eef_quat"].shape[-1]
                    + f[f"data/{demo}/obs/robot1_gripper_qpos"].shape[-1]
                )

            # Create empty dataset
            data = {
                "action": np.empty((size, action_dim), dtype=np.float32),
                "lowdim": np.empty((size, lowdim_dim), dtype=np.float32),
                "image": np.empty((size, 3 if is_dual_arm else 2, 3, 84, 84), dtype=np.uint8)
                if is_image_obs
                else None,
            }

            # Load dataset
            ptr = 0
            for demo in tqdm.tqdm(demo_list):
                seq_len = f[f"data/{demo}/actions"].shape[0]

                data["action"][ptr : ptr + seq_len] = f[f"data/{demo}/actions"][:]

                lowdim_list = [
                    f[f"data/{demo}/obs/object"][:],
                    f[f"data/{demo}/obs/robot0_eef_pos"][:],
                    f[f"data/{demo}/obs/robot0_eef_quat"][:],
                    f[f"data/{demo}/obs/robot0_gripper_qpos"][:],
                ]
                if is_dual_arm:
                    lowdim_list.extend(
                        [
                            f[f"data/{demo}/obs/robot1_eef_pos"][:],
                            f[f"data/{demo}/obs/robot1_eef_quat"][:],
                            f[f"data/{demo}/obs/robot1_gripper_qpos"][:],
                        ]
                    )
                data["lowdim"][ptr : ptr + seq_len] = np.concatenate(lowdim_list, axis=-1)

                if is_image_obs:
                    image_list = [
                        f[f"data/{demo}/obs/{third_person_camera_name}"][:],
                        f[f"data/{demo}/obs/robot0_eye_in_hand_image"][:],
                    ]
                    if is_dual_arm:
                        image_list.extend([f[f"data/{demo}/obs/robot1_eye_in_hand_image"][:]])

                    data["image"][ptr : ptr + seq_len] = np.stack(image_list, axis=1).transpose(
                        0, 1, 4, 2, 3
                    )

                ptr += seq_len

            data["action"] = self.action_converter.transform(data["action"])
            action_dim = data["action"].shape[-1]

        self.normalizers = {
            "lowdim": MinMaxNormalizer(data["lowdim"]),
            "action": MinMaxNormalizer(data["action"]),
        }
        data["lowdim"] = self.normalizers["lowdim"].normalize(data["lowdim"])
        data["action"] = self.normalizers["action"].normalize(data["action"])

        self.data = data
        self.meta = meta

        self.action_dim = action_dim
        self.lowdim_dim = lowdim_dim
        self.image_dim = data["image"].shape[1:] if data["image"] is not None else None

        self.indices = create_indices(
            episode_ends=meta["episode_ends"],
            sequence_length=Ta + To - 1,
            pad_before=To - 1,
            pad_after=Ta - 1,
        )

    def get_normalizer(self):
        return self.normalizers

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx, end_idx) = (
            self.indices[idx]
        )

        e_Ta = self.Ta - (self.To + self.Ta - 1 - sample_end_idx)
        action = self.data["action"][buffer_end_idx - e_Ta : buffer_end_idx]
        if self.To + self.Ta - 1 > sample_end_idx:
            action = np.pad(
                action, ((0, self.To + self.Ta - 1 - sample_end_idx), (0, 0)), mode="edge"
            )
        assert action.shape[0] == self.Ta, f"{action.shape[0]} != {self.Ta}"

        lowdim = self.data["lowdim"][
            buffer_start_idx : buffer_start_idx + self.To - sample_start_idx
        ]
        if sample_start_idx > 0:
            lowdim = np.pad(lowdim, ((sample_start_idx, 0), (0, 0)), mode="edge")

        if self.data["image"] is not None:
            image = self.data["image"][
                buffer_start_idx : buffer_start_idx + self.To - sample_start_idx
            ]
            if sample_start_idx > 0:
                image = np.pad(
                    image, ((sample_start_idx, 0), (0, 0), (0, 0), (0, 0), (0, 0)), mode="edge"
                )
        else:
            image = None

        return {
            "action": action,
            "lowdim": lowdim,
            "image": image,
        }


# class RobomimicDataset(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         dataset_dir: str = "dev/robomimic/datasets/square/mh/image_abs.hdf5",
#         meta: str = {
#             "robot0_eye_in_hand_image": "first_person_RGB",
#             "agentview_image": "third_person_RGB",
#             "robot0_eef_pos": "lowdim",
#             "robot0_eef_quat": "lowdim",
#             "robot0_gripper_qpos": "lowdim",
#             "actions": "action",
#         },
#         To: int = 2,
#         Ta: int = 12,
#         abs_action: bool = True,
#     ):
#         self.To, self.Ta = To, Ta
#         self.action_converter = ActionConverter(abs_action=abs_action)

#         dataset = {"data": {"action": list()}, "meta": {"episode_ends": list()}}
#         for k, v in meta.items():
#             if "RGB" in v:
#                 dataset["data"][v] = list()
#             elif v == "lowdim" and "lowdim" not in dataset["data"]:
#                 dataset["data"]["lowdim"] = list()

#         with h5py.File(dataset_dir, "r") as f:
#             demos = f["data"]
#             for i in range(len(demos)):
#                 demo = demos[f"demo_{i}"]

#                 lowdim_values = []
#                 for k, v in meta.items():
#                     if "RGB" in v:
#                         dataset["data"][v].append(demo["obs"][k][:].astype(np.uint8))
#                     elif v == "lowdim":
#                         lowdim_values.append(demo["obs"][k][:].astype(np.float32))
#                 dataset["data"]["action"].append(demo["actions"][:].astype(np.float32))
#                 dataset["meta"]["episode_ends"].append(demo["actions"].shape[0])
#                 dataset["data"]["lowdim"].append(np.concatenate(lowdim_values, axis=-1))

#         for k in dataset["data"].keys():
#             if k == "action":
#                 dataset["data"]["action"] = self.action_converter.transform(
#                     np.concatenate(dataset["data"]["action"], axis=0)
#                 )
#             else:
#                 dataset["data"][k] = np.concatenate(dataset["data"][k], axis=0)
#         dataset["meta"]["episode_ends"] = np.cumsum(dataset["meta"]["episode_ends"])

#         self.normalizers = {
#             "lowdim": MinMaxNormalizer(dataset["data"]["lowdim"]),
#             "action": MinMaxNormalizer(dataset["data"]["action"]),
#         }
#         dataset["data"]["lowdim"] = self.normalizers["lowdim"].normalize(dataset["data"]["lowdim"])
#         dataset["data"]["action"] = self.normalizers["action"].normalize(dataset["data"]["action"])

#         self.info = {}
#         for k in dataset["data"].keys():
#             self.info[k] = {
#                 "shape": dataset["data"][k].shape,
#                 "dtype": dataset["data"][k].dtype,
#             }

#         self.replay_buffer = ReplayBuffer(dataset)
#         self.sampler = SequenceSampler(
#             self.replay_buffer,
#             sequence_length=To + Ta - 1,
#             pad_before=To - 1,
#             pad_after=Ta - 1,
#         )

#     def get_normalizers(self):
#         return self.normalizers

#     def __str__(self):
#         return f"RobomimicDataset: {self.info}"

#     def __len__(self):
#         return len(self.sampler)

#     def __getitem__(self, idx):
#         batch = self.sampler.sample_sequence(idx)
#         for k in batch.keys():
#             if k != "action":
#                 batch[k] = batch[k][: self.To]
#         batch["action"] = batch["action"][self.To - 1 :]
#         return batch


if __name__ == "__main__":
    Ta = 8
    To = 2
    dataset = RobomimicDataset(
        "/home/dzb/CleanDiffuser/dev/robomimic/datasets/square/mh/image_abs.hdf5",
        To=To,
        Ta=Ta,
        abs_action=True,
    )
