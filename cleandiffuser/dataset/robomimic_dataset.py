import h5py
import numpy as np
import torch

from cleandiffuser.dataset.dataset_utils import ReplayBuffer, RotationTransformer, SequenceSampler
from cleandiffuser.utils import MinMaxNormalizer


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
            is_dual_arm = (action.shape[-1] == 20)
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
    def __init__(
        self,
        dataset_dir: str = "dev/robomimic/datasets/square/mh/image_abs.hdf5",
        meta: str = {
            "robot0_eye_in_hand_image": "first_person_RGB",
            "agentview_image": "third_person_RGB",
            "robot0_eef_pos": "lowdim",
            "robot0_eef_quat": "lowdim",
            "robot0_gripper_qpos": "lowdim",
            "actions": "action",
        },
        To: int = 2,
        Ta: int = 12,
        abs_action: bool = True,
    ):
        self.To, self.Ta = To, Ta
        self.action_converter = ActionConverter(abs_action=abs_action)

        dataset = {"data": {"action": list()}, "meta": {"episode_ends": list()}}
        for k, v in meta.items():
            if "RGB" in v:
                dataset["data"][v] = list()
            elif v == "lowdim" and "lowdim" not in dataset["data"]:
                dataset["data"]["lowdim"] = list()

        with h5py.File(dataset_dir, "r") as f:
            demos = f["data"]
            for i in range(len(demos)):
                demo = demos[f"demo_{i}"]

                lowdim_values = []
                for k, v in meta.items():
                    if "RGB" in v:
                        dataset["data"][v].append(demo["obs"][k][:].astype(np.uint8))
                    elif v == "lowdim":
                        lowdim_values.append(demo["obs"][k][:].astype(np.float32))
                dataset["data"]["action"].append(demo["actions"][:].astype(np.float32))
                dataset["meta"]["episode_ends"].append(demo["actions"].shape[0])
                dataset["data"]["lowdim"].append(np.concatenate(lowdim_values, axis=-1))

        for k in dataset["data"].keys():
            if k == "action":
                dataset["data"]["action"] = self.action_converter.transform(
                    np.concatenate(dataset["data"]["action"], axis=0)
                )
            else:
                dataset["data"][k] = np.concatenate(dataset["data"][k], axis=0)
        dataset["meta"]["episode_ends"] = np.cumsum(dataset["meta"]["episode_ends"])

        self.normalizers = {
            "lowdim": MinMaxNormalizer(dataset["data"]["lowdim"]),
            "action": MinMaxNormalizer(dataset["data"]["action"]),
        }
        dataset["data"]["lowdim"] = self.normalizers["lowdim"].normalize(dataset["data"]["lowdim"])
        dataset["data"]["action"] = self.normalizers["action"].normalize(dataset["data"]["action"])

        self.info = {}
        for k in dataset["data"].keys():
            self.info[k] = {
                "shape": dataset["data"][k].shape,
                "dtype": dataset["data"][k].dtype,
            }

        self.replay_buffer = ReplayBuffer(dataset)
        self.sampler = SequenceSampler(
            self.replay_buffer,
            sequence_length=To + Ta - 1,
            pad_before=To - 1,
            pad_after=Ta - 1,
        )

    def get_normalizers(self):
        return self.normalizers

    def __str__(self):
        return f"RobomimicDataset: {self.info}"

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        batch = self.sampler.sample_sequence(idx)
        for k in batch.keys():
            if k != "action":
                batch[k] = batch[k][: self.To]
        batch["action"] = batch["action"][self.To - 1 :]
        return batch


if __name__ == "__main__":
    dataset = RobomimicDataset(
        "dev/robomimic/datasets/square/mh/image_abs.hdf5",
        meta={
            "robot0_eef_pos": "lowdim",
            "robot0_eef_quat": "lowdim",
            "robot0_gripper_qpos": "lowdim",
            "actions": "action",
        },
    )
