import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import open3d as o3d
import torch
import zarr
from pytorch3d.ops import sample_farthest_points
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_real_depth_map

import libero
from cleandiffuser.utils.codecs import jpeg
from cleandiffuser.utils.t5 import T5LanguageEncoder
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def cammat2o3d(cam_mat, width, height):
    cx = cam_mat[0, 2]
    fx = cam_mat[0, 0]
    cy = cam_mat[1, 2]
    fy = cam_mat[1, 1]

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


def filter(pointclouds, device="cuda"):
    max_sz = max([each.shape[0] for each in pointclouds])
    all_pc = np.zeros((len(pointclouds), max_sz, 3), dtype=np.float32)
    lengths = np.zeros(len(pointclouds), dtype=np.int64)
    for i, each in enumerate(pointclouds):
        all_pc[i, : each.shape[0]] = each
        lengths[i] = each.shape[0]
    all_pc = torch.tensor(all_pc, device=device)
    lengths = torch.tensor(lengths, device=device)
    pcd = sample_farthest_points(all_pc, lengths, K=8192)[0]
    return pcd.cpu().numpy()


# --- Config ---

argparser = argparse.ArgumentParser()
argparser.add_argument("--require_depth", type=bool, default=False)
argparser.add_argument("--require_pointcloud", type=bool, default=False)
argparser.add_argument("--device", type=str, default="cuda:0")
argparser.add_argument("--path_to_t5", type=str, default="google-t5/t5-base")
argparser.add_argument("--image_size", type=int, default=224)
argparser.add_argument("--benchmark", type=str, default="libero_spatial")

args = argparser.parse_args()

REQUIRE_DEPTH = args.require_depth
REQUIRE_POINT_CLOUD = args.require_pointcloud
DEVICE = args.device
PATH_TO_T5 = args.path_to_t5
IMAGE_SIZE = args.image_size
BOUNDING_BOX = o3d.geometry.AxisAlignedBoundingBox(
    min_bound=(-1.0, -1.0, 0.0), max_bound=(1.0, 1.0, 1.6)
)

LIBERO_PATH = Path(os.path.dirname(libero.libero.__file__)).parents[0]
DATASET_PATH = LIBERO_PATH / "datasets"
BENCHMARKS = [args.benchmark]
SAVE_DATA_PATH = Path(__file__).parents[3] / "dev/libero"

# create save directory
SAVE_DATA_PATH.mkdir(parents=True, exist_ok=True)

# benchmark for suite
benchmark_dict = benchmark.get_benchmark_dict()

# Total number of tasks
num_tasks = 0
for bm in BENCHMARKS:
    benchmark_path = DATASET_PATH / bm
    num_tasks += len(list(benchmark_path.glob("*.hdf5")))

# languege encoder
language_encoder = T5LanguageEncoder(pretrained_model_name_or_path=PATH_TO_T5, device=DEVICE)
t5_hidden_dim = language_encoder.model.config.d_model

tasks_stored = 0
for bm in BENCHMARKS:
    print(f"############################# {bm} #############################")
    benchmark_path = DATASET_PATH / bm

    # Init env benchmark suite
    task_suite = benchmark_dict[bm]()

    # Init zarr dataset
    root = zarr.open(str(SAVE_DATA_PATH / f"{bm}.zarr"), "w")
    zarr_data = root.create_group("data")
    zarr_meta = root.create_group("meta")

    zarr_data.require_dataset(
        "color",
        shape=(0, 3, IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.uint8,
        chunks=(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        compressor=jpeg((1, 3, IMAGE_SIZE, IMAGE_SIZE), quality=50),
    )
    zarr_data.require_dataset(
        "color_ego",
        shape=(0, 3, IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.uint8,
        chunks=(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        compressor=jpeg((1, 3, IMAGE_SIZE, IMAGE_SIZE), quality=50),
    )
    if REQUIRE_DEPTH:
        zarr_data.require_dataset(
            "depth",
            shape=(0, IMAGE_SIZE, IMAGE_SIZE),
            dtype=np.uint16,
            chunks=(1, IMAGE_SIZE, IMAGE_SIZE),
            compressor=jpeg((1, IMAGE_SIZE, IMAGE_SIZE), quality=50),
        )
        zarr_data.require_dataset(
            "depth_ego",
            shape=(0, IMAGE_SIZE, IMAGE_SIZE),
            dtype=np.uint16,
            chunks=(1, IMAGE_SIZE, IMAGE_SIZE),
            compressor=jpeg((1, IMAGE_SIZE, IMAGE_SIZE), quality=50),
        )
    if REQUIRE_POINT_CLOUD:
        zarr_data.require_dataset(
            "pointcloud",
            shape=(0, 8192, 3),
            dtype=np.float32,
            chunks=(1, 8192, 3),
            compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=1),
        )
        zarr_data.require_dataset(
            "pointcloud_ego",
            shape=(0, 8192, 3),
            dtype=np.float32,
            chunks=(1, 8192, 3),
            compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=1),
        )
    zarr_meta.require_dataset(
        "episode_ends",
        shape=(0,),
        dtype=np.uint32,
    )
    zarr_meta.require_dataset(
        "language_embeddings",
        shape=(0, 32, t5_hidden_dim),
        dtype=np.float32,
    )
    zarr_meta.require_dataset(
        "language_masks",
        shape=(0, 32),
        dtype=np.float32,
    )
    zarr_meta.require_dataset("task_id", shape=(0,), dtype=np.uint16)
    init_others = False

    for task_file in benchmark_path.glob("*.hdf5"):
        print(f"Processing {tasks_stored + 1}/{num_tasks}: {task_file}")
        data = h5py.File(task_file, "r")["data"]

        # Init env
        task_name = str(task_file).split("/")[-1][:-10]
        # get task id from list of task names
        task_id = task_suite.get_task_names().index(task_name)
        # create environment
        task = task_suite.get_task(task_id)
        task_name = task.name
        task_bddl_file = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": IMAGE_SIZE,
            "camera_widths": IMAGE_SIZE,
            "camera_depths": REQUIRE_DEPTH or REQUIRE_POINT_CLOUD,
        }
        env = OffScreenRenderEnv(**env_args)

        # get camera intrinsic matrix
        cam_intrinsic = get_camera_intrinsic_matrix(env.sim, "agentview", IMAGE_SIZE, IMAGE_SIZE)
        cam_ego_intrinsic = get_camera_intrinsic_matrix(
            env.sim, "robot0_eye_in_hand", IMAGE_SIZE, IMAGE_SIZE
        )
        od_cammat = cammat2o3d(cam_intrinsic, IMAGE_SIZE, IMAGE_SIZE)
        od_cammat_ego = cammat2o3d(cam_ego_intrinsic, IMAGE_SIZE, IMAGE_SIZE)

        obs = env.reset()

        states = []
        actions = []
        rewards = []
        episode_ends = []

        for demo in data.keys():
            print(f"Processing demo {demo}")
            demo_data = data[demo]

            colors, colors_ego = [], []
            depths, depths_ego = [], []
            pcds, pcds_ego = [], []
            joint_states, eef_states, gripper_states = [], [], []

            for i in range(len(demo_data["states"])):
                obs = env.regenerate_obs_from_state(demo_data["states"][i])

                # get RGBD
                color = obs["agentview_image"][::-1]
                color_ego = obs["robot0_eye_in_hand_image"][::-1]

                if REQUIRE_DEPTH or REQUIRE_POINT_CLOUD:
                    depth = obs["agentview_depth"][::-1]
                    depth_ego = obs["robot0_eye_in_hand_depth"][::-1]
                    # to metric depth
                    depth = np.clip(get_real_depth_map(env.sim, depth), 0.0, 2.0)
                    depth_ego = np.clip(get_real_depth_map(env.sim, depth_ego), 0.0, 2.0)

                if REQUIRE_POINT_CLOUD:
                    # get pcd
                    od_depth = o3d.geometry.Image(depth)
                    o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(od_depth, od_cammat)
                    o3d_cloud = o3d_cloud.crop(BOUNDING_BOX)
                    o3d_cloud = o3d_cloud.voxel_down_sample(voxel_size=0.005)

                    od_depth_ego = o3d.geometry.Image(depth_ego)
                    o3d_cloud_ego = o3d.geometry.PointCloud.create_from_depth_image(
                        od_depth_ego, od_cammat_ego
                    )
                    o3d_cloud_ego = o3d_cloud_ego.crop(BOUNDING_BOX)
                    o3d_cloud_ego = o3d_cloud_ego.voxel_down_sample(voxel_size=0.003)

                joint_state = obs["robot0_joint_pos"]
                eef_state = np.concatenate([obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
                gripper_state = obs["robot0_gripper_qpos"]

                # append
                colors.append(color)
                colors_ego.append(color_ego)

                if REQUIRE_DEPTH:
                    depths.append(depth)
                    depths_ego.append(depth_ego)
                if REQUIRE_POINT_CLOUD:
                    pcds.append(np.asarray(o3d_cloud.points))
                    pcds_ego.append(np.asarray(o3d_cloud_ego.points))

                joint_states.append(joint_state)
                eef_states.append(eef_state)
                gripper_states.append(gripper_state)

                if not init_others:
                    init_others = True
                    zarr_data.require_dataset(
                        "joint_states",
                        shape=(0, joint_state.shape[-1]),
                        dtype=np.float32,
                        chunks=(1, joint_state.shape[-1]),
                    )
                    zarr_data.require_dataset(
                        "eef_states",
                        shape=(0, eef_state.shape[-1]),
                        dtype=np.float32,
                        chunks=(1, eef_state.shape[-1]),
                    )
                    zarr_data.require_dataset(
                        "gripper_states",
                        shape=(0, gripper_state.shape[-1]),
                        dtype=np.float32,
                        chunks=(1, gripper_state.shape[-1]),
                    )
                    if bm not in ["libero_90", "libero_10"]:
                        zarr_data.require_dataset(
                            "states",
                            shape=(0, demo_data["states"].shape[-1]),
                            dtype=np.float32,
                            chunks=(1, demo_data["states"].shape[-1]),
                        )
                    zarr_data.require_dataset(
                        "actions",
                        shape=(0, demo_data["actions"].shape[-1]),
                        dtype=np.float32,
                    )
                    zarr_data.require_dataset(
                        "rewards",
                        shape=(0,),
                        dtype=np.float32,
                        chunks=(1,),
                    )

            colors = np.array(colors, dtype=np.uint8).transpose(0, 3, 1, 2)
            colors_ego = np.array(colors_ego, dtype=np.uint8).transpose(0, 3, 1, 2)

            if REQUIRE_DEPTH:
                depths = (np.array(depths) * 1000.0).astype(np.uint16)[:, :, :, 0]
                depths_ego = (np.array(depths_ego) * 1000.0).astype(np.uint16)[:, :, :, 0]
            if REQUIRE_POINT_CLOUD:
                pointcloud = filter(pcds, device=DEVICE)
                pointcloud_ego = filter(pcds_ego, device=DEVICE)

            joint_states = np.array(joint_states, dtype=np.float32)
            eef_states = np.array(eef_states, dtype=np.float32)
            gripper_states = np.array(gripper_states, dtype=np.float32)
            episode_length = int(colors.shape[0])

            with torch.no_grad():
                lang_emb, lang_mask = language_encoder([env.language_instruction])
                lang_emb = lang_emb.cpu().numpy()[0]
                lang_mask = lang_mask.cpu().numpy()[0]
                if lang_emb.shape[0] < 32:
                    lang_emb = np.pad(lang_emb, ((0, 32 - lang_emb.shape[0]), (0, 0)))
                    lang_mask = np.pad(lang_mask, (0, 32 - lang_mask.shape[0]))
                elif lang_emb.shape[0] > 32:
                    lang_emb = lang_emb[:32, :]
                    lang_mask = lang_mask[:32]

            zarr_data["color"].append(colors)
            zarr_data["color_ego"].append(colors_ego)

            if REQUIRE_DEPTH:
                zarr_data["depth"].append(depths)
                zarr_data["depth_ego"].append(depths_ego)
            if REQUIRE_POINT_CLOUD:
                zarr_data["pointcloud"].append(pointcloud)
                zarr_data["pointcloud_ego"].append(pointcloud_ego)

            zarr_data["joint_states"].append(joint_states)
            zarr_data["eef_states"].append(eef_states)
            zarr_data["gripper_states"].append(gripper_states)

            if bm not in ["libero_90", "libero_10"]:
                zarr_data["states"].append(np.array(demo_data["states"], dtype=np.float32))
            zarr_data["actions"].append(np.array(demo_data["actions"], dtype=np.float32))
            zarr_data["rewards"].append(np.array(demo_data["rewards"], dtype=np.float32))

            zarr_meta["episode_ends"].append(
                [
                    episode_length
                    + (zarr_meta["episode_ends"][-1] if len(zarr_meta["episode_ends"]) > 0 else 0)
                ]
            )
            zarr_meta["task_id"].append([int(task_id)])
            zarr_meta["language_embeddings"].append(lang_emb[None])
            zarr_meta["language_masks"].append(lang_mask[None])

        print(f"Saved to {str(SAVE_DATA_PATH)}")
        tasks_stored += 1
