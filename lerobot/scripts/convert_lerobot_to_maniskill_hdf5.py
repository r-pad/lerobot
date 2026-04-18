#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import pytorch3d.transforms as transforms
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_ACTION_KEY = "action"
DEFAULT_STATE_KEY = "observation.state"
DEFAULT_POINTCLOUD_KEY = "observation.points.point_cloud"
DEFAULT_HAND_PCD_KEY = "observation.points.gripper_pcds"
DEFAULT_GOAL_PCD_KEY = "observation.points.goal_gripper_pcds"
DEFAULT_EEF_OBS_KEY = "observation.right_eef_pose"
DEFAULT_EEF_ACTION_KEY = "action.right_eef_pose"
DEFAULT_POINTCLOUD_SCENE_POINTS = 4000
DEFAULT_POINTCLOUD_HAND_POINTS = 500


def _resolve_key(requested_key: str, fallback_keys: list[str], available_keys: set[str], label: str) -> str:
    candidates = [requested_key] + fallback_keys
    for key in candidates:
        if key in available_keys:
            return key
    raise KeyError(
        f"Could not find {label} key. Checked {candidates}. "
        f"Available keys: {sorted(available_keys)}"
    )


def _stack_column(hf_dataset, key: str, start: int, end: int) -> np.ndarray:
    # Select a contiguous episode slice, then stack torch tensors into one numpy array.
    values = hf_dataset.select(range(start, end))[key]
    if len(values) == 0:
        raise ValueError(f"Empty slice for key='{key}' in range [{start}, {end}).")

    first_value = values[0]
    if isinstance(first_value, torch.Tensor):
        return torch.stack(values, dim=0).cpu().numpy()

    return np.stack(values, axis=0)


def _to_right_eef_relative_per_timestep(obs_eef: np.ndarray, act_eef: np.ndarray) -> np.ndarray:
    """Convert absolute EEF action to right_eef_relative using per-timestep observation.

    Expected EEF layout: [rot6d, xyz, tail], where tail is gripper (1) or hand joints (16).
    """
    if obs_eef.shape != act_eef.shape:
        raise ValueError(f"Shape mismatch for relative conversion: {obs_eef.shape=} vs {act_eef.shape=}")
    if obs_eef.ndim != 2 or obs_eef.shape[1] < 9:
        raise ValueError(f"Expected EEF arrays of shape (T, D>=9). Got {obs_eef.shape=}")

    obs = torch.from_numpy(obs_eef.astype(np.float32))
    act = torch.from_numpy(act_eef.astype(np.float32))

    obs_rot6d, obs_pos, obs_tail = obs[:, :6], obs[:, 6:9], obs[:, 9:]
    act_rot6d, act_pos, act_tail = act[:, :6], act[:, 6:9], act[:, 9:]

    r_obs = transforms.rotation_6d_to_matrix(obs_rot6d)
    r_act = transforms.rotation_6d_to_matrix(act_rot6d)
    r_relative = torch.matmul(r_obs.transpose(-2, -1), r_act)
    relative_rot6d = transforms.matrix_to_rotation_6d(r_relative)

    relative_pos = act_pos - obs_pos
    relative_tail = act_tail - obs_tail
    relative = torch.cat([relative_rot6d, relative_pos, relative_tail], dim=-1)
    return relative.numpy().astype(np.float32)


def _ensure_time_first(array: np.ndarray, target_t: int, name: str) -> np.ndarray:
    """Ensure the first dimension is time (T)."""
    if array.ndim == 0:
        raise ValueError(f"{name} must have at least 1 dimension. Got scalar.")

    if array.shape[0] == target_t:
        return array

    # Some datasets store point clouds as (N, T, C); transpose to (T, N, C).
    if array.ndim >= 2 and array.shape[1] == target_t:
        axes = list(range(array.ndim))
        axes[0], axes[1] = axes[1], axes[0]
        return np.transpose(array, axes=axes)

    raise ValueError(
        f"Could not align time dimension for {name}. "
        f"Expected time={target_t} on axis 0 (or axis 1 for transposed format), got shape={array.shape}."
    )


def _align_episode_modalities(
    actions: np.ndarray,
    state: np.ndarray,
    pointcloud: np.ndarray,
    hand_pcd: np.ndarray,
    goal_pcd: np.ndarray,
    source_ep_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align all episode modalities to the same timestep count."""
    target_t = actions.shape[0]

    state = _ensure_time_first(state, target_t, f"state(ep={source_ep_idx})")
    pointcloud = _ensure_time_first(pointcloud, target_t, f"pointcloud(ep={source_ep_idx})")
    hand_pcd = _ensure_time_first(hand_pcd, target_t, f"hand_pcd(ep={source_ep_idx})")
    goal_pcd = _ensure_time_first(goal_pcd, target_t, f"goal_pcd(ep={source_ep_idx})")

    lengths = [actions.shape[0], state.shape[0], pointcloud.shape[0], hand_pcd.shape[0], goal_pcd.shape[0]]
    min_t = min(lengths)
    max_t = max(lengths)

    if min_t != max_t:
        print(
            "[WARN] Episode length mismatch after axis normalization "
            f"(ep={source_ep_idx}): action={actions.shape[0]}, state={state.shape[0]}, "
            f"pointcloud={pointcloud.shape[0]}, hand={hand_pcd.shape[0]}, goal={goal_pcd.shape[0]}. "
            f"Truncating all to min length {min_t}."
        )
        actions = actions[:min_t]
        state = state[:min_t]
        pointcloud = pointcloud[:min_t]
        hand_pcd = hand_pcd[:min_t]
        goal_pcd = goal_pcd[:min_t]

    return actions, state, pointcloud, hand_pcd, goal_pcd


def _encode_pointcloud_with_embeddings(pointcloud: np.ndarray) -> np.ndarray:
    """Convert pointcloud to shape (T, 4500, 6) with category embeddings.

    First 4000 points get one-hot [1, 0, 0] and the remaining 500 points get [0, 1, 0].
    The original XYZ coordinates are preserved in the first 3 channels.
    """
    if pointcloud.ndim != 3:
        raise ValueError(f"Expected pointcloud with shape (T, N, 3). Got {pointcloud.shape}.")
    if pointcloud.shape[-1] != 3:
        raise ValueError(f"Expected pointcloud XYZ coordinates in the last dimension. Got {pointcloud.shape}.")

    expected_points = DEFAULT_POINTCLOUD_SCENE_POINTS + DEFAULT_POINTCLOUD_HAND_POINTS
    if pointcloud.shape[1] != expected_points:
        raise ValueError(
            f"Expected {expected_points} points per timestep for pointcloud encoding, got {pointcloud.shape[1]}."
        )

    pointcloud = pointcloud.astype(np.float32)
    batch_size = pointcloud.shape[0]
    scene_embed = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    hand_embed = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    embeddings = np.zeros((batch_size, expected_points, 3), dtype=np.float32)
    embeddings[:, :DEFAULT_POINTCLOUD_SCENE_POINTS, :] = scene_embed
    embeddings[:, DEFAULT_POINTCLOUD_SCENE_POINTS:, :] = hand_embed
    return np.concatenate([pointcloud, embeddings], axis=-1)


def convert_dataset(
    source_repo_id: str,
    output_hdf5: Path,
    root: str | None,
    episodes: list[int] | None,
    hand_pcd_mode: str,
    action_representation: str,
    action_key: str,
    eef_obs_key: str,
    eef_action_key: str,
    state_key: str,
    pointcloud_key: str,
    hand_pcd_key: str,
    goal_pcd_key: str,
    compression: str | None,
) -> None:
    source_dataset = LeRobotDataset(
        repo_id=source_repo_id,
        root=root,
        episodes=episodes,
        download_videos=False,
    )

    if source_dataset.episodes is not None:
        selected_episodes = source_dataset.episodes
    else:
        selected_episodes = list(range(source_dataset.meta.total_episodes))

    hf_keys = set(source_dataset.hf_dataset.features.keys())

    action_key = _resolve_key(action_key, [], hf_keys, "action")
    state_key = _resolve_key(state_key, ["agent_pos"], hf_keys, "state")
    pointcloud_key = _resolve_key(pointcloud_key, ["point_cloud"], hf_keys, "point cloud")
    hand_pcd_key = _resolve_key(hand_pcd_key, ["imagin_robot"], hf_keys, "hand point cloud")
    goal_pcd_key = _resolve_key(goal_pcd_key, ["goal_gripper_pcd"], hf_keys, "goal gripper point cloud")

    resolved_eef_obs_key = None
    resolved_eef_action_key = None
    if action_representation == "right_eef_relative":
        resolved_eef_obs_key = _resolve_key(eef_obs_key, [], hf_keys, "EEF observation")
        resolved_eef_action_key = _resolve_key(eef_action_key, [], hf_keys, "EEF action")

    hand_hdf5_key = "hand_pcd" if hand_pcd_mode == "sparse" else "hand_pcd_dense"
    goal_hdf5_key = "goal_gripper_pcd" if hand_pcd_mode == "sparse" else "goal_gripper_pcd_dense"

    output_hdf5.parent.mkdir(parents=True, exist_ok=True)

    if output_hdf5.exists():
        output_hdf5.unlink()

    print(f"Source repo: {source_repo_id}")
    print(f"Selected episodes: {len(selected_episodes)}")
    print(f"Output file: {output_hdf5}")
    print("Key mapping:")
    if action_representation == "right_eef_relative":
        print(f"  action: relative({resolved_eef_action_key} - {resolved_eef_obs_key}) -> actions")
    else:
        print(f"  action: {action_key} -> actions")
    print(f"  state: {state_key} -> obs/state")
    print(f"  point_cloud: {pointcloud_key} -> obs/pointcloud")
    print(f"  hand: {hand_pcd_key} -> obs/{hand_hdf5_key}")
    print(f"  goal: {goal_pcd_key} -> obs/{goal_hdf5_key}")

    with h5py.File(output_hdf5, "w") as h5f:
        data_group = h5f.create_group("data")
        meta_group = h5f.create_group("meta")

        episode_ends: list[int] = []
        source_episode_indices: list[int] = []
        shape_meta: dict[str, dict[str, dict[str, object]]] | None = None
        cumulative_steps = 0

        for local_ep_idx, source_ep_idx in tqdm(
            list(enumerate(selected_episodes)),
            desc="Converting episodes",
            total=len(selected_episodes),
        ):
            start = int(source_dataset.episode_data_index["from"][local_ep_idx].item())
            end = int(source_dataset.episode_data_index["to"][local_ep_idx].item())

            if action_representation == "right_eef_relative":
                obs_eef = _stack_column(source_dataset.hf_dataset, resolved_eef_obs_key, start, end)
                act_eef = _stack_column(source_dataset.hf_dataset, resolved_eef_action_key, start, end)
                actions = _to_right_eef_relative_per_timestep(obs_eef, act_eef)
            else:
                actions = _stack_column(source_dataset.hf_dataset, action_key, start, end).astype(np.float32)
            state = _stack_column(source_dataset.hf_dataset, state_key, start, end).astype(np.float32)
            pointcloud = _stack_column(source_dataset.hf_dataset, pointcloud_key, start, end).astype(np.float32)
            hand_pcd = _stack_column(source_dataset.hf_dataset, hand_pcd_key, start, end).astype(np.float32)
            goal_pcd = _stack_column(source_dataset.hf_dataset, goal_pcd_key, start, end).astype(np.float32)

            actions, state, pointcloud, hand_pcd, goal_pcd = _align_episode_modalities(
                actions=actions,
                state=state,
                pointcloud=pointcloud,
                hand_pcd=hand_pcd,
                goal_pcd=goal_pcd,
                source_ep_idx=int(source_ep_idx),
            )

            if shape_meta is None:
                shape_meta = {
                    "obs": {
                        "point_cloud": {"shape": [4500, 6], "type": "point_cloud"},
                        "imagin_robot": {"shape": list(hand_pcd.shape[1:]), "type": "point_cloud"},
                        "goal_gripper_pcd": {"shape": list(goal_pcd.shape[1:]), "type": "point_cloud"},
                        "agent_pos": {"shape": list(state.shape[1:]), "type": "low_dim"},
                    },
                    "action": {"shape": list(actions.shape[1:])},
                }

            pointcloud = _encode_pointcloud_with_embeddings(pointcloud)

            demo_group = data_group.create_group(f"demo_{local_ep_idx}")
            obs_group = demo_group.create_group("obs")

            print(actions.shape, state.shape, pointcloud.shape, hand_pcd.shape, goal_pcd.shape)

            demo_group.create_dataset("actions", data=actions, compression=compression)
            obs_group.create_dataset("state", data=state, compression=compression)
            obs_group.create_dataset("pointcloud", data=pointcloud, compression=compression)
            obs_group.create_dataset(hand_hdf5_key, data=hand_pcd, compression=compression)
            obs_group.create_dataset(goal_hdf5_key, data=goal_pcd, compression=compression)

            demo_group.attrs["source_episode_index"] = source_ep_idx
            demo_group.attrs["episode_length"] = int(actions.shape[0])

            cumulative_steps += int(actions.shape[0])
            episode_ends.append(cumulative_steps)
            source_episode_indices.append(int(source_ep_idx))

        meta_group.create_dataset(
            "episode_ends",
            data=np.asarray(episode_ends, dtype=np.int64),
            compression=compression,
        )
        meta_group.attrs["source_repo_id"] = source_repo_id
        meta_group.attrs["action_representation"] = action_representation
        meta_group.attrs["hand_pcd_mode"] = hand_pcd_mode
        meta_group.attrs["source_episode_indices_json"] = json.dumps(source_episode_indices)
        if shape_meta is not None:
            meta_group.attrs["shape_meta_json"] = json.dumps(shape_meta)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a LeRobot dataset to ManiSkill-style HDF5 for diffusion_policy_3d/dataset/maniskill_dataset.py."
    )
    parser.add_argument("--source_repo_id", type=str, required=True, help="LeRobot dataset repo id.")
    parser.add_argument("--output_hdf5", type=Path, required=True, help="Output HDF5 file path.")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Optional LeRobot root directory. If omitted, defaults to HF_LEROBOT_HOME behavior.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of source episode indices to export. Default exports all episodes.",
    )
    parser.add_argument(
        "--hand_pcd_mode",
        type=str,
        choices=["sparse", "dense"],
        default="sparse",
        help="Write sparse keys (hand_pcd/goal_gripper_pcd) or dense keys (hand_pcd_dense/goal_gripper_pcd_dense).",
    )

    parser.add_argument(
        "--action_representation",
        type=str,
        choices=["right_eef_relative", "absolute"],
        default="right_eef_relative",
        help="How to write HDF5 actions. right_eef_relative matches robot_adapters relative-action training mode.",
    )

    parser.add_argument("--action_key", type=str, default=DEFAULT_ACTION_KEY)
    parser.add_argument("--eef_obs_key", type=str, default=DEFAULT_EEF_OBS_KEY)
    parser.add_argument("--eef_action_key", type=str, default=DEFAULT_EEF_ACTION_KEY)
    parser.add_argument("--state_key", type=str, default=DEFAULT_STATE_KEY)
    parser.add_argument("--pointcloud_key", type=str, default=DEFAULT_POINTCLOUD_KEY)
    parser.add_argument("--hand_pcd_key", type=str, default=DEFAULT_HAND_PCD_KEY)
    parser.add_argument("--goal_pcd_key", type=str, default=DEFAULT_GOAL_PCD_KEY)

    parser.add_argument(
        "--compression",
        type=str,
        default=None,
        choices=["gzip", "lzf"],
        help="Optional HDF5 compression.",
    )

    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    convert_dataset(
        source_repo_id=args.source_repo_id,
        output_hdf5=args.output_hdf5,
        root=args.root,
        episodes=args.episodes,
        hand_pcd_mode=args.hand_pcd_mode,
        action_representation=args.action_representation,
        action_key=args.action_key,
        eef_obs_key=args.eef_obs_key,
        eef_action_key=args.eef_action_key,
        state_key=args.state_key,
        pointcloud_key=args.pointcloud_key,
        hand_pcd_key=args.hand_pcd_key,
        goal_pcd_key=args.goal_pcd_key,
        compression=args.compression,
    )


if __name__ == "__main__":
    main()
