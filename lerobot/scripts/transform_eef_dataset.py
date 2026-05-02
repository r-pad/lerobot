"""
Post-process a LeRobot dataset by transforming right_eef_pose fields:
  1. Apply -45° clockwise rotation around world Z to the rot6d part.
  2. Reorder layout from [rot6d(6), trans(3), gripper(1)]
                      to [trans(3), rot6d(6), gripper(1)].

All other fields (images, joint states, task) are copied unchanged.

Usage:
    pixi run python lerobot/scripts/transform_eef_dataset.py \
        --source xiaochyVera/insert_donut_realrobot_0427_1 \
        --target xiaochyVera/insert_donut_realrobot_0427_1_z45 \
        --target_fps 15 \
        [--push_to_hub]
"""

import argparse

import numpy as np
import pytorch3d.transforms as transforms
import torch
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

AUTO_FIELDS = {"episode_index", "frame_index", "index", "task_index", "timestamp"}

EEF_KEYS = ("observation.right_eef_pose", "action.right_eef_pose")

# -45° rotation matrix around world Z (clockwise when viewed from above)
_a = np.deg2rad(-45)
R_Z45 = torch.tensor(
    [[np.cos(_a), -np.sin(_a), 0.0],
     [np.sin(_a),  np.cos(_a), 0.0],
     [0.0,         0.0,        1.0]], dtype=torch.float32
)


def transform_eef(eef: torch.Tensor) -> torch.Tensor:
    """
    Input:  [rot6d(6), trans(3), gripper(1)]  (lerobot default)
    Output: [trans(3), rot6d_new(6), gripper(1)]
    Rotation: R_z(-45°) applied to the orientation.
    """
    rot6d   = eef[0:6]
    trans   = eef[6:9]
    gripper = eef[9:10]

    R_orig  = transforms.rotation_6d_to_matrix(rot6d.unsqueeze(0)).squeeze(0)  # (3,3)
    R_new   = R_Z45 @ R_orig
    rot6d_new = transforms.matrix_to_rotation_6d(R_new.unsqueeze(0)).squeeze(0)  # (6,)

    # Binarize gripper: lerobot <0.5 → closed (1.0), ≥0.5 → open (0.0)
    gripper_bin = torch.where(gripper < 0.5, torch.ones_like(gripper), torch.zeros_like(gripper))

    return torch.cat([trans, rot6d_new, gripper_bin])  # [trans(3), rot6d(6), gripper_bin(1)]


def transform_eef_dataset(source_repo_id: str, target_repo_id: str, target_fps: int | None = None):
    print(f"Loading source dataset: {source_repo_id}")
    source_dataset = LeRobotDataset(source_repo_id, tolerance_s=0.0004)
    source_meta    = LeRobotDatasetMetadata(source_repo_id)

    source_fps = source_dataset.fps
    if target_fps is not None and target_fps != source_fps:
        if source_fps % target_fps != 0:
            print(f"Warning: {source_fps} doesn't divide evenly by {target_fps}, slight drift possible")
        subsample_factor = source_fps // target_fps
        print(f"Subsampling from {source_fps}fps to {target_fps}fps (factor: {subsample_factor})")
    else:
        target_fps = source_fps
        subsample_factor = 1

    target_dataset = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=target_fps,
        features=source_dataset.features,
    )

    for episode_idx in range(source_meta.info["total_episodes"]):
        print(f"Processing episode {episode_idx}")
        start = source_dataset.episode_data_index["from"][episode_idx].item()
        end   = source_dataset.episode_data_index["to"][episode_idx].item()

        for idx in tqdm(range(start, end, subsample_factor), desc=f"Episode {episode_idx}"):
            frame = source_dataset[idx]

            frame_data = {
                k: v for k, v in frame.items()
                if k not in AUTO_FIELDS and k in source_dataset.features
            }
            frame_data["task"] = source_meta.tasks[frame["task_index"].item()]

            # Transform EEF pose fields
            for key in EEF_KEYS:
                if key in frame_data:
                    frame_data[key] = transform_eef(frame_data[key].float())

            # Decode images back to uint8 HWC (as expected by add_frame)
            for key in list(frame_data.keys()):
                if key.startswith("observation.images.cam_azure_kinect"):
                    if key.endswith(".color") or key.endswith(".goal_gripper_proj"):
                        frame_data[key] = (frame_data[key].permute(1, 2, 0) * 255).to(torch.uint8)
                    elif key.endswith(".transformed_depth"):
                        frame_data[key] = (frame_data[key].permute(1, 2, 0) * 1000).to(torch.uint16)

            if "observation.images.cam_wrist" in frame_data:
                frame_data["observation.images.cam_wrist"] = (
                    frame_data["observation.images.cam_wrist"].permute(1, 2, 0) * 255
                ).to(torch.uint8)

            target_dataset.add_frame(frame_data)

        target_dataset.save_episode()

    print(f"Done. Target dataset has {len(target_dataset)} frames at {target_dataset.fps} fps.")
    return target_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform EEF poses in a LeRobot dataset")
    parser.add_argument("--source", type=str, required=True, help="Source dataset repo ID")
    parser.add_argument("--target", type=str, required=True, help="Target dataset repo ID")
    parser.add_argument("--target_fps", type=int, default=None, help="Subsample to this fps (default: keep source fps)")
    parser.add_argument("--push_to_hub", action="store_true", help="Push target dataset to HuggingFace Hub")
    args = parser.parse_args()

    dataset = transform_eef_dataset(args.source, args.target, args.target_fps)

    if args.push_to_hub:
        dataset.push_to_hub(repo_id=args.target)

    print("Done!")
