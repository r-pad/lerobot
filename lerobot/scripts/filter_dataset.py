"""
Copy a LeRobotDataset from source to target, optionally discarding episodes.
No feature modifications — pure copy with episode filtering.
"""
import argparse
from typing import List

import torch
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

AUTO_FIELDS = {"episode_index", "frame_index", "index", "task_index", "timestamp"}


def filter_dataset(source_repo_id: str, target_repo_id: str, discard_episodes: List[int]):
    tolerance_s = 0.0004

    print(f"Loading source dataset: {source_repo_id}")
    source_dataset = LeRobotDataset(source_repo_id, tolerance_s=tolerance_s)
    source_meta = LeRobotDatasetMetadata(source_repo_id)

    print(f"Creating target dataset: {target_repo_id}")
    target_dataset = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=source_dataset.fps,
        features=source_dataset.features,
    )

    total = source_meta.info["total_episodes"]
    print(f"Copying {total} episodes (discarding {discard_episodes})...")

    for episode_idx in range(total):
        if episode_idx in discard_episodes:
            print(f"  Skipping episode {episode_idx}")
            continue

        episode_start = source_dataset.episode_data_index["from"][episode_idx].item()
        episode_end = source_dataset.episode_data_index["to"][episode_idx].item()
        episode_length = episode_end - episode_start

        print(f"  Copying episode {episode_idx} ({episode_length} frames)")
        for idx in tqdm(range(episode_start, episode_end), leave=False):
            frame = source_dataset[idx]

            frame_data = {
                k: v for k, v in frame.items()
                if k not in AUTO_FIELDS and k in source_dataset.features
            }
            frame_data["task"] = source_meta.tasks[frame["task_index"].item()]

            # Convert video/image tensors from (C,H,W) float to (H,W,C) uint8/uint16
            for key in list(frame_data.keys()):
                if key in source_dataset.features and source_dataset.features[key]["dtype"] in ["image", "video"]:
                    if "depth" in key:
                        frame_data[key] = (frame_data[key].permute(1, 2, 0) * 1000).to(torch.uint16)
                    else:
                        frame_data[key] = (frame_data[key].permute(1, 2, 0) * 255).to(torch.uint8)

            target_dataset.add_frame(frame_data)

        target_dataset.save_episode()

    print(f"Done. Filtered dataset saved to: {target_dataset.root}")
    return target_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy a LeRobot dataset, optionally discarding episodes.")
    parser.add_argument("--source_repo_id", type=str, required=True)
    parser.add_argument("--target_repo_id", type=str, required=True)
    parser.add_argument("--discard_episodes", type=int, nargs="*", default=[],
                        help="Episode indices to discard")
    parser.add_argument("--push_to_hub", action="store_true", default=False)
    args = parser.parse_args()

    dataset = filter_dataset(
        source_repo_id=args.source_repo_id,
        target_repo_id=args.target_repo_id,
        discard_episodes=args.discard_episodes,
    )

    if args.push_to_hub:
        dataset.push_to_hub(repo_id=args.target_repo_id)
        print("Pushed to hub.")
