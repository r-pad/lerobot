from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import torch
from tqdm import tqdm
import argparse

AUTO_FIELDS = {"episode_index", "frame_index", "index", "task_index", "timestamp"}

def subsample_dataset(source_repo_id: str, target_repo_id: str, target_fps: int = 15):
    tolerance_s = 0.0004

    print(f"Loading source dataset: {source_repo_id}")
    source_dataset = LeRobotDataset(source_repo_id, tolerance_s=tolerance_s)
    source_meta = LeRobotDatasetMetadata(source_repo_id)

    # Calculate subsampling factor
    source_fps = source_dataset.fps
    if source_fps % target_fps != 0:
        print(f"Warning: {source_fps} doesn't divide evenly by {target_fps}, might get slight drift")
    subsample_factor = source_fps // target_fps  # 60 // 15 = 4

    print(f"Subsampling from {source_fps}fps to {target_fps}fps (factor: {subsample_factor})")

    # Create new dataset with target fps
    subsampled_dataset = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=target_fps,  # Key change
        features=source_dataset.features,
    )

    for episode_idx in range(source_meta.info["total_episodes"]):
        print(f"Processing episode {episode_idx}")
        start = source_dataset.episode_data_index["from"][episode_idx].item()
        end = source_dataset.episode_data_index["to"][episode_idx].item()

        # Subsample by taking every nth frame
        indices = range(start, end, subsample_factor)

        for idx in tqdm(indices, desc=f"Episode {episode_idx}"):
            frame = source_dataset[idx]

            frame_data = {
                k: v for k, v in frame.items()
                if k not in AUTO_FIELDS and k in source_dataset.features
            }

            frame_data["task"] = source_meta.tasks[frame["task_index"].item()]

            # Handle all Azure Kinect cameras (with cam_azure_kinect prefix)
            for key in list(frame_data.keys()):
                if key.startswith("observation.images.cam_azure_kinect"):
                    if key.endswith(".color"):
                        frame_data[key] = (
                            frame_data[key].permute(1,2,0) * 255
                        ).to(torch.uint8)
                    elif key.endswith(".transformed_depth"):
                        frame_data[key] = (
                            frame_data[key].permute(1,2,0) * 1000
                        ).to(torch.uint16)
                    elif key.endswith(".goal_gripper_proj"):
                        frame_data[key] = (
                            frame_data[key].permute(1,2,0) * 255
                        ).to(torch.uint8)

            # Handle wrist camera if present
            if "observation.images.cam_wrist" in frame_data:
                frame_data["observation.images.cam_wrist"] = (
                    frame_data["observation.images.cam_wrist"].permute(1,2,0) * 255
                ).to(torch.uint8)
            subsampled_dataset.add_frame(frame_data)

        subsampled_dataset.save_episode()

    print(f"Subsampling complete! New dataset has {len(subsampled_dataset)} frames at {target_fps}fps")
    return subsampled_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample a LeRobot dataset to lower fps")
    parser.add_argument("--source", type=str, required=True, help="Source dataset repo ID")
    parser.add_argument("--target", type=str, required=True, help="Target dataset repo ID")
    parser.add_argument("--target_fps", type=int, default=15, help="Target FPS (default: 15)")
    parser.add_argument("--push_to_hub", action="store_true", help="Push to HuggingFace Hub")

    args = parser.parse_args()

    subsampled_dataset = subsample_dataset(args.source, args.target, args.target_fps)

    if args.push_to_hub:
        subsampled_dataset.push_to_hub(repo_id=args.target)

    print("Done!")
