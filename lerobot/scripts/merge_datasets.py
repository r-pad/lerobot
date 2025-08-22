from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import torch
from tqdm import tqdm
import argparse

AUTO_FIELDS = {"episode_index", "frame_index", "index", "task_index", "timestamp"}

def merge_datasets(dataset_repo_ids: list[str], target_repo_id: str):
    tolerance_s = 0.0004
    
    # Load all datasets
    datasets = []
    metas = []
    
    for i, repo_id in enumerate(dataset_repo_ids):
        print(f"Loading Dataset {i}: {repo_id}")
        dataset = LeRobotDataset(repo_id, tolerance_s=tolerance_s)
        meta = LeRobotDatasetMetadata(repo_id)
        datasets.append(dataset)
        metas.append(meta)
    
    # Validate all datasets are compatible
    reference_dataset = datasets[0]
    for i, dataset in enumerate(datasets[1:], 1):
        assert dataset.features == reference_dataset.features, f"Dataset {i} features don't match reference"
        assert dataset.fps == reference_dataset.fps, f"Dataset {i} fps doesn't match reference"

    print(f"Creating merged dataset: {target_repo_id}")
    merged_dataset = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=reference_dataset.fps,
        features=reference_dataset.features,
    )

    def copy_in_dataset(source_dataset, source_meta, label):
        for episode_idx in range(source_meta.info["total_episodes"]):
            print(f"[{label}] Copying episode {episode_idx}")
            start = source_dataset.episode_data_index["from"][episode_idx].item()
            end = source_dataset.episode_data_index["to"][episode_idx].item()
            for idx in tqdm(range(start, end)):
                frame = source_dataset[idx]
                frame_data = {
                    k: v for k, v in frame.items()
                    if k not in AUTO_FIELDS and k in source_dataset.features
                }
                frame_data["task"] = source_meta.tasks[frame["task_index"].item()]
                frame_data["observation.images.cam_azure_kinect.color"] = (frame_data["observation.images.cam_azure_kinect.color"].permute(1,2,0) * 255).to(torch.uint8)
                frame_data["observation.images.cam_azure_kinect.transformed_depth"] = (frame_data["observation.images.cam_azure_kinect.transformed_depth"].permute(1,2,0) * 1000).to(torch.uint16)
                if "observation.images.cam_azure_kinect.goal_gripper_proj" in frame_data:
                    frame_data["observation.images.cam_azure_kinect.goal_gripper_proj"] = (frame_data["observation.images.cam_azure_kinect.goal_gripper_proj"].permute(1,2,0) * 255).to(torch.uint8)
                if "next_event_idx" in frame_data:
                    frame_data["next_event_idx"] = frame_data["next_event_idx"].int().unsqueeze(0)
                merged_dataset.add_frame(frame_data)
            merged_dataset.save_episode()

    for i, (dataset, meta) in enumerate(zip(datasets, metas)):
        copy_in_dataset(dataset, meta, f"Dataset {i}")

    print("Merge complete!")
    return merged_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple LeRobot datasets.")
    parser.add_argument("--datasets", type=str, nargs="+", required=True, help="List of dataset repo IDs to merge")
    parser.add_argument("--target_repo_id", type=str, required=True, help="Merged output dataset repo ID")
    parser.add_argument("--push_to_hub", action="store_true", help="Push merged dataset to HuggingFace Hub")
    args = parser.parse_args()

    merged_dataset = merge_datasets(args.datasets, args.target_repo_id)

    if args.push_to_hub:
        merged_dataset.push_to_hub(repo_id=args.target_repo_id)

    print("Merged dataset saved!")
