from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import torch
from tqdm import tqdm
import argparse

AUTO_FIELDS = {"episode_index", "frame_index", "index", "task_index", "timestamp"}

def merge_datasets(dataset_a_repo_id: str, dataset_b_repo_id: str, target_repo_id: str):
    tolerance_s = 0.0004
    print(f"Loading Dataset A: {dataset_a_repo_id}")
    dataset_a = LeRobotDataset(dataset_a_repo_id, tolerance_s=tolerance_s)
    meta_a = LeRobotDatasetMetadata(dataset_a_repo_id)

    print(f"Loading Dataset B: {dataset_b_repo_id}")
    dataset_b = LeRobotDataset(dataset_b_repo_id, tolerance_s=tolerance_s)
    meta_b = LeRobotDatasetMetadata(dataset_b_repo_id)

    # Assumes compatible shapes/dtypes
    assert dataset_a.features == dataset_b.features
    assert dataset_a.fps == dataset_b.fps

    print(f"Creating merged dataset: {target_repo_id}")
    merged_dataset = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=dataset_a.fps,
        features=dataset_a.features,
    )

    def copy_dataset(source_dataset, source_meta, label):
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
                merged_dataset.add_frame(frame_data)
            merged_dataset.save_episode()

    copy_dataset(dataset_a, meta_a, "A")
    copy_dataset(dataset_b, meta_b, "B")

    print("Merge complete!")
    return merged_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two LeRobot datasets.")
    parser.add_argument("--dataset_a", type=str, required=True, help="First dataset repo ID")
    parser.add_argument("--dataset_b", type=str, required=True, help="Second dataset repo ID")
    parser.add_argument("--target_repo_id", type=str, required=True, help="Merged output dataset repo ID")
    parser.add_argument("--push_to_hub", action="store_true", help="Push merged dataset to HuggingFace Hub")
    args = parser.parse_args()

    merged_dataset = merge_datasets(args.dataset_a, args.dataset_b, args.target_repo_id)

    if args.push_to_hub:
        merged_dataset.push_to_hub(repo_id=args.target_repo_id)

    print("Merged dataset saved!")
