import random
import torch
from tqdm import tqdm
import argparse
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

AUTO_FIELDS = {"episode_index", "frame_index", "index", "task_index", "timestamp"}

_TORCH_DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "uint16": torch.uint16,
    "bool": torch.bool,
}


def union_features(datasets):
    """Return union of features dicts; raise on same-key dtype/shape conflicts."""
    merged = {}
    for ds in datasets:
        for k, spec in ds.features.items():
            if k in AUTO_FIELDS:
                continue
            if k not in merged:
                merged[k] = spec
            else:
                existing = merged[k]
                if existing.get("dtype") != spec.get("dtype") or existing.get("shape") != spec.get("shape"):
                    raise ValueError(
                        f"Feature '{k}' has conflicting specs across datasets: "
                        f"{existing} vs {spec}"
                    )
    return merged


# LeRobot image/video/pcd dtypes are stored as uint8/uint16/float32 tensors on disk
_LEROBOT_DTYPE_FALLBACK = {"video": torch.uint8, "image": torch.uint8, "pcd": torch.float32}


def default_value_for(spec):
    """Return a zero-like default tensor (or empty string) for a feature spec."""
    dtype_str = spec.get("dtype", "float32")
    if dtype_str == "string":
        return ""
    shape = spec.get("shape", ())
    torch_dtype = (
        _TORCH_DTYPE_MAP.get(dtype_str)
        or _LEROBOT_DTYPE_FALLBACK.get(dtype_str)
        or torch.float32
    )
    return torch.zeros(shape, dtype=torch_dtype)


def sample_episode_indices(meta, n):
    """Randomly sample n episode indices (without replacement) from a dataset."""
    total = meta.info["total_episodes"]
    if n is None or n >= total:
        return list(range(total))
    return sorted(random.sample(range(total), n))


def merge_datasets(dataset_repo_ids: list[str], target_repo_id: str, demo_numbers: list[int] | None = None):
    tolerance_s = 0.0004

    if demo_numbers is not None and len(demo_numbers) != len(dataset_repo_ids):
        raise ValueError("--demo_numbers must have the same length as --datasets")

    # Load all datasets
    datasets, metas = [], []
    for i, repo_id in enumerate(dataset_repo_ids):
        print(f"Loading Dataset {i}: {repo_id}")
        datasets.append(LeRobotDataset(repo_id, tolerance_s=tolerance_s))
        metas.append(LeRobotDatasetMetadata(repo_id))

    # Validate fps consistency
    fps = datasets[0].fps
    for i, ds in enumerate(datasets[1:], 1):
        assert ds.fps == fps, f"Dataset {i} fps {ds.fps} != reference fps {fps}"

    # Print per-dataset features (mirroring upgrade_dataset style)
    for i, ds in enumerate(datasets):
        ds_keys = [k for k in ds.features if k not in AUTO_FIELDS]
        print(f"Dataset {i} ({dataset_repo_ids[i]}) features ({len(ds_keys)}): {ds_keys}")

    # Build union feature set
    all_features = union_features(datasets)

    # Show which features are unique to each dataset vs shared
    all_keys = [set(k for k in ds.features if k not in AUTO_FIELDS) for ds in datasets]
    for i, keys in enumerate(all_keys):
        unique = keys - set().union(*(all_keys[:i] + all_keys[i+1:]))
        if unique:
            print(f"Features unique to Dataset {i}: {sorted(unique)}")
    shared = set.intersection(*all_keys) if all_keys else set()
    if shared:
        print(f"Features shared across all datasets: {sorted(shared)}")

    print(f"Total user features ({len(all_features)}) — AUTO_FIELDS excluded, LeRobot re-adds them: {list(all_features.keys())}")

    print(f"Creating merged dataset: {target_repo_id}")
    merged_dataset = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=fps,
        features=all_features,
    )

    def copy_in_dataset(source_dataset, source_meta, episode_indices, label):
        source_keys = {k for k in source_dataset.features if k not in AUTO_FIELDS}
        for ep_idx in episode_indices:
            print(f"[{label}] Copying episode {ep_idx}")
            start = source_dataset.episode_data_index["from"][ep_idx].item()
            end = source_dataset.episode_data_index["to"][ep_idx].item()
            for idx in tqdm(range(start, end)):
                frame = source_dataset[idx]
                task_str = source_meta.tasks[frame["task_index"].item()]

                frame_data = {"task": task_str}

                for k, spec in all_features.items():
                    if k in AUTO_FIELDS:
                        continue
                    if k in source_keys and k in frame:
                        frame_data[k] = frame[k]
                    else:
                        frame_data[k] = default_value_for(spec)

                # Image dtype conversions
                for key in list(frame_data.keys()):
                    if key.startswith("observation.images.") and ".color" in key:
                        v = frame_data[key]
                        if v.dtype != torch.uint8:
                            frame_data[key] = (v.permute(1, 2, 0) * 255).to(torch.uint8)
                    elif key.startswith("observation.images.") and ".transformed_depth" in key:
                        v = frame_data[key]
                        if v.dtype != torch.uint16:
                            frame_data[key] = (v.permute(1, 2, 0) * 1000).to(torch.uint16)
                    elif key.startswith("observation.images.") and ".goal_gripper_proj" in key:
                        v = frame_data[key]
                        if v.dtype != torch.uint8:
                            frame_data[key] = (v.permute(1, 2, 0) * 255).to(torch.uint8)

                if "next_event_idx" in frame_data and isinstance(frame_data["next_event_idx"], torch.Tensor):
                    frame_data["next_event_idx"] = frame_data["next_event_idx"].int().unsqueeze(0)

                merged_dataset.add_frame(frame_data)
            merged_dataset.save_episode()

    for i, (ds, meta) in enumerate(zip(datasets, metas)):
        n = demo_numbers[i] if demo_numbers is not None else None
        chosen = sample_episode_indices(meta, n)
        print(f"Dataset {i}: sampling {len(chosen)}/{meta.info['total_episodes']} episodes")
        copy_in_dataset(ds, meta, chosen, f"Dataset {i}")

    print("Merge complete!")
    print(f"Target dataset user features ({len(merged_dataset.features)}): {list(merged_dataset.features.keys())}")
    print(f"  (AUTO_FIELDS {sorted(AUTO_FIELDS)} are managed by LeRobot and will appear in the saved dataset)")
    return merged_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple LeRobot datasets.")
    parser.add_argument("--datasets", type=str, nargs="+", required=True,
                        help="List of dataset repo IDs to merge")
    parser.add_argument("--target_repo_id", type=str, required=True,
                        help="Merged output dataset repo ID")
    parser.add_argument("--demo_numbers", type=int, nargs="+", default=None,
                        help="Number of episodes to randomly sample from each dataset "
                             "(must match --datasets length; omit to use all episodes)")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push merged dataset to HuggingFace Hub")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for episode sampling (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)

    merged_dataset = merge_datasets(args.datasets, args.target_repo_id, args.demo_numbers)

    if args.push_to_hub:
        merged_dataset.push_to_hub(repo_id=args.target_repo_id)

    print("Merged dataset saved!")
    print(f"New dataset features: {list(merged_dataset.features.keys())}")
