import argparse
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

AUTO_FIELDS = {"episode_index", "frame_index", "index", "task_index", "timestamp"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True)
    args = parser.parse_args()

    meta = LeRobotDatasetMetadata(args.repo_id)
    features = meta.features

    user_features = {k: v for k, v in features.items() if k not in AUTO_FIELDS}
    auto_features = {k: v for k, v in features.items() if k in AUTO_FIELDS}

    print(f"\nDataset: {args.repo_id}")
    print(f"Total features: {len(features)}  (user: {len(user_features)}, auto-managed: {len(auto_features)})\n")

    print(f"User features ({len(user_features)}):")
    for name, spec in user_features.items():
        print(f"  {name}")
        print(f"    dtype : {spec.get('dtype')}")
        print(f"    shape : {spec.get('shape')}")
        if spec.get('info'):
            print(f"    info  : {spec.get('info')}")

    print(f"\nAuto-managed features ({len(auto_features)}): {sorted(auto_features.keys())}")
