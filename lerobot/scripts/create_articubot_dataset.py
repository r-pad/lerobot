"""
Create a LeRobotDataset from ArticuBot h5 trajectory files.

Each h5 file is one trajectory containing:
  - obs/cam{0,1}_image: (T, 256, 256, 3) uint8  (fixed cameras)
  - obs/cam2_image: (T, 256, 256, 3) uint8  (wrist camera)
  - obs/cam{0,1,2}_depth: (T, 256, 256) uint16  (depth in mm)
  - obs/state: (T, 10) float32  (ee_pos(3) + 6D_rot(6) + gripper(1))
  - action: (T, 10) float32
  - sim_states/task_config/lang: language description

Usage:
  pixi run python lerobot/scripts/create_articubot_dataset.py \
      --data_dir /home/sriram/Desktop/ArticuBot/data/rgb/41510 \
      --repo_id sriramsk/articubot_41510
"""

import argparse
from glob import glob
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


IMG_SHAPE = (256, 256)
FPS = 30

FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (10,),
        "names": [
            "ee_pos_0", "ee_pos_1", "ee_pos_2",
            "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5",
            "gripper_angle",
        ],
    },
    "action": {
        "dtype": "float32",
        "shape": (10,),
        "names": [
            "delta_ee_pos_0", "delta_ee_pos_1", "delta_ee_pos_2",
            "delta_rot6d_0", "delta_rot6d_1", "delta_rot6d_2",
            "delta_rot6d_3", "delta_rot6d_4", "delta_rot6d_5",
            "delta_gripper",
        ],
    },
    "observation.images.cam0": {
        "dtype": "video",
        "shape": (IMG_SHAPE[0], IMG_SHAPE[1], 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.cam1": {
        "dtype": "video",
        "shape": (IMG_SHAPE[0], IMG_SHAPE[1], 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.wrist": {
        "dtype": "video",
        "shape": (IMG_SHAPE[0], IMG_SHAPE[1], 3),
        "names": ["height", "width", "channels"],
    },
}


def create_articubot_dataset(
    data_dir: str,
    repo_id: str,
    only_good: bool = True,
):
    h5_files = sorted(glob(str(Path(data_dir) / "*.h5")))
    if not h5_files:
        raise FileNotFoundError(f"No h5 files found in {data_dir}")
    print(f"Found {len(h5_files)} h5 files in {data_dir}")

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=FEATURES,
    )

    skipped = 0
    for h5_path in tqdm(h5_files, desc="Processing trajectories"):
        with h5py.File(h5_path, "r") as hf:
            actions = np.asarray(hf["action"], dtype=np.float32)
            states = np.asarray(hf["obs/state"], dtype=np.float32)
            cam0_imgs = np.asarray(hf["obs/cam0_image"])
            cam1_imgs = np.asarray(hf["obs/cam1_image"])
            wrist_imgs = np.asarray(hf["obs/cam2_image"])

            num_steps = actions.shape[0]
            task = hf["sim_states/task_config/lang"][()].decode("utf-8")

            for t in range(num_steps):
                dataset.add_frame({
                    "observation.state": states[t],
                    "action": actions[t],
                    "observation.images.cam0": cam0_imgs[t],
                    "observation.images.cam1": cam1_imgs[t],
                    "observation.images.wrist": wrist_imgs[t],
                    "task": task,
                })

            dataset.save_episode()

    print(f"Done! {len(h5_files) - skipped} trajectories saved, {skipped} skipped.")
    print(f"Dataset saved to: {dataset.root}")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create LeRobotDataset from ArticuBot h5 files.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing h5 trajectory files")
    parser.add_argument("--repo_id", type=str, default="sriramsk/articubot_41510",
                        help="Repository ID for the dataset")
    parser.add_argument("--include_bad", action="store_true",
                        help="Include trajectories marked as bad")
    args = parser.parse_args()

    create_articubot_dataset(
        data_dir=args.data_dir,
        repo_id=args.repo_id,
        only_good=not args.include_bad,
    )
