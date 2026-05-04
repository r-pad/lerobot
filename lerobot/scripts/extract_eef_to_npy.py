"""
Extract observation.right_eef_pose from a LeRobot dataset into per-episode (T, 10)
.npy files for replay with replay_eef_actions.py.

Assumes the source dataset was produced by transform_eef_dataset.py (or equivalent),
i.e. observation.right_eef_pose is already in polaris layout
    [trans(3) | rot6d(6) | gripper_binary(1)]
with gripper in polaris convention (1=closed, 0=open). No further conversion is
applied here.

Usage:
    pixi run python lerobot/scripts/extract_eef_to_npy.py \\
        --source xiaochyVera/pick_red_mug_realrobot_3_z45_reorder_binary_ss_427 \\
        --out_dir /tmp/pick_red_mug_eef \\
        [--key observation.right_eef_pose]   # default

Then replay:
    pixi run python lerobot/scripts/replay_eef_actions.py \\
        --actions_dir /tmp/pick_red_mug_eef --fps 15
"""

import argparse
import pathlib

import numpy as np
import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="LeRobot dataset repo id")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to write demo_*.npy files")
    parser.add_argument("--key", type=str, default="observation.right_eef_pose",
                        help="Observation key to extract (default: observation.right_eef_pose)")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.source}")
    ds = LeRobotDataset(args.source, tolerance_s=0.0004)
    meta = LeRobotDatasetMetadata(args.source)

    n_eps = meta.info["total_episodes"]
    print(f"Episodes: {n_eps}, fps: {ds.fps}, key: {args.key}")

    for ep in range(n_eps):
        start = ds.episode_data_index["from"][ep].item()
        end = ds.episode_data_index["to"][ep].item()
        frames = []
        for i in range(start, end):
            v = ds[i][args.key]
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            frames.append(v.astype(np.float32))
        arr = np.stack(frames, axis=0)  # (T, 10)
        assert arr.ndim == 2 and arr.shape[1] == 10, f"Episode {ep} unexpected shape {arr.shape}"

        out_path = out_dir / f"demo_{ep}.npy"
        np.save(out_path, arr)

        if ep < 3 or ep == n_eps - 1:
            print(f"  {out_path.name}: shape={arr.shape}  "
                  f"first_trans={arr[0, 0:3].round(3)}  "
                  f"first_gripper={arr[0, 9]:.2f}  "
                  f"last_gripper={arr[-1, 9]:.2f}")

    print(f"Wrote {n_eps} demos to {out_dir}")


if __name__ == "__main__":
    main()
