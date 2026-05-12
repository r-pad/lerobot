"""
Inspect a polaris diffusion policy checkpoint to see observation keys and shapes.

Usage:
    conda activate robodiff
    python lerobot/common/policies/inspect_ckpt.py --ckpt_path /path/to/checkpoint.ckpt
"""

import argparse
import torch
import dill


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.ckpt_path} ...")
    payload = torch.load(args.ckpt_path, pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]

    shape_meta = cfg.task.shape_meta

    print("\n=== Action ===")
    print(f"  shape: {shape_meta.action.shape}")

    print("\n=== Observation keys ===")
    for key, attr in shape_meta.obs.items():
        print(f"  {key}: shape={list(attr.shape)}, type={attr.get('type', 'low_dim')}")

    print("\n=== Policy config ===")
    print(f"  n_obs_steps:    {cfg.n_obs_steps}")
    print(f"  n_action_steps: {cfg.policy.n_action_steps}")
    print(f"  horizon:        {cfg.policy.horizon}")


if __name__ == "__main__":
    main()
