"""Compute per-timestep statistics for relative actions."""

import logging

import numpy as np
from tqdm import tqdm

from lerobot.common.datasets.compute_stats import sample_indices
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def compute_relative_action_stats(
    dataset: LeRobotDataset,
    policy_config,
) -> None:
    """Compute per-timestep percentile stats for action.right_eef_pose_relative.

    Updates dataset.meta.stats in-place.

    Args:
        dataset: Dataset to compute stats from
        policy_config: Policy config with action_space, horizon, and robot adapter
    """
    # Only compute if using relative action space
    if not (hasattr(policy_config, "action_space") and policy_config.action_space == "right_eef_relative"):
        return

    REL_ACTION_KEY = "action.right_eef_pose_relative"
    horizon = policy_config.horizon
    robot_adapter = policy_config.get_robot_adapter()

    # Subsample dataset
    indices = sample_indices(len(dataset))[:10_000]
    logging.info(f"Computing per-timestep stats for '{REL_ACTION_KEY}' using {len(indices)} samples...")

    # Collect relative actions per timestep
    actions_per_timestep = [[] for _ in range(horizon)]

    for idx in tqdm(indices):
        sample = dataset[idx]

        # Convert absolute → relative using robot adapter
        batch = {
            robot_adapter.get_obs_key(): sample[robot_adapter.get_obs_key()].unsqueeze(0),
            "action.right_eef_pose": sample["action.right_eef_pose"].unsqueeze(0),
        }
        batch = robot_adapter.compute_relative_actions(batch)
        relative_action = batch[REL_ACTION_KEY].squeeze(0)  # (horizon, action_dim)

        # Collect per timestep
        for t in range(horizon):
            actions_per_timestep[t].append(relative_action[t].cpu().numpy())

    # Compute percentiles per timestep
    stats = {}
    for t in range(horizon):
        actions_t = np.stack(actions_per_timestep[t])  # (num_samples, action_dim)
        stats[f"timestep_{t}_p02"] = np.percentile(actions_t, 2, axis=0)
        stats[f"timestep_{t}_p98"] = np.percentile(actions_t, 98, axis=0)

    # Update dataset stats in-place
    dataset.meta.stats[REL_ACTION_KEY] = stats
    logging.info(f"Added per-timestep stats for '{REL_ACTION_KEY}'")
