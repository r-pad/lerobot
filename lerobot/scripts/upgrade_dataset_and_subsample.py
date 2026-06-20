"""
Subsample a LeRobot dataset to a lower FPS and add next_event_idx (gripper event goals).

Combines subsampling from subsample_dataset.py with next_event_idx extraction from
upgrade_dataset.py. Event detection runs on the full-rate episode; goal frame indices
are mapped into the subsampled episode index space.

Example:
```bash
python lerobot/scripts/upgrade_dataset_and_subsample.py \\
    --source_repo_id sriramsk/my_dataset \\
    --target_repo_id sriramsk/my_dataset_subsampled_events \\
    --target_fps 15 \\
    --close_threshold 58 \\
    --open_threshold 70 \\
    --visualize_goal_indices
```
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.scripts.upgrade_dataset import extract_events_with_gripper_pos

AUTO_FIELDS = {"episode_index", "frame_index", "index", "task_index", "timestamp"}


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


def _prepare_subsampled_frame(frame: dict, tasks: dict[int, str], feature_keys: set[str]) -> dict:
    frame_data = {k: v for k, v in frame.items() if k not in AUTO_FIELDS and k in feature_keys}
    frame_data["task"] = tasks[frame["task_index"].item()]

    for key in list(frame_data.keys()):
        if key.startswith("observation.images.cam_azure_kinect"):
            if key.endswith(".color"):
                frame_data[key] = (frame_data[key].permute(1, 2, 0) * 255).to(torch.uint8)
            elif key.endswith(".transformed_depth"):
                frame_data[key] = (frame_data[key].permute(1, 2, 0) * 1000).to(torch.uint16)
            elif key.endswith(".goal_gripper_proj"):
                frame_data[key] = (frame_data[key].permute(1, 2, 0) * 255).to(torch.uint8)

    if "observation.images.cam_wrist" in frame_data:
        frame_data["observation.images.cam_wrist"] = (
            frame_data["observation.images.cam_wrist"].permute(1, 2, 0) * 255
        ).to(torch.uint8)

    for key, value in frame_data.items():
        if isinstance(value, torch.Tensor):
            frame_data[key] = value.numpy()

    return frame_data


def _load_episode_joint_states(source_dataset: LeRobotDataset, ep_start: int, ep_end: int) -> np.ndarray:
    """Load observation.state from parquet only (no video decode)."""
    indices = list(range(ep_start, ep_end))
    states = source_dataset.hf_dataset.select(indices)["observation.state"]
    return torch.stack(tuple(states)).numpy()


def _rgb_camera_keys_from_frame(frame_data: dict) -> list[tuple[str, str]]:
    """Return (label, feature_key) pairs for RGB images in a frame dict."""
    rgb_keys = []
    for key in frame_data:
        if not key.endswith(".color") or not key.startswith("observation.images."):
            continue
        parts = key.split(".")
        if len(parts) >= 4:
            rgb_keys.append((parts[2], key))
    wrist_key = "observation.images.cam_wrist"
    if wrist_key in frame_data:
        rgb_keys.append(("cam_wrist", wrist_key))
    return rgb_keys


def _visualize_goal_rgb_images(
    frames: list[dict],
    goal_indices_sub: list[int],
    camera_names: list[str],
    episode_idx: int,
    target_repo_id: str,
    gripper_pos: Optional[np.ndarray] = None,
    goal_indices_full: Optional[list[int]] = None,
    subsample_factor: int = 1,
    show_interactive: bool = False,
) -> None:
    """
    Save goal RGB frames under data/{repo_id}_goal_images/ (same layout as upgrade_dataset.py).

    goal_indices_sub are indices into the subsampled `frames` list.
    goal_indices_full are the original full-fps episode indices (for labels).
    """
    if not frames:
        print(f"Episode {episode_idx}: no frames to visualize.")
        return

    rgb_keys = _rgb_camera_keys_from_frame(frames[0])
    if camera_names:
        rgb_keys = [(n, k) for n, k in rgb_keys if n in camera_names] or rgb_keys
    if not rgb_keys:
        print(f"No RGB images in episode {episode_idx}, skipping visualization.")
        return

    safe_repo_id = target_repo_id.replace("/", "_")
    output_dir = Path(f"/data/yufei/lerobot/data/{safe_repo_id}_goal_images") / f"episode_{episode_idx:06d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    mosaic_rows = []

    for goal_i, goal_idx_sub in enumerate(goal_indices_sub):
        if goal_idx_sub >= len(frames):
            continue

        row_images = []
        for cam_label, key in rgb_keys:
            img_arr = frames[goal_idx_sub][key]
            if isinstance(img_arr, torch.Tensor):
                img_arr = img_arr.numpy()
            img = Image.fromarray(img_arr).convert("RGB")
            draw = ImageDraw.Draw(img)

            full_idx = (
                _full_frame_for_sub_goal(
                    goal_idx_sub, goal_indices_full, subsample_factor
                )
                if goal_indices_full is not None
                else goal_idx_sub * subsample_factor
            )
            label = f"goal {goal_i} @ sub {goal_idx_sub} (full {full_idx}) | {cam_label}"
            if gripper_pos is not None and goal_idx_sub < len(gripper_pos):
                label += f" | gripper={gripper_pos[goal_idx_sub]:.1f}"
            draw.rectangle((0, 0, min(img.width, 900), 30), fill=(0, 0, 0))
            draw.text((4, 6), label, fill=(255, 255, 255))
            row_images.append(img)
            img.save(output_dir / f"goal{goal_i:02d}_sub{goal_idx_sub:06d}_full{full_idx:06d}_{cam_label}.png")

        row_width = sum(im.width for im in row_images)
        row_height = max(im.height for im in row_images)
        row_canvas = Image.new("RGB", (row_width, row_height))
        x_offset = 0
        for im in row_images:
            row_canvas.paste(im, (x_offset, 0))
            x_offset += im.width
        mosaic_rows.append(row_canvas)

    if not mosaic_rows:
        return

    mosaic_width = max(row.width for row in mosaic_rows)
    mosaic_height = sum(row.height for row in mosaic_rows)
    mosaic = Image.new("RGB", (mosaic_width, mosaic_height))
    y_offset = 0
    for row in mosaic_rows:
        mosaic.paste(row, (0, y_offset))
        y_offset += row.height

    mosaic.save(output_dir / "mosaic.png")
    with open(output_dir / "goal_indices_subsampled.txt", "w") as f:
        f.write("\n".join(str(i) for i in goal_indices_sub))
    if goal_indices_full is not None:
        with open(output_dir / "goal_indices_full_fps.txt", "w") as f:
            f.write("\n".join(str(i) for i in goal_indices_full))

    title = f"Episode {episode_idx} | sub goals: {goal_indices_sub}"
    if goal_indices_full is not None:
        title += f"\nfull fps goals: {goal_indices_full}"
    if gripper_pos is not None:
        gripper_vals = [
            f"{gripper_pos[i]:.1f}" for i in goal_indices_sub if i < len(gripper_pos)
        ]
        title += f"\ngripper @ sub goals: {gripper_vals}"

    fig, ax = plt.subplots(figsize=(min(16, mosaic_width / 80), min(10, mosaic_height / 80)))
    ax.imshow(mosaic)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_dir / "mosaic_labeled.png", dpi=150, bbox_inches="tight")
    print(f"Goal RGB visualization saved to {output_dir}")
    if show_interactive:
        print("Close the matplotlib window to continue.")
        plt.show(block=True)
        plt.close(fig)


def _extract_goal_indices(
    joint_states: np.ndarray,
    close_thresh: float,
    open_thresh: float,
    close_thresholds: Optional[list[float]],
    open_thresholds: Optional[list[float]],
) -> list[int]:
    if close_thresholds is not None and open_thresholds is not None:
        if len(close_thresholds) == 1 and len(open_thresholds) == 1:
            return extract_events_with_gripper_pos(
                joint_states,
                close_thresh=close_thresholds[0],
                open_thresh=open_thresholds[0],
            )
        return extract_events_with_gripper_pos(
            joint_states,
            close_thresholds=close_thresholds,
            open_thresholds=open_thresholds,
        )
    return extract_events_with_gripper_pos(
        joint_states,
        close_thresh=close_thresh,
        open_thresh=open_thresh,
    )


def _full_frame_for_sub_goal(
    goal_idx_sub: int,
    goal_indices_full: list[int],
    subsample_factor: int,
) -> int:
    for goal_idx in goal_indices_full:
        if goal_idx // subsample_factor == goal_idx_sub:
            return goal_idx
    return goal_idx_sub * subsample_factor


def _subsampled_goal_indices(
    goal_indices_full: list[int],
    subsample_factor: int,
    episode_length: int,
) -> list[int]:
    """Map full-rate goal frame indices to subsampled episode indices."""
    num_sub_frames = len(range(0, episode_length, subsample_factor))
    subsampled: list[int] = []
    for goal_idx in goal_indices_full:
        sub_idx = goal_idx // subsample_factor
        if not subsampled or subsampled[-1] != sub_idx:
            subsampled.append(sub_idx)
    if not subsampled or subsampled[-1] != num_sub_frames - 1:
        subsampled.append(num_sub_frames - 1)
    return subsampled


def _next_event_idx_value(
    local_frame_idx: int,
    goal_indices_full: list[int],
    subsampled_goal_indices: list[int],
) -> np.ndarray:
    """Index of the next goal in subsampled frame coordinates."""
    goal_slot = 0
    while goal_slot < len(goal_indices_full) - 1 and local_frame_idx >= goal_indices_full[goal_slot]:
        goal_slot += 1
    return np.array([subsampled_goal_indices[goal_slot]], dtype=np.int32)


def _read_subsampled_episode(
    episode_idx: int,
    source_repo_id: str,
    subsample_factor: int,
    tolerance_s: float,
    tasks: dict[int, str],
    feature_keys: set[str],
    close_thresh: float,
    open_thresh: float,
    close_thresholds: Optional[list[float]],
    open_thresholds: Optional[list[float]],
    add_next_event_idx: bool,
    show_frame_progress: bool = True,
) -> tuple[list[dict], float, int, list[int], list[int], np.ndarray]:
    """Load one episode, detect goals on full-rate state, return subsampled frames."""
    t0 = time.perf_counter()
    source_dataset = LeRobotDataset(
        source_repo_id,
        episodes=[episode_idx],
        tolerance_s=tolerance_s,
    )
    ep_start = source_dataset.episode_data_index["from"][0].item()
    ep_end = source_dataset.episode_data_index["to"][0].item()
    episode_length = ep_end - ep_start

    joint_states = _load_episode_joint_states(source_dataset, ep_start, ep_end)
    goal_indices_full = _extract_goal_indices(
        joint_states,
        close_thresh,
        open_thresh,
        close_thresholds,
        open_thresholds,
    )
    subsampled_goals = _subsampled_goal_indices(
        goal_indices_full, subsample_factor, episode_length
    )

    frames: list[dict] = []
    for global_idx in tqdm(
        range(ep_start, ep_end, subsample_factor),
        desc=f"Episode {episode_idx} (read)",
        leave=False,
        disable=not show_frame_progress,
    ):
        local_idx = global_idx - ep_start
        frame_data = _prepare_subsampled_frame(
            source_dataset[global_idx], tasks, feature_keys
        )
        if add_next_event_idx:
            frame_data["next_event_idx"] = _next_event_idx_value(
                local_idx, goal_indices_full, subsampled_goals
            )
        frames.append(frame_data)

    gripper_pos_sub = joint_states[::subsample_factor, 17]
    return (
        frames,
        time.perf_counter() - t0,
        len(frames),
        goal_indices_full,
        subsampled_goals,
        gripper_pos_sub,
    )


def _write_episode(
    target_dataset: LeRobotDataset,
    episode_idx: int,
    frames: list[dict],
    show_frame_progress: bool = False,
) -> float:
    t0 = time.perf_counter()
    for frame_data in tqdm(
        frames,
        desc=f"Episode {episode_idx} (write)",
        leave=False,
        disable=not show_frame_progress,
    ):
        target_dataset.add_frame(frame_data)
    target_dataset.save_episode()
    return time.perf_counter() - t0


def upgrade_and_subsample_dataset(
    source_repo_id: str,
    target_repo_id: str,
    target_fps: int = 15,
    discard_episodes: Optional[list[int]] = None,
    close_thresh: float = 52.0,
    open_thresh: float = 65.0,
    close_thresholds: Optional[list[float]] = None,
    open_thresholds: Optional[list[float]] = None,
    add_next_event_idx: bool = True,
    visualize_goal_indices: bool = False,
    show_goal_plot: bool = False,
    camera_names: Optional[list[str]] = None,
) -> LeRobotDataset:
    tolerance_s = 0.0004
    discard_episodes = discard_episodes or []

    print(f"Loading source dataset: {source_repo_id}")
    source_meta = LeRobotDatasetMetadata(source_repo_id)
    source_dataset = LeRobotDataset(source_repo_id, tolerance_s=tolerance_s)

    source_fps = source_dataset.fps
    if source_fps % target_fps != 0:
        print(
            f"Warning: {source_fps} doesn't divide evenly by {target_fps}, "
            "timestamps may drift slightly"
        )
    subsample_factor = source_fps // target_fps
    print(f"Subsampling from {source_fps}fps to {target_fps}fps (factor: {subsample_factor})")

    target_features = dict(source_dataset.features)
    if add_next_event_idx:
        target_features["next_event_idx"] = {
            "dtype": "int32",
            "shape": (1,),
            "names": ["idx"],
            "info": "Index of next gripper event goal in subsampled episode frames",
        }

    target_dataset = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=target_fps,
        features=target_features,
    )

    num_episodes = source_meta.info["total_episodes"]
    tasks = source_meta.tasks
    feature_keys = set(source_dataset.features.keys())

    episode_indices = [i for i in range(num_episodes) if i not in discard_episodes]
    if not episode_indices:
        print("No episodes to process.")
        return target_dataset

    print(f"Processing {len(episode_indices)} episodes sequentially")
    episode_times: list[float] = []

    for episode_idx in tqdm(episode_indices, desc="Upgrade+subsample", unit="ep"):
        ep_t0 = time.perf_counter()
        frames, read_s, n_frames, goals_full, goals_sub, gripper_sub = _read_subsampled_episode(
            episode_idx,
            source_repo_id,
            subsample_factor,
            tolerance_s,
            tasks,
            feature_keys,
            close_thresh,
            open_thresh,
            close_thresholds,
            open_thresholds,
            add_next_event_idx,
        )
        if visualize_goal_indices:
            _visualize_goal_rgb_images(
                frames,
                goals_sub,
                camera_names or [],
                episode_idx,
                target_repo_id,
                gripper_pos=gripper_sub,
                goal_indices_full=goals_full,
                subsample_factor=subsample_factor,
                show_interactive=show_goal_plot,
            )
        write_s = _write_episode(target_dataset, episode_idx, frames)
        episode_times.append(time.perf_counter() - ep_t0)
        avg_s = sum(episode_times) / len(episode_times)
        remaining = len(episode_indices) - len(episode_times)
        tqdm.write(
            f"Episode {episode_idx}: {n_frames} frames, "
            f"goals sub {goals_sub} (full {goals_full}), "
            f"read {_format_duration(read_s)}, write {_format_duration(write_s)}, "
            f"eta ~{_format_duration(remaining * avg_s)}"
        )

    print(
        f"Done! {target_repo_id}: {len(target_dataset)} frames at {target_fps}fps "
        f"({'with' if add_next_event_idx else 'without'} next_event_idx)"
    )
    return target_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Subsample a LeRobot dataset and add next_event_idx from gripper events."
    )
    parser.add_argument(
        "--source_repo_id",
        "--source",
        type=str,
        required=True,
        help="Source dataset repository ID",
    )
    parser.add_argument(
        "--target_repo_id",
        "--target",
        type=str,
        required=True,
        help="Target dataset repository ID",
    )
    parser.add_argument(
        "--target_fps",
        type=int,
        default=15,
        help="Target FPS after subsampling (default: 15)",
    )
    parser.add_argument(
        "--discard_episodes",
        type=int,
        nargs="*",
        default=[],
        help="Episode indices to skip",
    )
    parser.add_argument(
        "--close_threshold",
        type=float,
        nargs="+",
        default=[52],
        help="Close gripper threshold(s) on full-rate joint states",
    )
    parser.add_argument(
        "--open_threshold",
        type=float,
        nargs="+",
        default=[65],
        help="Open gripper threshold(s); length must match --close_threshold when multiple",
    )
    parser.add_argument(
        "--no_next_event_idx",
        action="store_true",
        help="Only subsample; do not add next_event_idx",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push result to Hugging Face Hub",
    )
    parser.add_argument(
        "--visualize_goal_indices",
        action="store_true",
        help="Save RGB mosaics at subsampled goal frames under data/{target_repo}_goal_images/",
    )
    parser.add_argument(
        "--show_goal_plot",
        action="store_true",
        help="Show interactive matplotlib window per episode",
    )
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="*",
        default=None,
        help="Cameras to include in visualization (default: all azure kinect + wrist)",
    )
    args = parser.parse_args()

    close_thresholds = args.close_threshold if len(args.close_threshold) > 1 else None
    open_thresholds = args.open_threshold if len(args.open_threshold) > 1 else None
    close_thresh = args.close_threshold[0]
    open_thresh = args.open_threshold[0]

    dataset = upgrade_and_subsample_dataset(
        source_repo_id=args.source_repo_id,
        target_repo_id=args.target_repo_id,
        target_fps=args.target_fps,
        discard_episodes=args.discard_episodes,
        close_thresh=close_thresh,
        open_thresh=open_thresh,
        close_thresholds=close_thresholds,
        open_thresholds=open_thresholds,
        add_next_event_idx=not args.no_next_event_idx,
        visualize_goal_indices=args.visualize_goal_indices,
        show_goal_plot=args.show_goal_plot,
        camera_names=args.camera_names,
    )

    if args.push_to_hub:
        dataset.push_to_hub(repo_id=args.target_repo_id)

    print("Dataset upgrade + subsample completed successfully!")
