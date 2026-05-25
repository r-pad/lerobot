import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import torch
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

AUTO_FIELDS = {"episode_index", "frame_index", "index", "task_index", "timestamp"}


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


def _prepare_subsampled_frame(frame: dict, tasks: dict[int, str], feature_keys: set[str]) -> dict:
    frame_data = {
        k: v for k, v in frame.items() if k not in AUTO_FIELDS and k in feature_keys
    }
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


def _subsample_episode_worker(
    episode_idx: int,
    source_repo_id: str,
    subsample_factor: int,
    tolerance_s: float,
    tasks: dict[int, str],
    feature_keys: set[str],
    show_frame_progress: bool = False,
) -> tuple[int, list[dict], float, int]:
    """Load one episode, subsample frames, and return converted frame dicts."""
    t0 = time.perf_counter()
    source_dataset = LeRobotDataset(
        source_repo_id,
        episodes=[episode_idx],
        tolerance_s=tolerance_s,
    )
    ep_start = source_dataset.episode_data_index["from"][0].item()
    ep_end = source_dataset.episode_data_index["to"][0].item()
    indices = range(ep_start, ep_end, subsample_factor)

    frames = []
    index_iter = tqdm(
        indices,
        desc=f"Episode {episode_idx} (read)",
        leave=False,
        disable=not show_frame_progress,
    )
    for idx in index_iter:
        frame = source_dataset[idx]
        frames.append(_prepare_subsampled_frame(frame, tasks, feature_keys))

    return episode_idx, frames, time.perf_counter() - t0, len(frames)


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


def subsample_dataset(
    source_repo_id: str,
    target_repo_id: str,
    target_fps: int = 15,
    num_workers: int = 1,
) -> LeRobotDataset:
    tolerance_s = 0.0004

    print(f"Loading source dataset: {source_repo_id}")
    source_meta = LeRobotDatasetMetadata(source_repo_id)
    source_dataset = LeRobotDataset(source_repo_id, tolerance_s=tolerance_s)

    source_fps = source_dataset.fps
    if source_fps % target_fps != 0:
        print(f"Warning: {source_fps} doesn't divide evenly by {target_fps}, might get slight drift")
    subsample_factor = source_fps // target_fps

    print(f"Subsampling from {source_fps}fps to {target_fps}fps (factor: {subsample_factor})")

    subsampled_dataset = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=target_fps,
        features=source_dataset.features,
    )

    num_episodes = source_meta.info["total_episodes"]
    tasks = source_meta.tasks
    feature_keys = set(source_dataset.features.keys())
    worker_fn = partial(
        _subsample_episode_worker,
        source_repo_id=source_repo_id,
        subsample_factor=subsample_factor,
        tolerance_s=tolerance_s,
        tasks=tasks,
        feature_keys=feature_keys,
    )

    if num_workers <= 1:
        read_fn = partial(worker_fn, show_frame_progress=True)
        for episode_idx in range(num_episodes):
            _, frames, read_s, n_frames = read_fn(episode_idx)
            write_s = _write_episode(
                subsampled_dataset, episode_idx, frames, show_frame_progress=True
            )
            tqdm.write(
                f"Episode {episode_idx}: {n_frames} frames, "
                f"read {_format_duration(read_s)}, write {_format_duration(write_s)}"
            )
    else:
        num_workers = min(num_workers, num_episodes)
        print(f"Processing {num_episodes} episodes with {num_workers} workers")

        results: list[tuple[list[dict], float, int] | None] = [None] * num_episodes
        completed_times: list[float] = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(worker_fn, episode_idx): episode_idx
                for episode_idx in range(num_episodes)
            }
            with tqdm(
                total=num_episodes,
                desc="Subsample (read)",
                unit="ep",
                dynamic_ncols=True,
            ) as pbar:
                for future in as_completed(futures):
                    episode_idx, frames, elapsed_s, n_frames = future.result()
                    results[episode_idx] = (frames, elapsed_s, n_frames)
                    completed_times.append(elapsed_s)
                    avg_s = sum(completed_times) / len(completed_times)
                    remaining = num_episodes - len(completed_times)
                    eta_s = remaining * avg_s / num_workers
                    pbar.set_postfix(
                        last_ep=episode_idx,
                        last_s=f"{elapsed_s:.0f}s",
                        frames=n_frames,
                        avg=f"{avg_s:.0f}s/ep",
                        eta=f"~{_format_duration(eta_s)}",
                    )
                    pbar.update(1)
                    tqdm.write(
                        f"Episode {episode_idx} read done: {n_frames} frames in {_format_duration(elapsed_s)}"
                    )

        write_times: list[float] = []
        with tqdm(total=num_episodes, desc="Write to disk", unit="ep", dynamic_ncols=True) as pbar:
            for episode_idx in range(num_episodes):
                frames, read_s, n_frames = results[episode_idx]
                write_s = _write_episode(subsampled_dataset, episode_idx, frames)
                write_times.append(write_s)
                avg_s = sum(write_times) / len(write_times)
                remaining = num_episodes - len(write_times)
                eta_s = remaining * avg_s
                pbar.set_postfix(
                    ep=episode_idx,
                    frames=n_frames,
                    write_s=f"{write_s:.0f}s",
                    avg=f"{avg_s:.0f}s/ep",
                    eta=f"~{_format_duration(eta_s)}",
                )
                pbar.update(1)
                tqdm.write(
                    f"Episode {episode_idx} written: {n_frames} frames, "
                    f"read {_format_duration(read_s)}, write {_format_duration(write_s)}"
                )

    print(
        f"Subsampling complete! New dataset has {len(subsampled_dataset)} frames at {target_fps}fps"
    )
    return subsampled_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample a LeRobot dataset to lower fps")
    parser.add_argument("--source", type=str, required=True, help="Source dataset repo ID")
    parser.add_argument("--target", type=str, required=True, help="Target dataset repo ID")
    parser.add_argument("--target_fps", type=int, default=15, help="Target FPS (default: 15)")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Parallel workers for episode processing (default: CPU count)",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Push to HuggingFace Hub")

    args = parser.parse_args()
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = os.cpu_count() or 1

    subsampled_dataset = subsample_dataset(
        args.source, args.target, args.target_fps, num_workers=num_workers
    )

    if args.push_to_hub:
        subsampled_dataset.push_to_hub(repo_id=args.target)

    print("Done!")
