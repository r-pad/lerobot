"""
Debug h2rb dataset by overlaying goal_gripper_proj heatmaps on RGB videos.

Automatically discovers all color/heatmap pairs in the dataset and writes
side-by-side overlay videos for visual inspection.

Usage:
    python debug_h2rb_dataset.py \
        --repo_id   haotian/pick_place_red_mug_100 \
        --episode   0 \
        --output    /tmp/debug_overlay
"""

import argparse
import json
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
from tqdm import tqdm


LEROBOT_CACHE = Path.home() / ".cache" / "huggingface" / "lerobot"


def repo_root(repo_id: str) -> Path:
    return LEROBOT_CACHE / repo_id


def load_rgb_video(path: Path) -> np.ndarray:
    frames = iio.imread(str(path), plugin="pyav")
    return frames  # (T, H, W, 3) uint8


def colorize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """
    heatmap: (H, W, 3) uint8 — values are distance-based (higher = farther from goal).
    Returns (H, W, 3) uint8 RGB with jet colormap, goal points appear bright red.
    """
    gray = heatmap[:, :, 0]                              # all channels equal
    gray_inv = 255 - gray                                # invert: goal = bright
    colored_bgr = cv2.applyColorMap(gray_inv, cv2.COLORMAP_JET)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)  # (H, W, 3) RGB


def overlay_heatmap(rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    colored = colorize_heatmap(heatmap).astype(np.float32)
    blended = rgb.astype(np.float32) * (1 - alpha) + colored * alpha
    return blended.clip(0, 255).astype(np.uint8)


def find_cam_pairs(features: dict) -> list[tuple[str, str]]:
    """Return (color_key, heatmap_key) pairs present in the dataset."""
    pairs = []
    for key in sorted(features):
        if features[key]["dtype"] == "video" and key.endswith(".color"):
            prefix = key[: -len(".color")]
            heatmap_key = f"{prefix}.goal_gripper_proj"
            if heatmap_key in features:
                pairs.append((key, heatmap_key))
    return pairs


def get_video_path(root: Path, info: dict, ep_idx: int, vid_key: str) -> Path:
    ep_chunk = ep_idx // info.get("chunks_size", 1000)
    rel = info["video_path"].format(
        episode_chunk=ep_chunk, video_key=vid_key, episode_index=ep_idx
    )
    path = root / rel
    if not path.exists():
        path = path.with_suffix(".mkv")
    return path


def debug_episode(repo_id: str, ep_idx: int, output_dir: Path, alpha: float) -> None:
    root = repo_root(repo_id)
    if not root.exists():
        raise FileNotFoundError(f"Dataset not found locally: {root}")

    with open(root / "meta" / "info.json") as f:
        info = json.load(f)

    features = info["features"]
    n_episodes = info["total_episodes"]

    if ep_idx >= n_episodes:
        raise ValueError(f"Episode {ep_idx} out of range (dataset has {n_episodes} episodes)")

    cam_pairs = find_cam_pairs(features)
    if not cam_pairs:
        all_video_keys = [k for k, v in features.items() if v["dtype"] == "video"]
        print(f"No color/heatmap pairs found. Video features: {all_video_keys}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    fps = info.get("fps", 30)

    print(f"Dataset : {repo_id}")
    print(f"Episode : {ep_idx} / {n_episodes - 1}")
    print(f"Cameras : {[c for c, _ in cam_pairs]}")
    print(f"Output  : {output_dir}")

    for color_key, heatmap_key in cam_pairs:
        cam_name = color_key.removeprefix("observation.images.").removesuffix(".color")

        color_path   = get_video_path(root, info, ep_idx, color_key)
        heatmap_path = get_video_path(root, info, ep_idx, heatmap_key)

        if not color_path.exists():
            print(f"  [{cam_name}] Missing: {color_path}")
            continue
        if not heatmap_path.exists():
            print(f"  [{cam_name}] Missing: {heatmap_path}")
            continue

        print(f"  [{cam_name}] Loading videos...")
        color_frames   = load_rgb_video(color_path)    # (T, H, W, 3)
        heatmap_frames = load_rgb_video(heatmap_path)  # (T, H, W, 3)

        T = min(len(color_frames), len(heatmap_frames))
        out_frames = []

        for t in tqdm(range(T), desc=f"  [{cam_name}]", leave=False):
            rgb     = color_frames[t]
            heat    = heatmap_frames[t]
            blended = overlay_heatmap(rgb, heat, alpha=alpha)
            # Side-by-side: original | overlay
            row = np.concatenate([rgb, blended], axis=1)
            out_frames.append(row)

        out_path = output_dir / f"ep{ep_idx:04d}_{cam_name}.mp4"
        iio.imwrite(str(out_path), np.stack(out_frames), fps=fps,
                    plugin="pyav", codec="h264")
        print(f"  [{cam_name}] Saved → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overlay goal_gripper_proj heatmaps on RGB videos for debugging.",
    )
    parser.add_argument("--repo_id",  type=str, required=True,
                        help="Local LeRobot repo_id (e.g. haotian/pick_place_red_mug_100)")
    parser.add_argument("--episode",  type=int, default=0,
                        help="Episode index to visualize (default: 0)")
    parser.add_argument("--output",   type=str, default="/tmp/debug_h2rb",
                        help="Output directory for overlay videos")
    parser.add_argument("--alpha",    type=float, default=0.4,
                        help="Heatmap blend alpha: 0=invisible, 1=full (default: 0.5)")
    args = parser.parse_args()

    debug_episode(
        repo_id=args.repo_id,
        ep_idx=args.episode,
        output_dir=Path(args.output),
        alpha=args.alpha,
    )
