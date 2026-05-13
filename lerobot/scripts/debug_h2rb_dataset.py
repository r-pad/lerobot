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

# point index → (label, BGR color)
_POINT_META = {
    0: ("right", (0,   255,   0)),
    1: ("left",  (0,   200, 255)),
    2: ("top",   (255, 100,   0)),
    3: ("grasp", (0,     0, 255)),
}


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


def project_points(points_world: np.ndarray, K: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    """Project (N, 3) world-frame points → (N, 2) pixel coords."""
    world_to_cam = np.linalg.inv(T_world_cam)
    homo = np.concatenate([points_world, np.ones((len(points_world), 1))], axis=-1)  # (N, 4)
    pts_cam = (world_to_cam @ homo.T).T[:, :3]                                       # (N, 3)
    pts_h   = (K @ pts_cam.T).T                                                       # (N, 3)
    return pts_h[:, :2] / pts_h[:, 2:3]                                              # (N, 2)


def draw_gripper_pcd(img_bgr: np.ndarray, pts_2d: np.ndarray, alpha: float = 0.9) -> np.ndarray:
    """Draw projected gripper points with index + label on a BGR image."""
    img = img_bgr.copy()
    H, W = img.shape[:2]
    for i, pt in enumerate(pts_2d):
        label, color = _POINT_META[i]
        x, y = int(round(pt[0])), int(round(pt[1]))
        if not (0 <= x < W and 0 <= y < H):
            continue
        cv2.circle(img, (x, y), 7, color, -1)
        cv2.circle(img, (x, y), 7, (255, 255, 255), 1)   # white outline
        cv2.putText(img, f"{i}:{label}", (x + 9, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return img


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


def debug_gripper_pcd_episode(repo_id: str, ep_idx: int, output_dir: Path) -> None:
    """
    For each frame in the episode, project observation.points.gripper_pcds and
    goal_gripper_pcds onto the front/left camera views, label each point 0-3,
    and write a side-by-side video.
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset    = LeRobotDataset(repo_id)
    from_idx   = dataset.episode_data_index["from"][ep_idx].item()
    to_idx     = dataset.episode_data_index["to"][ep_idx].item()
    fps        = dataset.meta.fps
    root       = repo_root(repo_id)

    cam_cfgs = [
        ("cam_azure_kinect_front", "observation.images.cam_azure_kinect_front.color"),
        ("cam_azure_kinect_left",  "observation.images.cam_azure_kinect_left.color"),
    ]

    for cam_key, vid_feat_key in cam_cfgs:
        intr_key = f"observation.{cam_key}.intrinsics"
        extr_key = f"observation.{cam_key}.extrinsics"
        if intr_key not in dataset.meta.features:
            print(f"  [{cam_key}] intrinsics not found in dataset, skipping.")
            continue

        with open(root / "meta" / "info.json") as f:
            info = json.load(f)
        vid_path = get_video_path(root, info, ep_idx, vid_feat_key)
        if not vid_path.exists():
            print(f"  [{cam_key}] video not found: {vid_path}")
            continue

        color_frames = load_rgb_video(vid_path)   # (T, H, W, 3) RGB uint8
        T = min(len(color_frames), to_idx - from_idx)
        out_frames = []

        # Batch-load all non-video features via to_pydict() → plain Python lists,
        # then convert to numpy once. This avoids per-row PyArrow deprecation warnings.
        ep_dict  = dataset.hf_dataset.select(range(from_idx, from_idx + T)).to_pydict()
        curr_pcds = np.array(ep_dict["observation.points.gripper_pcds"],      dtype=np.float32)  # (T, 4, 3)
        goal_pcds = np.array(ep_dict["observation.points.goal_gripper_pcds"], dtype=np.float32)  # (T, 4, 3)
        Ks        = np.array(ep_dict[intr_key], dtype=np.float32)                                # (T, 3, 3)
        T_wcs     = np.array(ep_dict[extr_key], dtype=np.float32)                               # (T, 4, 4)

        print(f"  [{cam_key}] Rendering {T} frames...")
        for t in tqdm(range(T), desc=f"  [{cam_key}]", leave=False):
            curr_pcd = curr_pcds[t]   # (4, 3)
            goal_pcd = goal_pcds[t]   # (4, 3)
            K        = Ks[t]          # (3, 3)
            T_wc     = T_wcs[t]       # (4, 4) cam→world

            rgb = color_frames[t]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            curr_pts = project_points(curr_pcd, K, T_wc)
            goal_pts = project_points(goal_pcd, K, T_wc)

            bgr_curr = draw_gripper_pcd(bgr, curr_pts)       # current  (solid colors)
            bgr_goal = draw_gripper_pcd(bgr, goal_pts)        # goal     (same colors, labeled)

            row = np.concatenate([
                cv2.cvtColor(bgr_curr, cv2.COLOR_BGR2RGB),
                cv2.cvtColor(bgr_goal, cv2.COLOR_BGR2RGB),
            ], axis=1)
            out_frames.append(row)

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"ep{ep_idx:04d}_{cam_key}_gripper_pcd.mp4"
        iio.imwrite(str(out_path), np.stack(out_frames), fps=fps,
                    plugin="pyav", codec="h264")
        print(f"  [{cam_key}] Saved → {out_path}")

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
    parser.add_argument("--alpha",       type=float, default=0.4,
                        help="Heatmap blend alpha: 0=invisible, 1=full (default: 0.4)")
    parser.add_argument("--gripper_pcd", action="store_true",
                        help="Visualize gripper PCD points with numbered labels instead of heatmaps")
    args = parser.parse_args()

    if args.gripper_pcd:
        debug_gripper_pcd_episode(
            repo_id=args.repo_id,
            ep_idx=args.episode,
            output_dir=Path(args.output),
        )
    else:
        debug_episode(
            repo_id=args.repo_id,
            ep_idx=args.episode,
            output_dir=Path(args.output),
            alpha=args.alpha,
        )
