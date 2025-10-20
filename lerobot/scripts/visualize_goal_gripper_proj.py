#!/usr/bin/env python3
"""
Visualize goal gripper projection heatmaps overlaid on RGB videos.

Usage:
    python lerobot/scripts/visualize_goal_gripper_proj.py --repo-id sriramsk/eval_fold_bottoms_multiTask_gc_dino3dgp_moreHuman
"""

import argparse
from pathlib import Path
import numpy as np
import imageio.v3 as iio
from matplotlib import cm
from tqdm import tqdm


def get_heatmap_viz(rgb_image, heatmap, alpha=0.4):
    """
    Overlay heatmap on RGB image with transparency.

    Args:
        rgb_image: (H, W, 3) RGB image
        heatmap: (H, W) predicted heatmap
        alpha: transparency factor for heatmap overlay

    Returns:
        overlay: (np.ndarray) RGB image with heatmap overlay
    """
    # Ensure heatmap is numpy array
    if not isinstance(heatmap, np.ndarray):
        heatmap = np.array(heatmap)

    # Ensure single channel
    if heatmap.ndim > 2:
        heatmap = heatmap.squeeze()

    # Normalize heatmap to 0-1
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # Convert to colormap (jet colormap)
    colormap = cm.jet(heatmap)[:, :, :3]  # (H, W, 3)
    colormap = (colormap * 255).astype(np.uint8)

    # Blend with original image
    overlay = (rgb_image * (1 - alpha) + colormap * alpha).astype(np.uint8)

    return overlay


def process_video_pair(rgb_video_path, heatmap_video_path, output_video_path, alpha=0.4):
    """
    Process a pair of RGB and heatmap videos to create overlay visualization.

    Args:
        rgb_video_path: Path to RGB video
        heatmap_video_path: Path to heatmap video
        output_video_path: Path to save output video
        alpha: transparency factor for heatmap overlay
    """
    # Create output directory if it doesn't exist
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    # Read videos using imageio
    rgb_video = iio.imiter(str(rgb_video_path))
    heatmap_video = iio.imiter(str(heatmap_video_path))
    fps = 60

    # Process frames
    output_frames = []
    frame_count = 0

    for rgb_frame, heatmap_frame in zip(rgb_video, heatmap_video):
        # Extract first channel from heatmap (it's 3-channel but we only need first)
        heatmap = heatmap_frame[:, :, 0]

        # Create overlay
        overlay = get_heatmap_viz(rgb_frame, heatmap, alpha=alpha)

        output_frames.append(overlay)
        frame_count += 1

    # Write output video
    iio.imwrite(str(output_video_path), output_frames, fps=fps)

    return frame_count


def main():
    parser = argparse.ArgumentParser(description="Visualize goal gripper projection heatmaps")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID (e.g., sriramsk/eval_fold_bottoms_multiTask_gc_dino3dgp_moreHuman)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory (default: ~/.cache/huggingface/lerobot)",
    )
    parser.add_argument(
        "--chunk",
        type=str,
        default="chunk-000",
        help="Chunk directory name (default: chunk-000)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Transparency factor for heatmap overlay (default: 0.4)",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="observation.images.cam_azure_kinect.color_with_heatmap",
        help="Output subdirectory name (default: observation.images.cam_azure_kinect.color_with_heatmap)",
    )

    args = parser.parse_args()

    # Determine cache directory
    if args.cache_dir is None:
        cache_dir = Path.home() / ".cache" / "huggingface" / "lerobot"
    else:
        cache_dir = Path(args.cache_dir)

    # Construct paths
    base_path = cache_dir / args.repo_id / "videos" / args.chunk
    rgb_dir = base_path / "observation.images.cam_azure_kinect.color"
    heatmap_dir = base_path / "observation.images.cam_azure_kinect.goal_gripper_proj"
    output_dir = base_path / args.output_subdir

    # Verify directories exist
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    if not heatmap_dir.exists():
        raise FileNotFoundError(f"Heatmap directory not found: {heatmap_dir}")

    # Get list of video files
    rgb_videos = sorted(rgb_dir.glob("episode_*.mp4"))
    heatmap_videos = sorted(heatmap_dir.glob("episode_*.mp4"))

    if len(rgb_videos) != len(heatmap_videos):
        raise ValueError(
            f"Mismatch in number of videos: {len(rgb_videos)} RGB videos vs {len(heatmap_videos)} heatmap videos"
        )

    if len(rgb_videos) == 0:
        raise ValueError(f"No videos found in {rgb_dir}")

    print(f"Found {len(rgb_videos)} video pairs to process")
    print(f"Output directory: {output_dir}")

    # Process each video pair
    for rgb_video, heatmap_video in tqdm(
        zip(rgb_videos, heatmap_videos), total=len(rgb_videos), desc="Processing videos"
    ):
        # Verify filenames match
        if rgb_video.name != heatmap_video.name:
            raise ValueError(f"Filename mismatch: {rgb_video.name} vs {heatmap_video.name}")

        output_video = output_dir / rgb_video.name

        # Process video pair
        frame_count = process_video_pair(rgb_video, heatmap_video, output_video, alpha=args.alpha)

    print(f"\nDone! Output videos saved to: {output_dir}")


if __name__ == "__main__":
    main()
