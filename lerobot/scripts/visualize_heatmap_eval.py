import os
import glob
import argparse

import cv2
import torch
import numpy as np
import imageio.v3 as iio
from matplotlib import cm
from typing import List, Union

def generate_heatmap_from_points(points_2d, img_shape):
    """
    Generate a 3-channel heatmap from projected 2D points.
    Creates a distance-based heatmap where each channel represents the distance
    from every pixel to one of the projected gripper points.
    Args:
        points_2d (np.ndarray): Nx2 array of 2D pixel coordinates
        img_shape (tuple): (height, width) of output image
    Returns:
        np.ndarray: HxWx3 heatmap image with uint8 values [0-255]
    """
    height, width = img_shape[:2]
    max_distance = np.sqrt(width**2 + height**2)

    # Clip points to image bounds
    clipped_points = np.clip(points_2d, [0, 0], [width - 1, height - 1]).astype(int)

    goal_image = np.zeros((height, width, 3))
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    pixel_coords = np.stack([x_coords, y_coords], axis=-1)

    # Use first 3 points for the 3 channels
    for i in range(3):
        target_point = clipped_points[i]  # (2,)
        distances = np.linalg.norm(
            pixel_coords - target_point, axis=-1
        )  # (height, width)
        goal_image[:, :, i] = distances

    # Apply square root transformation for steeper near-target gradients
    goal_image = np.sqrt(goal_image / max_distance) * 255
    goal_image = np.clip(goal_image, 0, 255).astype(np.uint8)
    return goal_image


def save_video(
    save_path: str, frames: Union[np.ndarray, List], max_size: int = None, fps: int = 30
):
    if not isinstance(frames, np.ndarray):
        frames = np.stack(frames)
    n, h, w = frames.shape[:-1]
    if max_size is not None:
        scale = max_size / max(w, h)
        h = int(h * scale // 16) * 16
        w = int(w * scale // 16) * 16
        frames = (cv2.resize(frames, output_shape=(n, h, w)) * 255).astype(np.uint8)
    print(f"saving video of size {(n, h, w)} to {save_path}")
    iio.imwrite(save_path, frames, fps=fps, extension=".webm", codec="vp9")

def get_heatmap_viz(rgb_image, heatmap, alpha=0.4):
    """
    Overlay heatmap on RGB image with transparency.

    Args:
        rgb_image: (H, W, 3) RGB image
        heatmap: (1, H, W) predicted heatmap
        alpha: transparency factor for heatmap overlay

    Returns:
        overlay: (np.ndarray) RGB image with heatmap overlay
    """
    # Get single heatmap
    if torch.is_tensor(heatmap):
        heatmap = heatmap.squeeze().cpu().numpy()  # (H, W)

    # Normalize heatmap to 0-1
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # Convert to colormap (jet colormap)
    colormap = cm.jet(heatmap)[:, :, :3]  # (H, W, 3)
    colormap = (colormap * 255).astype(np.uint8)

    # Blend with original image
    overlay = cv2.addWeighted(rgb_image, 1 - alpha, colormap, alpha, 0)

    return overlay

def cat_videos(heatmap, rgbs):
    heatmaps = iio.imread(heatmap)
    rgbs = iio.imread(rgbs)

    output_frames = []
    for i, (heatmap, rgb) in enumerate(zip(heatmaps, rgbs)):
        H, W, _ = rgb.shape
        target_pixel = np.array([np.argmin(heatmap[:,:,0]) % W, np.argmin(heatmap[:,:,0]) // W ])
        
        heatmap = generate_heatmap_from_points(np.stack([target_pixel] * 3, axis=0), heatmap.shape)
        heatmap_cat = get_heatmap_viz(rgb, heatmap[:, :, 0])
        output_frames.append(heatmap_cat)
    return output_frames
    
def process_heatmap(eval_dirs):
    for dir in eval_dirs:
        files = [f for f in os.listdir(dir) if f.lower().endswith(".mp4")]
        for episode in range(len(files)//2):
            heatmap_path = os.path.join(dir,f"eval_episode_{episode}_goal_gripper_proj.mp4")
            rgb_path = os.path.join(dir, f"eval_episode_{episode}.mp4")
            output_frames = cat_videos(heatmap_path, rgb_path)
            save_path = os.path.join(dir, f"eval_episode_{episode}_heatmap.mp4")
            
            save_video(save_path, output_frames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval path")
    parser.add_argument("--eval_dir", type=str, default="/home/haotian/lerobot/outputs/eval/2025-10-09/11-32-22_libero_diffusion",
                        help="eval_dir")
    
    parser.add_argument("--video_dir", type=str, default="all", nargs="+",)

    args = parser.parse_args()

    eval_dir_path = args.eval_dir
    video_dir = args.video_dir

    if video_dir == "all":
        eval_dirs = glob.glob(os.path.join(eval_dir_path,"videos","*"))
    else:
        eval_dirs = []
        for dir in video_dir:
            eval_dirs.append(os.path.join(eval_dir_path,"videos",dir))

    process_heatmap(eval_dirs)