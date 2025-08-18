"""
Shared utilities for dataset creation scripts to reduce code duplication.
"""
import numpy as np


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
        distances = np.linalg.norm(pixel_coords - target_point, axis=-1)  # (height, width)
        goal_image[:, :, i] = distances

    # Apply square root transformation for steeper near-target gradients
    goal_image = (np.sqrt(goal_image / max_distance) * 255)
    goal_image = np.clip(goal_image, 0, 255).astype(np.uint8)
    return goal_image


def project_points_to_image(points_3d, intrinsic_matrix):
    """
    Project 3D points to 2D image coordinates using camera intrinsics.

    Args:
        points_3d (np.ndarray): Nx3 array of 3D points in camera frame
        intrinsic_matrix (np.ndarray): 3x3 camera intrinsic matrix

    Returns:
        np.ndarray: Nx2 array of 2D pixel coordinates
    """
    points_2d_hom = intrinsic_matrix @ points_3d.T
    points_2d = points_2d_hom[:2].T / points_2d_hom[2].T[:, np.newaxis]
    return points_2d


def get_subgoal_indices_from_gripper_actions(gripper_actions):
    """
    Get subgoal indices based on gripper action changes.
    
    Subgoal indices are timesteps at which gripper action changes from open->close or vice versa
    and the last frame of the demo.
    
    Args:
        gripper_actions (np.ndarray): Array of gripper actions
        
    Returns:
        np.ndarray: Array of subgoal indices
    """
    subgoal_indices = np.where(np.diff(gripper_actions) != 0)[0] + 1
    if len(subgoal_indices) == 0 or subgoal_indices[-1] != len(gripper_actions) - 1:
        subgoal_indices = np.concatenate([subgoal_indices, [len(gripper_actions) - 1]])
    return subgoal_indices