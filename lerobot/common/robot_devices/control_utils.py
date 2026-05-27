# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

########################################################################################
# Utilities
########################################################################################


import logging
import json
import os
import time
import traceback
from contextlib import nullcontext
from copy import copy
from functools import cache

import numpy as np
import torch
from deepdiff import DeepDiff
from termcolor import colored
import pytorch3d.transforms as transforms
from scipy.spatial.transform import Rotation as R

from lerobot.common.datasets.image_writer import safe_stop_image_writer
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import get_features_from_robot
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device, has_method
from lerobot.common.utils.pointcloud_rgbd import render_top_down_custom, render_bottom_up_custom
from lerobot.common.utils.aloha_utils import ALOHA_CONFIGURATION, ALOHA_MODEL, VIRTUAL_CAMERA_MAPPING, forward_kinematics, render_and_overlay, setup_renderer
from PIL import Image
import sys
import termios
import tty
import select
import numpy as np


@cache
def get_droid_ik_wrapper():
    """Lazily construct the deoxys IK wrapper used by Droid/Script Franka control."""
    from deoxys.utils.ik_utils import IKWrapper

    return IKWrapper()


def droid_eef_to_joints(eef_action: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """Convert a Droid/Franka EEF action to an 8D joint-space action.

    Args:
        eef_action: (10,) [rot6d, xyz, gripper] in LeRobot convention.
        state: (8,) current state [7 joints, gripper].
    """
    eef_action = eef_action.squeeze()
    state = state.squeeze()

    rot6d = eef_action[:6]
    pos = eef_action[6:9]
    gripper = eef_action[9:10]

    ik_wrapper = get_droid_ik_wrapper()
    target_mat = transforms.rotation_6d_to_matrix(rot6d[None]).squeeze().detach().cpu().numpy()
    target_pos = pos.detach().cpu().numpy()
    current_joints = state[:7].detach().cpu().numpy().tolist()

    joint_positions = ik_wrapper.inverse_kinematics(
        ik_wrapper.model,
        ik_wrapper.data,
        target_mat,
        target_pos,
        current_joints,
    )
    return torch.cat(
        [
            torch.as_tensor(joint_positions, dtype=torch.float32, device=eef_action.device),
            gripper,
        ]
    )


def droid_ik_model_eef_pose(joints) -> tuple[np.ndarray, np.ndarray]:
    """Return the IK model's current controlled site pose for a 7D Franka joint state."""
    import mujoco

    ik_wrapper = get_droid_ik_wrapper()
    joints = np.asarray(joints, dtype=np.float64).reshape(7)
    ik_wrapper.data.qpos[:] = joints.tolist() + [0.04] * 2
    mujoco.mj_fwdPosition(ik_wrapper.model, ik_wrapper.data)
    gripper_site_id = ik_wrapper.model.site("grip_site").id
    pos = np.copy(ik_wrapper.data.site(gripper_site_id).xpos)
    mat = np.copy(ik_wrapper.data.site(gripper_site_id).xmat).reshape(3, 3)
    return mat, pos


def droid_delta_eef_to_joints(
    delta_pos,
    state: torch.Tensor,
    *,
    target_rot: torch.Tensor | np.ndarray | None = None,
    gripper_action: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a Cartesian delta in the IK model frame and return an 8D joint action.

    This avoids feeding Franka's reported O_T_EE pose directly into MuJoCo IK,
    which may use a different controlled site frame.
    """
    state = state.squeeze()
    current_mat, current_pos = droid_ik_model_eef_pose(state[:7].detach().cpu().numpy())
    target_pos = current_pos + np.asarray(delta_pos, dtype=np.float64).reshape(3)

    if target_rot is None:
        target_mat = torch.as_tensor(current_mat, dtype=state.dtype, device=state.device)
    elif isinstance(target_rot, torch.Tensor):
        target_mat = target_rot.to(device=state.device, dtype=state.dtype)
    else:
        target_mat = torch.as_tensor(target_rot, dtype=state.dtype, device=state.device)

    rot6d = transforms.matrix_to_rotation_6d(target_mat[None]).squeeze(0)
    pos = torch.as_tensor(target_pos, dtype=state.dtype, device=state.device)
    if gripper_action is None:
        gripper = state[-1:]
    else:
        gripper = torch.as_tensor([gripper_action], dtype=state.dtype, device=state.device).reshape(1)
    return droid_eef_to_joints(torch.cat([rot6d, pos, gripper]), state)


def print_droid_ik_diagnostics(robot, state: torch.Tensor, action: torch.Tensor | None = None, prefix: str = "[ik]"):
    """Print frame and joint-delta diagnostics for Droid/Script IK debugging."""
    state = state.squeeze()
    robot_rot, robot_pos = robot.robot_interface.last_eef_rot_and_pos
    model_rot, model_pos = droid_ik_model_eef_pose(state[:7].detach().cpu().numpy())

    msg = (
        f"{prefix} robot_eef_pos={np.round(robot_pos.squeeze(), 5).tolist()} "
        f"ik_model_grip_site_pos={np.round(model_pos, 5).tolist()} "
        f"frame_pos_delta={np.round(model_pos - robot_pos.squeeze(), 5).tolist()}"
    )
    if action is not None:
        joint_delta = action[:7].detach().cpu() - state[:7].detach().cpu()
        msg += f" joint_delta={np.round(joint_delta.numpy(), 6).tolist()}"
    print(msg)


def droid_pose_to_eef_action(
    target_pos,
    target_quat,
    gripper_action,
    *,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Build a LeRobot Droid EEF action [rot6d, xyz, gripper] from pose + gripper."""
    from deoxys.utils import transform_utils

    target_pos = np.array(target_pos, dtype=np.float32)
    target_quat = np.array(target_quat, dtype=np.float32)
    target_mat = torch.from_numpy(transform_utils.quat2mat(target_quat)).to(device=device, dtype=dtype)
    rot6d = transforms.matrix_to_rotation_6d(target_mat[None]).squeeze(0)
    pos = torch.as_tensor(target_pos, device=device, dtype=dtype)
    gripper = torch.as_tensor([gripper_action], device=device, dtype=dtype)
    return torch.cat([rot6d, pos, gripper])


def send_droid_eef_action(robot, eef_action: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
    """IK an EEF action and send it through the robot's joint-space send_action path."""
    if state is None:
        state = torch.tensor(
            list(robot._get_franka_joints()) + [robot._get_gripper_width()],
            dtype=torch.float32,
        )
    elif not isinstance(state, torch.Tensor):
        state = torch.as_tensor(state, dtype=torch.float32)

    joint_action = droid_eef_to_joints(eef_action.float(), state.float())
    robot.send_action(joint_action)
    return joint_action


def droid_robot_pose_to_joint_action(
    robot,
    target_pos,
    target_quat,
    state: torch.Tensor | None = None,
    gripper_action: float | None = None,
) -> torch.Tensor:
    """Build a joint action for a Franka-reported EEF target pose.

    The robot reports Franka O_T_EE, while deoxys IK controls MuJoCo's grip_site.
    This maps the desired robot-frame position into the current IK-site frame
    before solving IK.
    """
    if state is None:
        state = torch.tensor(
            list(robot._get_franka_joints()) + [robot._get_gripper_width()],
            dtype=torch.float32,
        )
    elif not isinstance(state, torch.Tensor):
        state = torch.as_tensor(state, dtype=torch.float32)

    _, robot_pos = robot.robot_interface.last_eef_rot_and_pos
    _, ik_pos = droid_ik_model_eef_pose(state[:7].detach().cpu().numpy())
    robot_to_ik_pos_offset = ik_pos - robot_pos.squeeze()
    target_ik_pos = np.asarray(target_pos, dtype=np.float64).reshape(3) + robot_to_ik_pos_offset
    if gripper_action is None:
        gripper_action = state[-1].item()
    eef_action = droid_pose_to_eef_action(
        target_ik_pos,
        target_quat,
        gripper_action,
        device=state.device,
        dtype=state.dtype,
    )
    return droid_eef_to_joints(eef_action, state)


def read_key(timeout_s: float | None = None):
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        if timeout_s is not None:
            ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
            if not ready:
                return None
        ch = sys.stdin.read(1)

        if ch == "\r" or ch == "\n":
            return "enter"

        if ch == "\x1b":  # arrow keys start escape sequence
            seq = sys.stdin.read(2)
            if seq == "[D":
                return "left"
            if seq == "[C":
                return "right"
            if seq == "[A":
                return "up"
            if seq == "[B":
                return "down"

        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def hold_pose_until_key(robot, target_pos, target_quat, prompt: str):
    print(prompt, end="", flush=True)
    while True:
        key = read_key(timeout_s=0.02)
        if key is not None:
            print(key)
            return key.strip().lower()

        state = torch.tensor(
            list(robot._get_franka_joints()) + [robot._get_gripper_width()],
            dtype=torch.float32,
        )
        joint_action = droid_robot_pose_to_joint_action(
            robot,
            target_pos,
            target_quat,
            state=state,
            gripper_action=state[-1].item(),
        )
        robot.send_action(joint_action)

def add_eef_pose(robot, real_joints):
    if robot.robot_type == "aloha":
        eef_pose, eef_pose_se3 = forward_kinematics(ALOHA_CONFIGURATION, real_joints)
        eef_pose = torch.cat([eef_pose, real_joints[-1][None]], axis=0).float()
    elif robot.robot_type in ["droid", "script"]:
        eef_rot, eef_pos = robot.robot_interface.last_eef_rot_and_pos
        rot_6d = transforms.matrix_to_rotation_6d(torch.from_numpy(eef_rot[None])).squeeze()
        trans = torch.from_numpy(eef_pos.squeeze())
        eef_pose = torch.cat([rot_6d, trans, real_joints[-1:]], axis=0).float()
    elif robot.robot_type == "franka_leap":
        eef_rot, eef_pos = robot.robot_interface.last_eef_rot_and_pos
        rot_6d = transforms.matrix_to_rotation_6d(torch.from_numpy(eef_rot[None])).squeeze()
        trans = torch.from_numpy(eef_pos.squeeze())
        # 6 rot + 3 trans + 16 hand joints
        eef_pose = torch.cat([rot_6d, trans, real_joints[7:]], axis=0).float()
    elif robot.robot_type == "dummy":
        eef_pose = torch.zeros(10, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown robot type {robot.robot_type}")
    return eef_pose


def smooth_pose_move_to(
    robot,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    step_m: float = 0.01,
    num_steps_per_waypoint: int = 20,
    num_additional_steps: int = 0,
):
    """Interpolate a Franka EEF target and send direct joint-space actions."""
    _, eef_pos = robot.robot_interface.last_eef_rot_and_pos
    current_pos = eef_pos.squeeze().copy()
    target_pos = np.array(target_pos, dtype=np.float64)
    target_quat = np.array(target_quat, dtype=np.float64)

    delta = target_pos - current_pos
    max_delta = np.linalg.norm(delta)
    num_waypoints = max(int(np.ceil(max_delta / step_m)), 1)
    num_repeats = max(int(num_steps_per_waypoint), 1)

    for i in range(num_waypoints):
        alpha = (i + 1) / num_waypoints
        alpha = 0.5 - 0.5 * np.cos(np.pi * alpha)
        waypoint_pos = current_pos + alpha * delta
        print(
            f"Step {i + 1}/{num_waypoints}: "
            f"Sending joint action for EEF {np.round(waypoint_pos, 4).tolist()}"
        )
        for _ in range(num_repeats):
            state = torch.tensor(
                list(robot._get_franka_joints()) + [robot._get_gripper_width()],
                dtype=torch.float32,
            )
            joint_action = droid_robot_pose_to_joint_action(
                robot,
                waypoint_pos,
                target_quat,
                state=state,
                gripper_action=state[-1].item(),
            )
            robot.send_action(joint_action)
            time.sleep(0.01)

    return max_delta, num_waypoints


def slow_close_gripper(robot, speed: int = 60, force: int = 100):
    """Close Robotiq more gently when the gripper API exposes speed control."""
    if hasattr(robot.robotiq_gripper, "goTo"):
        robot.robotiq_gripper.goTo(255, speed=speed, force=force)
    else:
        robot.robotiq_gripper.close()
    robot._last_gripper_action = robot.config.gripper_close_action


def get_auxiliary_zed_camera(robot):
    """Return the auxiliary ZED camera configured for the scripted grasp sequence."""
    candidate_names = ("cam_auxiliary", "camera_auxiliary", "cam_auxiliray", "camera_auxiliray")
    for name in candidate_names:
        if name in robot.cameras:
            return name, robot.cameras[name]

    raise KeyError(
        "Could not find the auxiliary camera. Expected one of "
        f"{candidate_names}, got {tuple(robot.cameras.keys())}."
    )


def read_zed_stereo_rgb(camera) -> dict[str, np.ndarray]:
    """Read synchronized left/right RGB images from one connected ZED camera."""
    if camera.__class__.__name__ != "ZedCamera":
        raise TypeError(f"Expected auxiliary camera to be a ZedCamera, got {camera.__class__.__name__}.")
    if not camera.is_connected:
        raise RuntimeError(f"ZedCamera({camera.serial_number}) is not connected.")

    import cv2
    import pyzed.sl as sl

    start_time = time.perf_counter()
    err = camera.camera.grab(camera._runtime_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise OSError(f"Can't grab stereo frame from ZedCamera({camera.serial_number}): {err}")

    stereo_images = {}
    for image_name, view in (("left", sl.VIEW.LEFT), ("right", sl.VIEW.RIGHT)):
        sl_image = sl.Mat()
        camera.camera.retrieve_image(sl_image, view)
        image = sl_image.get_data()[:, :, :3].copy()

        if camera.color_mode == "rgb":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif camera.color_mode != "bgr":
            raise ValueError(f"Expected color mode 'rgb' or 'bgr', got {camera.color_mode}.")

        h, w, _ = image.shape
        if h != camera.capture_height or w != camera.capture_width:
            raise OSError(
                f"Can't capture {image_name} image with expected height and width "
                f"({camera.capture_height} x {camera.capture_width}). ({h} x {w}) returned instead."
            )

        if camera.rotation is not None:
            image = cv2.rotate(image, camera.rotation)

        stereo_images[image_name] = image

    camera.logs["delta_timestamp_s"] = time.perf_counter() - start_time
    return stereo_images


def get_zed_intrinsics_and_baseline(camera) -> tuple[np.ndarray, float]:
    """Return rectified left-camera K and stereo baseline in meters for a ZED camera."""
    cam_info = camera.camera.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters
    left = calib.left_cam

    k = np.array(
        [
            [left.fx, 0.0, left.cx],
            [0.0, left.fy, left.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    baseline_m = None
    if hasattr(calib, "get_camera_baseline"):
        baseline_m = abs(float(calib.get_camera_baseline()))

    if baseline_m is None and hasattr(calib, "stereo_transform"):
        transform = calib.stereo_transform
        if hasattr(transform, "get_translation"):
            translation = transform.get_translation()
            if hasattr(translation, "get"):
                baseline_m = abs(float(translation.get()[0]))
            elif hasattr(translation, "x"):
                baseline_m = abs(float(translation.x))
            else:
                baseline_m = abs(float(translation[0]))

    if baseline_m is None:
        raise RuntimeError(f"Could not read ZED baseline for camera {camera.serial_number}.")

    if baseline_m > 1.0:
        baseline_m /= 1000.0
    return k, baseline_m


def make_contrast_depth_vis(depth: np.ndarray, max_depth: float | None = None) -> np.ndarray:
    """Convert metric depth to a high-contrast RGB visualization."""
    import cv2

    depth = depth.astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if max_depth is not None:
        valid &= depth < max_depth
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    lo, hi = np.percentile(depth[valid], [2, 98])
    depth_clip = np.clip(depth, lo, hi)
    depth_norm = (depth_clip - lo) / max(hi - lo, 1e-6)
    depth_u8 = ((1.0 - depth_norm) * 255.0).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    depth_u8 = clahe.apply(depth_u8)

    depth_bgr = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    depth_rgb = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2RGB)
    depth_rgb[~valid] = 0
    return depth_rgb


def warp_left_depth_vis_to_right(
    depth: np.ndarray,
    depth_vis_rgb: np.ndarray,
    fx: float,
    baseline_m: float,
) -> np.ndarray:
    """Warp a left-registered depth visualization into the right camera frame."""
    h, w = depth.shape
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.zeros_like(depth_vis_rgb)

    yy, xx = np.nonzero(valid)
    depth_values = depth[yy, xx]
    disparity = fx * baseline_m / depth_values
    right_x = np.rint(xx - disparity).astype(np.int32)
    in_bounds = (right_x >= 0) & (right_x < w)
    if not np.any(in_bounds):
        return np.zeros_like(depth_vis_rgb)

    yy = yy[in_bounds]
    right_x = right_x[in_bounds]
    left_x = xx[in_bounds]
    depth_values = depth_values[in_bounds]

    # Splat far-to-near so nearer surfaces win when multiple left pixels land
    # on the same right pixel.
    order = np.argsort(depth_values)[::-1]
    right_vis = np.zeros_like(depth_vis_rgb)
    right_vis[yy[order], right_x[order]] = depth_vis_rgb[yy[order], left_x[order]]
    return right_vis


@cache
def get_foundation_stereo_depth(
    ckpt: str = "/home/yinongh/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth",
    fs_dir: str = "/home/yinongh/FoundationStereo",
    valid_iters: int = 16,
    scale: float = 0.5,
):
    """Load FoundationStereo once and reuse it across scripted grasp runs."""
    frankapanda_root = "/home/yinongh/automate/real_world_visual_planning/frankapanda"
    if frankapanda_root not in sys.path:
        sys.path.insert(0, frankapanda_root)

    from zed_cams.foundation_stereo_depth import FoundationStereoDepth

    original_torch_load = torch.load

    def torch_load_with_pickle(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    try:
        torch.load = torch_load_with_pickle
        return FoundationStereoDepth(
            ckpt=ckpt,
            fs_dir=fs_dir,
            valid_iters=valid_iters,
            scale=scale,
            device="cuda",
        )
    finally:
        torch.load = original_torch_load


def compute_foundation_stereo_depth(
    images: dict[str, np.ndarray],
    camera,
) -> np.ndarray:
    """Run FoundationStereo on an auxiliary ZED pair and return metric depth."""
    k, baseline_m = get_zed_intrinsics_and_baseline(camera)
    left_rgb = images["left"]
    right_rgb = images["right"]

    fs_depth = get_foundation_stereo_depth()
    depth = fs_depth.infer_depth(
        left_rgb,
        right_rgb,
        fx=float(k[0, 0]),
        baseline_m=baseline_m,
        remove_invisible=True,
    )
    if depth.shape[:2] != left_rgb.shape[:2]:
        import cv2

        depth = cv2.resize(
            depth,
            (left_rgb.shape[1], left_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return depth


def depth_meters_to_uint16_mm(depth: np.ndarray) -> np.ndarray:
    """Convert float meter depth to a dataset-friendly uint16 millimeter image."""
    depth_mm = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0
    depth_mm = np.clip(depth_mm, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    return depth_mm[..., None]


def depth_rgb_to_camera_point_cloud(
    depth: np.ndarray,
    rgb: np.ndarray,
    k: np.ndarray,
    *,
    stride: int = 2,
    max_depth_m: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject a depth image and RGB image into a camera-frame point cloud."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = np.squeeze(depth, axis=-1)
    h, w = depth.shape

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    sample = np.zeros((h, w), dtype=bool)
    sample[::stride, ::stride] = True
    valid = np.isfinite(depth) & (depth > 0) & (depth <= max_depth_m) & sample
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    z = depth[valid]
    x = (xx[valid].astype(np.float32) - float(k[0, 2])) * z / float(k[0, 0])
    y = (yy[valid].astype(np.float32) - float(k[1, 2])) * z / float(k[1, 1])
    points = np.stack([x, y, z], axis=-1).astype(np.float32)
    colors = rgb[valid].astype(np.uint8)
    return points, colors


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to Nx3 points."""
    if points.shape[0] == 0:
        return points
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    return (np.asarray(transform, dtype=np.float64) @ points_h.T).T[:, :3].astype(np.float32)
def crop_rgb_depth_foreground_center(
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    crop_h: int = 480,
    crop_w: int = 640,
    margin: int = 20,
    center_x: float | None = None,
    center_y: float | None = None,
    trim_quantile: float = 0.02,
    debug: bool = False,
    debug_path: str = "debug_crop.png",
):
    rgb = np.asarray(rgb)
    depth = np.asarray(depth)
    h, w = depth.shape[:2]

    bg = np.nanmax(depth)
    valid = np.isfinite(depth) & (depth > 0) & (depth < bg - 1e-6)

    if np.any(valid):
        ys, xs = np.where(valid)

        y_min, y_max = np.quantile(ys, [trim_quantile, 1 - trim_quantile])
        x_min, x_max = np.quantile(xs, [trim_quantile, 1 - trim_quantile])

        bbox = (x_min, y_min, x_max, y_max)

        crop_center_y = 0.5 * (y_min + y_max)
        crop_center_x = 0.5 * (x_min + x_max)
    else:
        bbox = None
        crop_center_y, crop_center_x = h / 2, w / 2

    y0 = int(round(crop_center_y - crop_h / 2))
    x0 = int(round(crop_center_x - crop_w / 2))

    y0 = max(0, min(y0, h - crop_h))
    x0 = max(0, min(x0, w - crop_w))

    y1 = y0 + crop_h
    x1 = x0 + crop_w

    if debug:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(rgb)

        ax.add_patch(
            patches.Rectangle(
                (x0, y0),
                crop_w,
                crop_h,
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
                label="crop",
            )
        )

        if bbox is not None:
            bx0, by0, bx1, by1 = bbox
            ax.add_patch(
                patches.Rectangle(
                    (bx0, by0),
                    bx1 - bx0,
                    by1 - by0,
                    linewidth=2,
                    edgecolor="yellow",
                    facecolor="none",
                    label="foreground bbox",
                )
            )

        ax.scatter(
            [crop_center_x],
            [crop_center_y],
            c="cyan",
            s=60,
            marker="+",
            label="crop center",
        )

        if center_x is not None and center_y is not None:
            ax.scatter(
                [center_x],
                [center_y],
                c="red",
                s=60,
                marker="x",
                label="aligned center",
            )

        ax.set_title(
            f"crop_center=({crop_center_x:.1f}, {crop_center_y:.1f}), "
            f"crop=({x0}:{x1}, {y0}:{y1})"
        )
        ax.legend()
        ax.axis("off")

        fig.savefig(debug_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return rgb[y0:y1, x0:x1], depth[y0:y1, x0:x1]

def visualize_open3d_point_cloud(points: np.ndarray, colors: np.ndarray, window_name: str) -> None:
    """Render an RGB point cloud with the world-frame axes in Open3D."""
    import open3d as o3d

    if points.shape[0] == 0:
        print(f"[pointcloud] No points to visualize for {window_name}.")
        return

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(cloud)
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0.0, 0.0, 0.0])
    vis.add_geometry(world_frame)
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
    vis.run()
    vis.destroy_window()


def attach_auxiliary_observation_to_frame(observation: dict, robot, dataset: LeRobotDataset | None) -> None:
    """Attach one-shot auxiliary RGB/depth captures to dataset frames when declared."""
    if dataset is None:
        return

    initial_wrist_points = getattr(robot, "_initial_wrist_points_world", None)
    initial_wrist_points_key = "observation.points.initial_wrist_points_world"
    if initial_wrist_points is not None and initial_wrist_points_key in dataset.features:
        observation[initial_wrist_points_key] = initial_wrist_points

    left_rgb = getattr(robot, "_last_auxiliary_left_rgb", None)
    depth = getattr(robot, "_last_auxiliary_depth", None)
    if left_rgb is None or depth is None:
        return

    rgb_key = "observation.images.cam_auxiliary.left"
    depth_key = "observation.images.cam_auxiliary.depth"
    if rgb_key in dataset.features:
        observation[rgb_key] = left_rgb
    if depth_key in dataset.features:
        observation[depth_key] = depth_meters_to_uint16_mm(depth)


def command_gripper(robot, action, label, ticks=5, sleep_s=0.2):
    """Franka gripper convention: negative opens, nonnegative closes."""
    print(f"Commanding gripper {label}...")
    for tick in range(ticks):
        robot.robot_interface.gripper_control(action)
        print(f"  {label} command {tick + 1}/{ticks}")
        time.sleep(sleep_s)


WRIST_CAM_TO_GRIPPER = np.array(
    [
        [-0.00768086, -0.94557934, -0.32530096, 0.07294499],
        [0.99995759, -0.00891583, 0.00230583, -0.03177615],
        [-0.00508067, -0.32526946, 0.94560772, -0.08727812],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def visualize_world_and_wrist_camera_frames(robot) -> None:
    """Show world and wrist-camera frames in an interactive Open3D window."""
    import open3d as o3d

    eef_pose = np.asarray(robot._robot_ik_controller.eef_pose, dtype=np.float64).reshape(4, 4)
    if hasattr(robot, "_wrist_camera_extrinsics"):
        world_from_cam = np.asarray(robot._wrist_camera_extrinsics(eef_pose), dtype=np.float64).reshape(4, 4)
    else:
        world_from_cam = eef_pose @ WRIST_CAM_TO_GRIPPER

    print("[script] Opening interactive frame visualizer. Close the window to continue.")
    print("[script] Open3D axis colors: +X = red, +Y = green, +Z = blue.")
    print("[script] Large frame at origin is world; smaller frame/frustum is cam_wrist.")
    print("[script] T_world_cam_wrist:\n", np.array2string(world_from_cam, precision=6, suppress_small=True))
    print(
        "[script] cam_wrist axes in world frame: "
        f"x={np.round(world_from_cam[:3, 0], 6).tolist()} "
        f"y={np.round(world_from_cam[:3, 1], 6).tolist()} "
        f"z={np.round(world_from_cam[:3, 2], 6).tolist()}"
    )

    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12, origin=[0.0, 0.0, 0.0])
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.07, origin=[0.0, 0.0, 0.0])
    camera_frame.transform(world_from_cam)

    frustum_cam = np.array(
        [
            [0.0, 0.0, 0.0],
            [-0.04, -0.03, 0.08],
            [0.04, -0.03, 0.08],
            [0.04, 0.03, 0.08],
            [-0.04, 0.03, 0.08],
        ],
        dtype=np.float64,
    )
    frustum_world = transform_points(frustum_cam.astype(np.float32), world_from_cam).astype(np.float64)
    frustum = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(frustum_world),
        lines=o3d.utility.Vector2iVector(
            np.array(
                [
                    [0, 1],
                    [0, 2],
                    [0, 3],
                    [0, 4],
                    [1, 2],
                    [2, 3],
                    [3, 4],
                    [4, 1],
                ],
                dtype=np.int32,
            )
        ),
    )
    frustum.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1.0, 0.85, 0.0]]), (8, 1)))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="World frame and cam_wrist frame")
    vis.add_geometry(world_frame)
    vis.add_geometry(camera_frame)
    vis.add_geometry(frustum)
    vis.get_render_option().background_color = np.array([0.05, 0.05, 0.05])
    vis.run()
    vis.destroy_window()

def center_zoom_rgb(rgb_image, scale=3.5):
    import numpy as np
    from PIL import Image

    rgb = np.asarray(rgb_image)

    h, w = rgb.shape[:2]
    crop_h = int(round(h / scale))
    crop_w = int(round(w / scale))

    cy, cx = h // 2, w // 2
    y0 = max(0, cy - crop_h // 2)
    x0 = max(0, cx - crop_w // 2)
    y1 = y0 + crop_h
    x1 = x0 + crop_w

    cropped = rgb[y0:y1, x0:x1]

    pil = Image.fromarray(cropped)
    pil = pil.resize((w, h), Image.BILINEAR)

    return np.asarray(pil)


def normalize_depth_for_shape(depth, invalid_fill=0.0):
    """
    Normalize one depth image independently to [0, 1],
    keeping only relative shape information.

    Works for:
    - positive depth maps
    - negative IsaacGym-style depth maps
    - maps containing inf / -inf / nan
    """
    depth = depth.astype(np.float32)

    # valid pixels: finite only
    valid = np.isfinite(depth)

    # if nothing valid, return zeros
    if valid.sum() == 0:
        return np.full_like(depth, invalid_fill, dtype=np.float32)

    d = depth.copy()

    # normalize using only valid region
    d_valid = d[valid]
    d_min = d_valid.min()
    d_max = d_valid.max()

    # avoid divide-by-zero if all valid depths are identical
    if d_max - d_min < 1e-8:
        out = np.full_like(d, invalid_fill, dtype=np.float32)
        out[valid] = 1.0
        return out

    out = np.full_like(d, invalid_fill, dtype=np.float32)
    out[valid] = (d[valid] - d_min) / (d_max - d_min)

    return out.astype(np.float32)

def center_crop_rgb(rgb_image, crop_h=480, crop_w=640):
    import numpy as np

    rgb = np.asarray(rgb_image)
    h, w = rgb.shape[:2]

    cy, cx = h // 2, w // 2

    y0 = cy - crop_h // 2
    x0 = cx - crop_w // 2
    y1 = y0 + crop_h
    x1 = x0 + crop_w

    return rgb[y0:y1, x0:x1]

def closest_vertical_yaw_rot(cur_rot, vertical_rot):
    A = vertical_rot.T @ cur_rot

    yaw = np.arctan2(
        A[1, 0] - A[0, 1],
        A[0, 0] + A[1, 1],
    )

    c, s = np.cos(yaw), np.sin(yaw)
    yaw_rot = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ])

    return vertical_rot @ yaw_rot, yaw

def run_scripted_grasp_sequence(robot):
    # visualize_world_and_wrist_camera_frames(robot)
    # exit(0)
    # return
    record = {}
    skip_grasping = True
    if not skip_grasping:
        home_joints = np.array(
            [-0.05045543, -0.07240624, -0.03830516, -2.48442205, -0.05757582, 2.33608194, 0.73499261],
            dtype=np.float64,
        )
        print(f"[script] Resetting Franka to hard-coded home joints: {np.round(home_joints, 6).tolist()}")
        gripper_action = getattr(robot, "_last_gripper_action", robot.config.gripper_open_action)
        home_action = home_joints.tolist() + [gripper_action]
        max_home_steps = int(getattr(robot.config, "script_joint_wait_times", 100))
        home_tolerance = float(getattr(robot.config, "script_joint_convergence_tolerance", 1e-3))
        for step_idx in range(max_home_steps):
            robot.robot_interface.control(
                controller_type=robot.config.deoxys_controller_type,
                action=home_action,
                controller_cfg=robot.controller_cfg,
            )
            joint_error = float(np.max(np.abs(np.asarray(robot.robot_interface.last_q) - home_joints)))
            if joint_error < home_tolerance:
                print(f"[script] Home joint target reached in {step_idx + 1} steps, max_error={joint_error:.6f}")
                break
        else:
            print(f"[script] Home joint target not fully reached, max_error={joint_error:.6f}")

        target_quat = np.array(robot.config.target_quat, dtype=np.float64)
        target_pos = np.array(robot.config.approach_pos, dtype=np.float64)
        target_rot = transforms.quaternion_to_matrix(
                torch.tensor([target_quat[3], target_quat[0], target_quat[1], target_quat[2]], dtype=torch.float64)
            ).numpy()
        vertical_rot = target_rot.copy()
        robot._robot_ik_controller.control(
                target_pos=target_pos,
                target_rot=target_rot,
                grasping_action=getattr(robot, "_last_gripper_action", robot.config.gripper_open_action),
                # wait_times=int(getattr(robot.config, "script_joint_wait_times", 50)),
                wait_times=100,
                joint_threshold=float(getattr(robot.config, "script_joint_solution_threshold", 0.5)),
            )
        z_plane = robot._robot_ik_controller.eef_pose[2,3]
        while True:
            key = input("w/a/s/d to adjust XY, Enter to finish: ").strip().lower()

            if key in ("", "enter"):
                break
            target_pos = robot._robot_ik_controller.eef_pose[:3,3]
            if key == "a":
                target_pos[1] -= 0.001   # y -
            elif key == "d":
                target_pos[1] += 0.001   # y +
            elif key == "w":
                target_pos[0] -= 0.001   # x -
            elif key == "s":
                target_pos[0] += 0.001   # x +
            else:
                continue
            target_pos[2] = z_plane
            robot._robot_ik_controller.control(
                target_pos=target_pos,
                target_rot=target_rot,
                grasping_action=getattr(robot, "_last_gripper_action", robot.config.gripper_open_action),
                wait_times=100,
                joint_threshold=float(getattr(robot.config, "script_joint_solution_threshold", 0.5)),
            )
        
        command_gripper(robot,action = 1.0, label="close")
        # slow_close_gripper(robot)
        print("Press Enter when the gripper is at the insertion pose to record it...")
        print("Press 'r' to reorient to target_rot, Enter to record insertion pose...")

        while True:
            key = input().strip().lower()

            if key == "r":
                cur_pose = robot._robot_ik_controller.eef_pose
                cur_pos = cur_pose[:3, 3]
                cur_rot = cur_pose[:3, :3]

                target_rot, yaw = closest_vertical_yaw_rot(cur_rot, vertical_rot)
                import pdb;pdb.set_trace()

                robot._robot_ik_controller.control(
                    target_pos=target_rot,
                    target_rot=cur_pos,
                    grasping_action=getattr(robot, "_last_gripper_action", robot.config.gripper_open_action),
                    wait_times=10,
                    joint_threshold=float(getattr(robot.config, "script_joint_solution_threshold", 0.5)),
                )

                print(f"Reoriented vertical. yaw={yaw:.3f} rad. Press 'r' again or Enter to record.")

            elif key == "":
                break
        aligned_pose = robot._robot_ik_controller.eef_pose
        aligned_pos = aligned_pose[:3,3]
        aligned_rot = aligned_pose[:3,:3]
        # print("aligned_pose:\n", aligned_pose)
        # print("aligned_pos:\n", aligned_pos)
        # print("aligned_rot:\n", aligned_rot)
        record = {
            "aligned_pose": aligned_pose,
            "aligned_pos": aligned_pos,
            "aligned_rot": aligned_rot,
        }
        # # Lift Up the Gripper
        target_pos = aligned_pos.copy()
        target_pos[2] += 0.03
        target_pos[1] += np.random.uniform(-0.01, 0.01)
        target_pos[0] += np.random.uniform(-0.01, 0.01)
        # yaw_noise = np.random.uniform(-np.pi / 2, np.pi / 2)
        yaw_noise = np.pi/2
        yaw_quat_xyzw = np.array(
            [0.0, 0.0, np.sin(yaw_noise * 0.5), np.cos(yaw_noise * 0.5)],
            dtype=np.float64,
        )
        aligned_quat_xyzw = R.from_matrix(aligned_rot).as_quat()
        ctrl_tgt_quat_xyzw = R.from_quat(yaw_quat_xyzw) * R.from_quat(aligned_quat_xyzw)
        target_rot = ctrl_tgt_quat_xyzw.as_matrix()
        
    skip_initialization = True
    if not skip_initialization:
        total_init_steps = 100
        for i in range(total_init_steps):
            current_rot = robot._robot_ik_controller.eef_pose[:3,:3]
            current_pos = robot._robot_ik_controller.eef_pose[:3,3]
            # Next tgt pos is the interpolation between current pos and target pos, with a small step size to ensure smooth movement and better IK convergence
            if i < 50:
                next_tgt_pos = current_pos.copy()
                next_tgt_pos[2] += 0.001
                next_tgt_rot = current_rot.copy()
            else:
                next_tgt_pos = (target_pos - current_pos) / (total_init_steps-i) + current_pos
                next_tgt_rot, _, _, _ = robot._interpolate_rotation_matrix(
                    current_rot,
                    target_rot,
                    max_angle_step_deg=2.0,
                )
            robot._robot_ik_controller.control(
                    target_pos=next_tgt_pos,
                    target_rot=next_tgt_rot,
                    grasping_action=getattr(robot, "_last_gripper_action", robot.config.gripper_open_action),
                    wait_times=100,
                    joint_threshold=float(getattr(robot.config, "script_joint_solution_threshold", 0.5)),
                )
        init_pos = robot._robot_ik_controller.eef_pose[:3,3]
        init_rot = robot._robot_ik_controller.eef_pose[:3,:3]
        print("[script] Rendering initial wrist point cloud in world frame...")
        target_pos = init_pos.copy()
        target_pos[2] += 0.04 # Lift the gripper by 4cm to ensure the wrist camera has a clear view of the scene for the initial point cloud capture
        print("Lifting up the gripper")
        for i in range(10):
                current_rot = robot._robot_ik_controller.eef_pose[:3,:3]
                current_pos = robot._robot_ik_controller.eef_pose[:3,3]
                # Next tgt pos is the interpolation between current pos and target pos, with a small step size to ensure smooth movement and better IK convergence
                next_tgt_pos = (target_pos - current_pos) / (10-i) + current_pos
                next_tgt_rot, _, _, _ = robot._interpolate_rotation_matrix(
                    current_rot,
                    target_rot,
                    max_angle_step_deg=float(getattr(robot.config, "script_rot_max_angle_step_deg", 0.3)),
                )
                robot._robot_ik_controller.control(
                        target_pos=next_tgt_pos,
                        target_rot=next_tgt_rot,
                        grasping_action=getattr(robot, "_last_gripper_action", robot.config.gripper_open_action),
                        wait_times=100,
                        joint_threshold=float(getattr(robot.config, "script_joint_solution_threshold", 0.5)),
                    )
        wrist_camera = robot.cameras["cam_wrist"]
        wrist_images = read_zed_stereo_rgb(wrist_camera)
        wrist_depth = compute_foundation_stereo_depth(wrist_images, wrist_camera)
        wrist_k, _ = get_zed_intrinsics_and_baseline(wrist_camera)
        wrist_points_cam, wrist_colors = depth_rgb_to_camera_point_cloud(
                wrist_depth,
                wrist_images["left"],
                wrist_k,
                stride=2,
                max_depth_m=0.5,
            )

        cam_to_gripper = np.array(
                [
                    [-0.00768086, -0.94557934, -0.32530096, 0.07294499],
                    [0.99995759, -0.00891583, 0.00230583, -0.03177615],
                    [-0.00508067, -0.32526946, 0.94560772, -0.08727812],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
        world_from_gripper = np.asarray(robot._robot_ik_controller.eef_pose, dtype=np.float64)
        world_from_cam = world_from_gripper @ cam_to_gripper
        wrist_points_world = transform_points(wrist_points_cam, world_from_cam)
        # world_z_low_threshold = 0.0
        # world_z_threshold = 0.09
        world_z_low_threshold = 0.025
        world_z_threshold = 0.06

        keep = (wrist_points_world[:, 2] >= world_z_low_threshold) & (wrist_points_world[:, 2]<= world_z_threshold)

        wrist_points_world = wrist_points_world[keep]
        wrist_colors = wrist_colors[keep]
        robot._initial_wrist_points_world = wrist_points_world.astype(np.float32)
        robot._initial_wrist_points_world_colored = np.concatenate(
            [wrist_points_world.astype(np.float32), wrist_colors.astype(np.float32)],
            axis=1,
        )
        colored_pcd_path = os.path.expanduser("~/automate/lerobot/outputs/initial_wrist_points_world_colored.ply")
        os.makedirs(os.path.dirname(colored_pcd_path), exist_ok=True)
        try:
            import open3d as o3d

            colored_pcd = o3d.geometry.PointCloud()
            colored_pcd.points = o3d.utility.Vector3dVector(wrist_points_world.astype(np.float64))
            colored_pcd.colors = o3d.utility.Vector3dVector(wrist_colors.astype(np.float64) / 255.0)
            o3d.io.write_point_cloud(colored_pcd_path, colored_pcd)
            print(f"[script] Saved colored initial wrist point cloud to {colored_pcd_path}")
        except Exception as exc:
            print(f"[script] Failed to save colored initial wrist point cloud: {exc}")
        visualize_open3d_point_cloud(
            wrist_points_world,
            wrist_colors,
            "Initial wrist point cloud in world frame",
        )
        record["init_socket_pcd"] = robot._initial_wrist_points_world_colored
        init_socket_pcd = np.asarray(record["init_socket_pcd"], dtype=np.float32)
        init_points = init_socket_pcd[:, :3].copy()
        init_colors = init_socket_pcd[:, 3:].copy()
        rgb_init, depth_init = render_top_down_custom(
            torch.as_tensor(init_points, dtype=torch.float32),
            torch.as_tensor(init_colors[:, :3], dtype=torch.float32),
            center_x=record["aligned_pos"][0],
            center_y=record["aligned_pos"][1],
            H=720,
            W=1280,
            camera_height_offset=0.02,
            fov_deg=68.66,
            point_radius=5,
        )
        rgb_img = rgb_init.clone().detach().cpu().numpy()
        depth_img = depth_init.clone().detach().cpu().numpy()
        rgb_crop, depth_crop = crop_rgb_depth_foreground_center(
            rgb_img,
            depth_img,
            center_x=record["aligned_pos"][0],
            center_y=record["aligned_pos"][1],
            crop_h=640,
            crop_w=480,
            debug = True,
            debug_path="/home/yinongh/automate/lerobot/outputs/debug_initial_crop.png",
        )
        output_dir = "/home/yinongh/automate/lerobot/outputs"
        os.makedirs(output_dir, exist_ok=True)
        rgb_np = (np.clip(rgb_img, 0.0, 1.0) * 255).astype(np.uint8)
        rgb_crop = np.rot90(rgb_crop, k=1)
        depth_crop = np.rot90(depth_crop, k=1)
        rgb_crop_np = (np.clip(rgb_crop, 0.0, 1.0) * 255).astype(np.uint8)
        depth_normalized = (depth_img / max(float(depth_img.max()), 1e-8) * 255).astype(np.uint8)
        depth_crop_normalized = (depth_crop / max(float(depth_crop.max()), 1e-8) * 255).astype(np.uint8)
        
        Image.fromarray(rgb_np).save(f"{output_dir}/initial_socket_rgb.png")
        Image.fromarray(rgb_crop_np).save(f"{output_dir}/initial_socket_rgb_crop.png")
        Image.fromarray(depth_normalized).save(f"{output_dir}/initial_socket_depth.png")
        Image.fromarray(depth_crop_normalized).save(f"{output_dir}/initial_socket_depth_crop.png")
        print("Inspect the initial socket RGB and depth captures, then press Enter to continue...")
        input()
    skip_plug_photo = False
    if not skip_plug_photo:
        print("Initial Pose Achieved")
        time.sleep(3)
        target_rot = robot._robot_ik_controller.eef_pose[:3,:3]
        target_pos = np.array([0.485, -0.22, 0.25], dtype=np.float64)
        # Translate to take photo
        total_photo_steps = 5
        for i in range(total_photo_steps):
            current_rot = robot._robot_ik_controller.eef_pose[:3,:3]
            current_pos = robot._robot_ik_controller.eef_pose[:3,3]
            # Next tgt pos is the interpolation between current pos and target pos, with a small step size to ensure smooth movement and better IK convergence
            next_tgt_pos = (target_pos - current_pos) / (total_photo_steps-i) + current_pos
            next_tgt_rot, _, _, _ = robot._interpolate_rotation_matrix(
                current_rot,
                target_rot,
                max_angle_step_deg=float(getattr(robot.config, "script_rot_max_angle_step_deg", 0.3)),
            )
            robot._robot_ik_controller.control(
                    target_pos=next_tgt_pos,
                    target_rot=next_tgt_rot,
                    grasping_action=getattr(robot, "_last_gripper_action", robot.config.gripper_open_action),
                    wait_times=100,
                    joint_threshold=float(getattr(robot.config, "script_joint_solution_threshold", 0.5)),
                )
        # Taking Photo
        print("Taking auxiliary ZED stereo photo...")
    if 1:
        auxiliary_camera_name, auxiliary_camera = get_auxiliary_zed_camera(robot)
        auxiliary_images = read_zed_stereo_rgb(auxiliary_camera)
        robot._last_auxiliary_stereo_rgb = auxiliary_images
        robot._last_auxiliary_left_rgb = auxiliary_images["left"]
        robot._last_auxiliary_depth = compute_foundation_stereo_depth(auxiliary_images, auxiliary_camera)
        auxiliary_k, _ = get_zed_intrinsics_and_baseline(auxiliary_camera)
        auxiliary_points_cam, auxiliary_colors = depth_rgb_to_camera_point_cloud(
            robot._last_auxiliary_depth,
            robot._last_auxiliary_left_rgb,
            auxiliary_k,
            stride=4,
            max_depth_m=float(getattr(robot.config, "script_auxiliary_point_cloud_max_depth_m", 0.5)),
        )
        visualize_open3d_point_cloud(
            auxiliary_points_cam,
            auxiliary_colors,
            f"Auxiliary {auxiliary_camera_name} point cloud in camera frame",
        )
        rgb_img, depth_img = render_bottom_up_custom(
            torch.from_numpy(auxiliary_points_cam),
            torch.from_numpy(auxiliary_colors / 255.0),
            center_x=0.025,
            center_y=0.074,
            H=480,
            W=640,
            camera_height_offset=0.08,
            fov_deg=30,
            brightness_scale=1,
            point_radius=5
        )
        def save_rgb(rgb_image, img_name):
            import matplotlib.pyplot as plt
            plt.imsave(f"/home/yinongh/automate/lerobot/outputs/{img_name}.png", rgb_image)
        def save_depth_vis(depth_image, img_name):
            import numpy as np
            import matplotlib.pyplot as plt

            depth = np.asarray(depth_image)

            valid = np.isfinite(depth) & (depth > 0)
            if np.any(valid):
                vmin = np.percentile(depth[valid], 1)
                vmax = np.percentile(depth[valid], 99)
                depth_vis = np.clip((depth - vmin) / max(vmax - vmin, 1e-8), 0, 1)
            else:
                depth_vis = np.zeros_like(depth, dtype=np.float32)
            plt.imsave(
                f"/home/yinongh/automate/lerobot/outputs/{img_name}.png",
                depth_vis,
                cmap="viridis",
            )
        save_rgb(rgb_img, "processed_auxiliary_rgb.png")
        save_depth_vis(depth_img, "processed_auxiliary_depth.png")
        # zoomed_rgb = center_zoom_rgb(robot._last_auxiliary_left_rgb, scale=3.5)
        # cropped_rgb = center_crop_rgb(
        #     zoomed_rgb,
        #     crop_h=480,
        #     crop_w=640,
        # )
        # flipped_rgb = cropped_rgb[::-1, :, :].copy()
        # save_rgb(flipped_rgb, "processed_auxiliary_left_rgb")
        # zoomed_depth = center_zoom_rgb(robot._last_auxiliary_depth, scale=3.5)
        # cropped_depth = center_crop_rgb(zoomed_depth, crop_h=480, crop_w=640)
        # flipped_depth = cropped_depth[::-1, :].copy()
        # init_plug_depth = normalize_depth_for_shape(flipped_depth)
        # save_depth_vis(init_plug_depth, "processed_auxiliary_depth")
        # print(
        #     f"[script] Captured auxiliary depth and left RGB for dataset: "
        #     f"camera={auxiliary_camera_name} rgb={robot._last_auxiliary_left_rgb.shape} "
        #     f"depth={robot._last_auxiliary_depth.shape}"
        # )
        exit(0)
        target_pos = init_pos
        target_rot = init_rot
        for i in range(total_photo_steps):
            current_rot = robot._robot_ik_controller.eef_pose[:3,:3]
            current_pos = robot._robot_ik_controller.eef_pose[:3,3]
            # Next tgt pos is the interpolation between current pos and target pos, with a small step size to ensure smooth movement and better IK convergence
            next_tgt_pos = (target_pos - current_pos) / (total_photo_steps-i) + current_pos
            next_tgt_rot, _, _, _ = robot._interpolate_rotation_matrix(
                current_rot,
                target_rot,
                max_angle_step_deg=float(getattr(robot.config, "script_rot_max_angle_step_deg", 0.3)),
            )
            robot._robot_ik_controller.control(
                    target_pos=next_tgt_pos,
                    target_rot=next_tgt_rot,
                    grasping_action=getattr(robot, "_last_gripper_action", robot.config.gripper_open_action),
                    wait_times=100,
                    joint_threshold=float(getattr(robot.config, "script_joint_solution_threshold", 0.5)),
                )
        # Fine adjustment to ensure we are back to the initial pose
        for _ in range(5):
            robot._robot_ik_controller.control(
                    target_pos=init_pos,
                    target_rot=init_rot,
                    grasping_action=getattr(robot, "_last_gripper_action", robot.config.gripper_open_action),
                    wait_times=100,
                    joint_threshold=float(getattr(robot.config, "script_joint_solution_threshold", 0.5)),
                )


    record["init_EEF_pose"] = robot._robot_ik_controller.eef_pose
    record["init_socket_depth_img"] = normalize_depth_for_shape(depth_crop)
    return record

def log_control_info(robot: Robot, dt_s, episode_index=None, frame_index=None, fps=None):
    log_items = []
    if episode_index is not None:
        log_items.append(f"ep:{episode_index}")
    if frame_index is not None:
        log_items.append(f"frame:{frame_index}")

    def log_dt(shortname, dt_val_s):
        nonlocal log_items, fps
        info_str = f"{shortname}:{dt_val_s * 1000:5.2f} ({1 / dt_val_s:3.1f}hz)"
        if fps is not None:
            actual_fps = 1 / dt_val_s
            if actual_fps < fps - 1:
                info_str = colored(info_str, "yellow")
        log_items.append(info_str)

    # total step time displayed in milliseconds and its frequency
    log_dt("dt", dt_s)

    # TODO(aliberts): move robot-specific logs logic in robot.print_logs()
    if robot.robot_type not in ["stretch", "droid", "script", "dummy", "franka_leap"]:
        for name in robot.leader_arms:
            key = f"read_leader_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRlead", robot.logs[key])

        for name in robot.follower_arms:
            key = f"write_follower_{name}_goal_pos_dt_s"
            if key in robot.logs:
                log_dt("dtWfoll", robot.logs[key])

            key = f"read_follower_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRfoll", robot.logs[key])

        for name in robot.cameras:
            key = f"read_camera_{name}_dt_s"
            if key in robot.logs:
                log_dt(f"dtR{name}", robot.logs[key])

    info_str = " ".join(log_items)
    # logging.info(info_str)


@cache
def is_headless():
    """Detects if python is running without a monitor."""
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True


def predict_action(observation, policy, device, use_amp):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            if type(observation[name]) == str: observation[name] = [observation[name]]; continue
            if "image" in name:
                if observation[name].dtype == torch.uint8:
                    observation[name] = observation[name].type(torch.float32) / 255
                elif observation[name].dtype == torch.uint16: # depth
                    observation[name] = observation[name].type(torch.float32) / 1000.
                else:
                    raise NotImplementedError
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        # Compute the next action with the policy
        # based on the current observation
        action, action_eef = policy.select_action(observation)

        # Remove batch dimension
        action, action_eef = action.squeeze(0), action_eef.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")
        action_eef = action_eef.to("cpu")

    return action, action_eef


def init_keyboard_listener():
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        listener = None
        return listener, events

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
                print("Escape key pressed. Stopping data recording...")
                events["stop_recording"] = True
                events["exit_early"] = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


def warmup_record(
    robot,
    events,
    enable_teleoperation,
    warmup_time_s,
    display_data,
    fps,
):
    control_loop(
        robot=robot,
        control_time_s=warmup_time_s,
        display_data=display_data,
        events=events,
        fps=fps,
        teleoperate=enable_teleoperation,
    )


def record_episode(
    robot,
    dataset,
    events,
    episode_time_s,
    display_data,
    policy,
    fps,
    single_task,
):
    control_loop(
        robot=robot,
        control_time_s=episode_time_s,
        display_data=display_data,
        dataset=dataset,
        events=events,
        policy=policy,
        fps=fps,
        teleoperate=policy is None,
        single_task=single_task,
    )

def get_camera_names_from_observation(observation):
    """Get the camera names matching a given pattern from the observation, exclude the wrist camera."""
    return [s for s in observation.keys() if s.startswith('observation.images.cam_') and s.endswith('.color') and 'wrist' not in s]

def compute_goal_prediction(policy, policy_cfg, single_task, observation):
    if not (hasattr(policy_cfg, "enable_goal_conditioning") and policy_cfg.enable_goal_conditioning):
        return observation

    # Generate new goal prediction when queue is empty
    # This code is specific to diffusion policy / kinect :(
    if hasattr(policy, "_queues") and len(policy._queues[policy.act_key]) == 0:
        # Gather observations from all configured cameras
        camera_obs = {}

        # Gather camera observations and apply phantomize if needed
        for cam_name in policy.high_level.camera_names:
            rgb_key = f"observation.images.{cam_name}.color"
            depth_key = f"observation.images.{cam_name}.transformed_depth"

            # Validate camera data exists
            if rgb_key not in observation or depth_key not in observation:
                raise ValueError(
                    f"Required camera observation '{cam_name}' not found. "
                    f"Available keys: {policy.high_level.camera_names}"
                )
            camera_obs[cam_name] = {
                "rgb": observation[rgb_key].numpy(),
                "depth": observation[depth_key].numpy().squeeze()
            }

        # Get dict of projections for all cameras
        gripper_projs = policy.high_level.predict_and_project(
            single_task, camera_obs,
            robot_type=policy.config.robot_type,
            robot_kwargs={"observation.state": observation["observation.state"]}
        )  # Returns dict[str, np.ndarray]

        # Store as dict of tensors
        for cam_name, proj in gripper_projs.items():
            policy.latest_gripper_proj[cam_name] = torch.from_numpy(proj)

    # Add goal projection to each camera observation
    for cam_name in policy.high_level.camera_names:
        observation[f"observation.images.{cam_name}.goal_gripper_proj"] = policy.latest_gripper_proj[cam_name]
    return observation

def get_phantomized_observation(policy, policy_cfg, camera_names, observation):
    if not(hasattr(policy_cfg, "phantomize") and policy_cfg.phantomize):
        return observation

    # Setup renderer once for all cameras if using phantomize
    if policy.renderer is None:
        intrinsics_txts, extrinsics_txts, virtual_camera_names = [], [], []
        for cam_name in camera_names:
            cam_name_ = cam_name.split('.')[2] # assuming camera names follow convention of `observation.images.<cam_name>`
            intrinsics_txts.append(f"lerobot/scripts/{policy.calibration_data[cam_name_]['intrinsics']}")
            extrinsics_txts.append(f"lerobot/scripts/{policy.calibration_data[cam_name_]['extrinsics']}")
            virtual_camera_names.append(VIRTUAL_CAMERA_MAPPING[cam_name_])

        # Get image dimensions from first camera
        height, width, _ = observation[camera_names[0]].numpy().shape

        # Setup renderer with all cameras at once
        policy.renderer = setup_renderer(
            ALOHA_MODEL,
            intrinsics_txts,
            extrinsics_txts,
            policy.downsample_factor,
            width,
            height,
            virtual_camera_names
        )

    state = observation["observation.state"].numpy()
    for cam_name in camera_names:
        # Overlay RGB with rendered robot
        render = render_and_overlay(
            policy.renderer,
            ALOHA_MODEL,
            state,
            observation[cam_name].numpy().copy(),
            policy.downsample_factor,
            VIRTUAL_CAMERA_MAPPING[cam_name.split('.')[2]],
        )
        observation[cam_name] = torch.from_numpy(render)
    return observation

@safe_stop_image_writer
def control_loop(
    robot,
    control_time_s=None,
    teleoperate=False,
    display_data=False,
    dataset: LeRobotDataset | None = None,
    events=None,
    policy: PreTrainedPolicy = None,
    fps: int | None = None,
    single_task: str | None = None,
):
    # TODO(rcadene): Add option to record logs
    if not robot.is_connected:
        robot.connect()

    if events is None:
        events = {"exit_early": False}

    if control_time_s is None:
        control_time_s = float("inf")

    if teleoperate and policy is not None:
        raise ValueError("When `teleoperate` is True, `policy` should be None.")

    if dataset is not None and single_task is None:
        raise ValueError("You need to provide a task as argument in `single_task`.")

    if dataset is not None and fps is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset['fps']} != {fps}).")

    timestamp = 0
    start_episode_t = time.perf_counter()

    # Controls starts, if policy is given it needs cleaning up
    if policy is not None:
        policy.reset()

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if not getattr(robot, "_scripted_grasp_sequence_done", False):
            robot._scripted_grasp_sequence_done = True
            robot._scripted_insert_meta_data = run_scripted_grasp_sequence(robot)
        insert_meta_data = getattr(robot, "_scripted_insert_meta_data", None)

        observation, action = robot.teleop_step(record_data=True, insert_meta_data=insert_meta_data)
        attach_auxiliary_observation_to_frame(observation, robot, dataset)

        if policy is not None and getattr(policy, "_current_vis_frame", None) is not None:
            # Each policy declares the observation key for its visualization frame
            # via cfg.vis_obs_key; we just route the (H, W*v, 3) uint8 RGB array
            # to that key here without knowing what policy produced it.
            vis_key = getattr(getattr(policy, "config", None), "vis_obs_key", None) \
                or getattr(policy, "vis_obs_key", None)
            if vis_key:
                import torch as _torch
                observation[vis_key] = _torch.from_numpy(policy._current_vis_frame)

        if dataset is not None:
            frame = {**observation, **action, "task": single_task}
            dataset.add_frame(frame)
        if fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - start_loop_t
        log_control_info(robot, dt_s, fps=fps)

        timestamp = time.perf_counter() - start_episode_t
        if events["exit_early"]:
            events["exit_early"] = False
            break


def reset_environment(robot, events, reset_time_s, fps, teleoperate=True):
    # TODO(rcadene): refactor warmup_record and reset_environment
    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()

    control_loop(
        robot=robot,
        control_time_s=reset_time_s,
        events=events,
        fps=fps,
        teleoperate=teleoperate,
    )


def stop_recording(robot, listener, display_data):
    robot.disconnect()

    if not is_headless() and listener is not None:
        listener.stop()


def sanity_check_dataset_name(repo_id, policy_cfg):
    _, dataset_name = repo_id.split("/")
    # either repo_id doesnt start with "eval_" and there is no policy
    # or repo_id starts with "eval_" and there is a policy

    # Check if dataset_name starts with "eval_" but policy is missing
    if dataset_name.startswith("eval_") and policy_cfg is None:
        raise ValueError(
            f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided ({policy_cfg.type})."
        )

    # Check if dataset_name does not start with "eval_" but policy is provided
    if not dataset_name.startswith("eval_") and policy_cfg is not None:
        raise ValueError(
            f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided ({policy_cfg.type})."
        )


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, use_videos: bool
) -> None:
    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, get_features_from_robot(robot, use_videos)),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )
