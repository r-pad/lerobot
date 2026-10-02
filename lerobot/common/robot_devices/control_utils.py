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
import warnings
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
from lerobot.common.utils.agos_canonical import (
    green_mask,
    agos_figure,
    canonical_normalize,
    canonical_views,
    paper_overlay,
    render_plug_canonical,
    render_socket_canonical,
    world_points_to_fingertip_frame,
)
from lerobot.common.utils.aloha_utils import ALOHA_CONFIGURATION, ALOHA_MODEL, VIRTUAL_CAMERA_MAPPING, forward_kinematics, render_and_overlay, setup_renderer
from PIL import Image
import sys
import termios
import tty
import select
import numpy as np

# Silence noisy third-party warnings from FoundationStereo / DINOv2 (torch hub).
warnings.filterwarnings("ignore", message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated", category=FutureWarning)
warnings.filterwarnings("ignore", message=r"xFormers is (disabled|not available)", category=UserWarning)



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
    original_hub_load = torch.hub.load

    def torch_load_with_pickle(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    def hub_load_prefer_cache(repo_or_dir, model, *args, **kwargs):
        # FoundationStereo builds its DINOv2 backbone with torch.hub.load("facebookresearch/dinov2", ...).
        # Even with the repo cached, torch.hub asks GitHub for the default branch first, so a GitHub
        # outage (e.g. HTTP 504) crashes the run. Load the cached copy directly when it exists.
        cached = os.path.join(torch.hub.get_dir(), "facebookresearch_dinov2_main")
        if repo_or_dir == "facebookresearch/dinov2" and os.path.isfile(os.path.join(cached, "hubconf.py")):
            kwargs.pop("source", None)
            kwargs.pop("trust_repo", None)
            kwargs.pop("force_reload", None)
            return original_hub_load(cached, model, *args, source="local", **kwargs)
        return original_hub_load(repo_or_dir, model, *args, **kwargs)

    try:
        torch.load = torch_load_with_pickle
        torch.hub.load = hub_load_prefer_cache
        return FoundationStereoDepth(
            ckpt=ckpt,
            fs_dir=fs_dir,
            valid_iters=valid_iters,
            scale=scale,
            device="cuda",
        )
    finally:
        torch.load = original_torch_load
        torch.hub.load = original_hub_load


def compute_foundation_stereo_depth(
    images: dict[str, np.ndarray],
    camera,
    scale: float | None = None,
) -> np.ndarray:
    """Run FoundationStereo on an auxiliary ZED pair and return metric depth.

    ``scale`` overrides the model's input downscale for this call (1.0 = full resolution).
    """
    k, baseline_m = get_zed_intrinsics_and_baseline(camera)
    left_rgb = images["left"]
    right_rgb = images["right"]

    fs_depth = get_foundation_stereo_depth()
    default_scale = fs_depth.scale
    if scale is not None:
        fs_depth.scale = float(scale)
    try:
        depth = fs_depth.infer_depth(
            left_rgb,
            right_rgb,
            fx=float(k[0, 0]),
            baseline_m=baseline_m,
            remove_invisible=True,
        )
    finally:
        fs_depth.scale = default_scale
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

def depth_rgb_to_auxiliary_camera_point_cloud(
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
    stride = max(1, int(stride))
    dense_stride = max(1, stride // 2)
    far_sample = (yy % stride == 0) & (xx % stride == 0)
    dense_sample = (yy % dense_stride == 0) & (xx % dense_stride == 0)
    sample = np.where(depth < 0.25, True, np.where(depth < 0.3, dense_sample, far_sample))
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
        import matplotlib
        matplotlib.use("Agg", force=True)
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

def visualize_open3d_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    window_name: str,
    center_on_cloud: bool = False,
) -> None:
    """Render an RGB point cloud with the world-frame axes in Open3D.

    With ``center_on_cloud`` the default view is kept but re-centred on the cloud.
    """
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
    if center_on_cloud:
        vis.get_view_control().set_lookat(np.median(points, axis=0))  # median: robust to stray points
    vis.run()
    vis.destroy_window()


def attach_auxiliary_observation_to_frame(observation: dict, robot, dataset: LeRobotDataset | None) -> None:
    """Attach one-shot auxiliary RGB/depth captures to dataset frames when declared."""
    if dataset is None:
        return

    initial_wrist_points = getattr(robot, "_initial_wrist_points_world", None)
    initial_wrist_points_key = "observation.points.initial_wrist_points_world"
    # A skipped scan is recorded as a single NaN point (the feature must be present in every frame).
    missing_points = np.full((1, 3), np.nan, dtype=np.float32)
    if initial_wrist_points_key in dataset.features:
        observation[initial_wrist_points_key] = (
            initial_wrist_points if initial_wrist_points is not None else missing_points
        )

    plug_points = getattr(robot, "_plug_points_fingertip", None)
    plug_points_key = "observation.points.init_plug_points_fingertip"
    if plug_points_key in dataset.features:
        # AGOS stores the plug cloud in the fingertip frame (init_plug_points_fingertip).
        observation[plug_points_key] = (
            plug_points if plug_points is not None else missing_points
        )

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

def save_rgb(rgb_image, img_name):
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            plt.imsave(f"/home/yinongh/automate/lerobot/outputs/{img_name}.png", rgb_image)
def save_depth_vis(depth_image, img_name):
            import numpy as np
            import matplotlib
            matplotlib.use("Agg", force=True)
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

FRANKA_JOINT7_LIMIT = 2.8973


def twist_angle_deg(rot: R, axis: np.ndarray) -> float:
    """Signed twist of ``rot`` about unit ``axis`` (swing-twist decomposition), in degrees."""
    q = rot.as_quat()  # xyzw
    twist = 2.0 * np.arctan2(float(np.dot(q[:3], axis)), float(q[3]))
    return float(np.rad2deg((twist + np.pi) % (2.0 * np.pi) - np.pi))


def sample_agos_initial_pose(robot, aligned_pos, aligned_rot, max_attempts=200):
    """Sample the initial EEF pose with the AGOS (AutoMateTaskAGOS) distribution.

    Mirrors ``AutoMateTaskTiltedInsertion._randomize_gripper_pose`` in third_party/AGOS:
    lift along the socket axis, add uniform world-frame position noise, and rotate about
    the fingertip by ``delta = q_rpy * q_axial`` (left-multiplied, i.e. world frame), where
    q_rpy ~ U(+-init_rot_noise_deg) per axis (Isaac Gym quat_from_euler_xyz convention) and
    q_axial is a spin about the insertion axis ~ U(+-init_axial_spin_deg). As in the AGOS
    wrist budget, samples whose total twist exceeds ``init_max_total_twist_deg`` or would
    push joint 7 past its limit are rejected.
    """
    cfg = robot.config
    aligned_pos = np.asarray(aligned_pos, dtype=np.float64)
    aligned_rot = np.asarray(aligned_rot, dtype=np.float64)
    # The socket sits flat on the table, so the insertion (disassembly) axis is world +z.
    insertion_axis = np.array([0.0, 0.0, 1.0])
    base_pos = aligned_pos + insertion_axis * cfg.init_lift_height
    # Joint 7 rotates about the EEF z axis, so a world twist phi about the insertion axis
    # changes joint 7 by about phi * <insertion_axis, eef_z>.
    j7_per_twist = float(np.dot(insertion_axis, aligned_rot[:, 2]))
    j7_now = float(np.asarray(robot._robot_ik_controller.robot_interface.last_q)[6])
    j7_limit = FRANKA_JOINT7_LIMIT - cfg.init_joint7_limit_margin
    pos_noise = np.asarray(cfg.init_pos_noise, dtype=np.float64)
    rot_noise = np.deg2rad(np.asarray(cfg.init_rot_noise_deg, dtype=np.float64))
    rng = np.random.default_rng()

    for attempt in range(max_attempts):
        pos_offset = rng.uniform(-1.0, 1.0, 3) * pos_noise
        rpy = rng.uniform(-1.0, 1.0, 3) * rot_noise
        # scipy lowercase "xyz" is extrinsic: Rz(yaw) Ry(pitch) Rx(roll), same as Isaac Gym.
        q_rpy = R.from_euler("xyz", rpy)
        spin_deg = rng.uniform(-cfg.init_axial_spin_deg, cfg.init_axial_spin_deg)
        q_axial = R.from_rotvec(np.deg2rad(spin_deg) * insertion_axis)
        delta = q_rpy * q_axial
        total_twist_deg = twist_angle_deg(delta, insertion_axis)
        if abs(total_twist_deg) > cfg.init_max_total_twist_deg:
            continue
        j7_pred = j7_now + np.deg2rad(total_twist_deg) * j7_per_twist
        if abs(j7_pred) > j7_limit:
            continue
        target_pos = base_pos + pos_offset
        target_rot = (delta * R.from_matrix(aligned_rot)).as_matrix()
        info = {
            "pos_offset": pos_offset,
            "rpy_deg": np.rad2deg(rpy),
            "axial_spin_deg": spin_deg,
            "total_twist_deg": total_twist_deg,
            "joint7_pred": j7_pred,
        }
        print(
            f"[script] AGOS initial pose (attempt {attempt + 1}): pos_offset={np.round(pos_offset * 1e3, 1).tolist()} mm, "
            f"rpy={np.round(info['rpy_deg'], 1).tolist()} deg, spin={spin_deg:.1f} deg, twist={total_twist_deg:.1f} deg"
        )
        return target_pos, target_rot, info

    print("[script] No AGOS initial pose satisfied the wrist budget; using the lifted aligned pose.")
    return base_pos, aligned_rot.copy(), {"pos_offset": np.zeros(3), "rpy_deg": np.zeros(3), "axial_spin_deg": 0.0}


def untilted_rotation(current_rot: np.ndarray, axis_world: np.ndarray) -> np.ndarray:
    """Smallest rotation of ``current_rot`` that points its z axis (plug axis) along ``axis_world``.

    Removes tilt only; the spin about the axis is kept (no extra joint-7 rotation).
    """
    current_rot = np.asarray(current_rot, dtype=np.float64)
    z = current_rot[:, 2] / np.linalg.norm(current_rot[:, 2])
    a = np.asarray(axis_world, dtype=np.float64) / np.linalg.norm(axis_world)
    cross = np.cross(z, a)
    sin, cos = np.linalg.norm(cross), float(np.dot(z, a))
    if sin < 1e-9:
        return current_rot.copy() if cos > 0 else (R.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0])).as_matrix() @ current_rot)
    swing = R.from_rotvec(cross / sin * np.arctan2(sin, cos)).as_matrix()
    return swing @ current_rot


def rotation_steps(robot, target_rot, min_steps: int) -> int:
    """Interpolation steps needed to reach ``target_rot`` at script_rot_max_angle_step_deg per step."""
    current_rot = np.asarray(robot._robot_ik_controller.eef_pose, dtype=np.float64)[:3, :3]
    angle_deg = np.rad2deg((R.from_matrix(target_rot) * R.from_matrix(current_rot).inv()).magnitude())
    max_step = float(getattr(robot.config, "script_rot_max_angle_step_deg", 2.0))
    return max(int(min_steps), int(np.ceil(angle_deg / max(max_step, 1e-3))) + 2)


def move_to_pose_interpolated(robot, target_pos, target_rot, num_steps=5):
    """Interpolate from the measured pose to the target in ``num_steps`` IK commands."""
    target_pos = np.asarray(target_pos, dtype=np.float64)
    for i in range(num_steps):
        current_rot = robot._robot_ik_controller.eef_pose[:3, :3]
        current_pos = robot._robot_ik_controller.eef_pose[:3, 3]
        next_tgt_pos = (target_pos - current_pos) / (num_steps - i) + current_pos
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


def depth_edge_mask(depth: np.ndarray, rel_thresh: float = 0.02, window: int = 5) -> np.ndarray:
    """True where depth is valid and locally smooth.

    Stereo depth smears "flying pixels" along camera rays at depth discontinuities; those
    show up as ghost layers when views are fused. A pixel is dropped when the depth range in
    its ``window`` neighbourhood exceeds ``rel_thresh`` * depth (holes count as edges).
    """
    from scipy.ndimage import maximum_filter, minimum_filter

    depth = np.nan_to_num(np.asarray(depth, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if depth.ndim == 3:
        depth = depth[..., 0]
    local_range = maximum_filter(depth, size=window) - minimum_filter(depth, size=window)
    return (depth > 0) & (local_range <= rel_thresh * depth)


def _views_to_fingertip(cam_views, fingertip_poses, world_from_aux):
    return [
        world_points_to_fingertip_frame(transform_points(points_cam, world_from_aux), pose).astype(np.float64)
        for points_cam, pose in zip(cam_views, fingertip_poses)
    ]


def multiview_alignment_error(views_ft, max_dist=0.005, sample=2000, seed=0):
    """Mean distance (m) from each view's points to the nearest point of the other views."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    dists = []
    for i, view in enumerate(views_ft):
        others = [v for j, v in enumerate(views_ft) if j != i and len(v)]
        if not len(view) or not others:
            continue
        pts = view[rng.choice(len(view), min(sample, len(view)), replace=False)]
        d, _ = cKDTree(np.concatenate(others)).query(pts, distance_upper_bound=max_dist)
        dists.append(d[np.isfinite(d)])
    dists = np.concatenate(dists) if dists else np.zeros(0)
    return float(dists.mean()) if dists.size else float("nan")


def refine_aux_rotation_icp(cam_views, fingertip_poses, world_from_aux, iters=30, max_dist=0.004, sample=3000):
    """Refine world_from_aux rotation (3 DoF) so all plug views overlap in the fingertip frame.

    The views only differ by fingertip translation, so the camera translation shifts every view
    equally and only the rotation decides how well they overlap. ICP-style Gauss-Newton: for each
    view, residual r = p_ft - q (q = nearest point of the other views, within max_dist), with
    p_ft = R_ft^T (R p_cam + t - t_ft) and an update R <- exp(delta) R, so dr/d delta = -R_ft^T [R p_cam]x.
    """
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(0)
    world_from_aux = world_from_aux.copy()
    samples = [v[rng.choice(len(v), min(sample, len(v)), replace=False)] for v in cam_views]
    for _ in range(int(iters)):
        views_ft = _views_to_fingertip(cam_views, fingertip_poses, world_from_aux)
        rot = world_from_aux[:3, :3]
        jtj, jtr, used = np.zeros((3, 3)), np.zeros(3), 0
        for i, pts_cam in enumerate(samples):
            others = [v for j, v in enumerate(views_ft) if j != i and len(v)]
            if not len(pts_cam) or not others:
                continue
            ft_rot = np.asarray(fingertip_poses[i])[:3, :3]
            a = pts_cam.astype(np.float64) @ rot.T  # R p_cam
            p_ft = _views_to_fingertip([pts_cam], [fingertip_poses[i]], world_from_aux)[0]
            d, idx = cKDTree(np.concatenate(others)).query(p_ft, distance_upper_bound=max_dist)
            ok = np.isfinite(d)
            if not ok.any():
                continue
            q = np.concatenate(others)[idx[ok]]
            r = p_ft[ok] - q
            # J_k = -R_ft^T [a_k]x  (3x3 per point)
            ax = np.zeros((ok.sum(), 3, 3))
            ak = a[ok]
            ax[:, 0, 1], ax[:, 0, 2] = -ak[:, 2], ak[:, 1]
            ax[:, 1, 0], ax[:, 1, 2] = ak[:, 2], -ak[:, 0]
            ax[:, 2, 0], ax[:, 2, 1] = -ak[:, 1], ak[:, 0]
            jac = -np.einsum("ji,njk->nik", ft_rot, ax)
            jtj += np.einsum("nij,nik->jk", jac, jac)
            jtr += np.einsum("nij,ni->j", jac, r)
            used += int(ok.sum())
        if used < 10:
            break
        delta = -np.linalg.solve(jtj + 1e-9 * np.eye(3), jtr)
        delta = np.clip(delta, -np.deg2rad(2.0), np.deg2rad(2.0))  # damp large steps
        world_from_aux[:3, :3] = R.from_rotvec(delta).as_matrix() @ rot
        if np.linalg.norm(delta) < 1e-5:
            break
    return world_from_aux


def refine_views_icp(views_ft, max_dist=0.002, max_translation=0.003, max_rotation_deg=3.0):
    """Small per-view rigid ICP correction (robot pose / stereo errors), bounded and anchored.

    The view with the most points is the anchor; the others are registered, largest first, to
    the union of the views accepted so far (point-to-plane). A correction is only applied if it
    is small and improves the fitness.
    """
    import open3d as o3d

    def to_pcd(pts):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
        return pcd

    order = np.argsort([-len(v) for v in views_ft])
    out = [np.asarray(v, dtype=np.float64).copy() for v in views_ft]
    model = [out[order[0]]]
    for i in order[1:]:
        if len(out[i]) < 10:
            continue
        target = to_pcd(np.concatenate(model)).voxel_down_sample(0.0005)
        target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.003, max_nn=30))
        source = to_pcd(out[i])
        before = o3d.pipelines.registration.evaluate_registration(source, target, max_dist, np.eye(4))
        result = o3d.pipelines.registration.registration_icp(
            source, target, max_dist, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
        )
        tf = np.asarray(result.transformation)
        trans = np.linalg.norm(tf[:3, 3])
        ang = np.rad2deg(R.from_matrix(tf[:3, :3]).magnitude())
        if trans <= max_translation and ang <= max_rotation_deg and result.fitness >= before.fitness:
            out[i] = out[i] @ tf[:3, :3].T + tf[:3, 3]
        model.append(out[i])
    return out


def largest_cluster_mask(points: np.ndarray, eps_m: float, min_points: int = 5, voxel_m: float = 0.001) -> np.ndarray:
    """Mask of the points in the largest DBSCAN cluster (computed on a voxel grid for speed)."""
    points = np.asarray(points, dtype=np.float64)
    if eps_m <= 0 or points.shape[0] < max(min_points, 2):
        return np.ones(points.shape[0], dtype=bool)
    import open3d as o3d
    from scipy.spatial import cKDTree

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    voxels = np.asarray(pcd.voxel_down_sample(voxel_m).points) if voxel_m > 0 else points
    vpcd = o3d.geometry.PointCloud()
    vpcd.points = o3d.utility.Vector3dVector(voxels)
    labels = np.asarray(vpcd.cluster_dbscan(eps=eps_m, min_points=min_points))
    if labels.size == 0 or labels.max() < 0:
        return np.ones(points.shape[0], dtype=bool)
    largest = np.bincount(labels[labels >= 0]).argmax()
    # Map every original point to its nearest voxel's label.
    _, nearest = cKDTree(voxels).query(points)
    return labels[nearest] == largest


def extreme_band_centroid(points: np.ndarray, axis: int, side: str = "max", band_m: float = 0.002,
                          percentile: float = 0.5) -> np.ndarray:
    """Centroid of the points within ``band_m`` of the robust extreme along ``axis``."""
    points = np.asarray(points, dtype=np.float64)
    coord = points[:, axis]
    if side == "max":
        extreme = np.percentile(coord, 100.0 - percentile)
        sel = coord >= extreme - band_m
    else:
        extreme = np.percentile(coord, percentile)
        sel = coord <= extreme + band_m
    return points[sel].mean(axis=0)


def align_view_tips(views_ft, band_m=0.002, max_shift_m=0.006):
    """Translate each view so its plug-tip face centre matches the median over views.

    In the fingertip frame +z points from the hand to the plug tip, so the tip face is the band
    of largest z. The plug walls are parallel to that axis and cannot constrain it in ICP; the
    tip face can. Returns the aligned views and the per-view shifts (m).
    """
    tips = np.stack([extreme_band_centroid(v, axis=2, side="max", band_m=band_m) if len(v) else np.full(3, np.nan)
                     for v in views_ft])
    target = np.nanmedian(tips, axis=0)
    out, shifts = [], []
    for view, tip in zip(views_ft, tips):
        shift = target - tip if np.all(np.isfinite(tip)) else np.zeros(3)
        if np.linalg.norm(shift) > max_shift_m:  # implausible: likely a bad view, leave it to ICP
            shift = np.zeros(3)
        out.append(np.asarray(view, dtype=np.float64) + shift)
        shifts.append(shift)
    return out, np.asarray(shifts)


def tip_spread_mm(views_ft, band_m=0.002) -> float:
    """Spread (max - min, mm) of the per-view tip height along the plug axis."""
    z = [extreme_band_centroid(v, axis=2, side="max", band_m=band_m)[2] for v in views_ft if len(v)]
    return float(np.ptp(z) * 1e3) if z else float("nan")


def carve_free_space(view_points, view_colors, view_depths, view_cam_poses, k, margin_m=0.004, window=5):
    """Multi-view visibility check: drop points that another view saw through.

    Fusing views by concatenation lets one view's false surface (e.g. stereo filling a dark,
    textureless hole with a flat "lid") hide what the other views saw, because the top-down render
    keeps the nearest point. For each point of view i, project it into every other view j: if view
    j measured a depth clearly *behind* the point along that ray (its local minimum depth is more
    than ``margin_m`` farther), view j looked through that spot, so the point is not a surface.
    The local minimum (``window`` px) keeps silhouette edges from being carved by mistake.
    """
    from scipy.ndimage import minimum_filter

    k = np.asarray(k, dtype=np.float64)
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    local_min = []
    for depth in view_depths:
        d = np.asarray(depth, dtype=np.float32).copy()
        d[~(d > 0)] = np.inf  # no measurement: no evidence either way
        local_min.append(minimum_filter(d, size=window))
    out_points, out_colors = [], []
    for i, (points, colors) in enumerate(zip(view_points, view_colors)):
        carved = np.zeros(len(points), dtype=bool)
        for j, (d_min, cam_pose) in enumerate(zip(local_min, view_cam_poses)):
            if i == j or not len(points):
                continue
            cam_from_world = np.linalg.inv(np.asarray(cam_pose, dtype=np.float64))
            pc = points @ cam_from_world[:3, :3].T + cam_from_world[:3, 3]
            z = pc[:, 2]
            ok = z > 1e-3
            u = np.round(fx * pc[:, 0] / np.where(ok, z, 1) + cx).astype(int)
            v = np.round(fy * pc[:, 1] / np.where(ok, z, 1) + cy).astype(int)
            h, w = d_min.shape
            ok &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
            measured = np.full(len(points), np.inf)
            measured[ok] = d_min[v[ok], u[ok]]
            carved |= ok & np.isfinite(measured) & (measured > z + margin_m)
        out_points.append(points[~carved])
        out_colors.append(colors[~carved])
        print(f"[script] Socket view {i + 1}: carved {int(carved.sum())} / {len(points)} points seen through by other views")
    return out_points, out_colors


def socket_hole_mask_from_color(points: np.ndarray, colors: np.ndarray, cfg) -> np.ndarray:
    """Points of the socket hole, found by colour.

    The socket is dark and its hole interior is textureless, so stereo fills the hole with depth
    at the top-face height; the hole is still clearly darker in colour. Among the top-face points
    (within ``socket_hole_top_band_m`` of the top), split the grey levels with Otsu and take the
    largest dark cluster. Lower points (side walls, near-black edges) are left out of the split.
    Returns an all-False mask if the split is not convincing.
    """
    points = np.asarray(points, dtype=np.float64)
    gray = np.asarray(colors, dtype=np.float32)[:, :3].mean(axis=1)
    none = np.zeros(len(points), dtype=bool)
    if len(points) < 100:
        return none
    from scipy.ndimage import binary_erosion, binary_fill_holes

    top_z = np.percentile(points[:, 2], 95)
    top = points[:, 2] >= top_z - cfg.socket_hole_top_band_m
    # Interior of the top face only: the outer rim is near-black too (grazing angle / shadow).
    cell = 0.0005
    lo = points[top, :2].min(axis=0)
    ij = np.floor((points[:, :2] - lo) / cell).astype(int)
    shape = tuple(np.maximum(ij[top].max(axis=0) + 1, 1))
    occ = np.zeros(shape, dtype=bool)
    occ[ij[top, 0], ij[top, 1]] = True
    interior = binary_erosion(binary_fill_holes(occ), iterations=max(1, int(round(cfg.socket_hole_rim_m / cell))))
    inside = (ij >= 0).all(axis=1) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    in_interior = np.zeros(len(points), dtype=bool)
    in_interior[inside] = interior[ij[inside, 0], ij[inside, 1]]
    top &= in_interior
    if top.sum() < 50:
        return none
    g = gray[top]
    hist, edges = np.histogram(g, bins=64, range=(float(g.min()), float(g.max()) + 1e-3))
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist); w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    threshold = centers[np.argmax(w0 * w1 * (m0 - m1) ** 2)]
    dark = top & (gray <= threshold)
    frac = dark.sum() / max(top.sum(), 1)
    contrast = g[g > threshold].mean() - g[g <= threshold].mean() if (g <= threshold).any() and (g > threshold).any() else 0.0
    if not (cfg.socket_hole_min_fraction <= frac <= cfg.socket_hole_max_fraction) or contrast < cfg.socket_hole_min_contrast:
        print(f"[script] No convincing socket hole by colour (dark fraction {frac:.2f}, contrast {contrast:.1f}).")
        return none
    hole = dark.copy()
    hole[dark] = largest_cluster_mask(points[dark], cfg.plug_cluster_eps_m, cfg.plug_cluster_min_points,
                                      cfg.plug_cluster_voxel_m)
    return hole


def estimate_aux_camera_rotation(plug_cam_views, fingertip_poses, fallback_rot, min_points=50):
    """Estimate world_from_aux rotation from the plug scan itself (no extrinsic calibration needed).

    All views keep the same fingertip orientation and only translate, so the plug moves rigidly
    by the fingertip displacement: R_world_cam (c_i - c_0) = t_ft_i - t_ft_0, where c_i is the
    plug's (median) position in the camera frame. Solve for R by Kabsch on the displacements.
    The camera translation is not observable this way, and the plug canonical view does not
    depend on it.
    """
    idx = [i for i, pts in enumerate(plug_cam_views) if pts.shape[0] >= min_points]
    if len(idx) < 3:
        print(f"[script] Only {len(idx)} plug views with >= {min_points} points; keeping configured aux rotation.")
        return fallback_rot
    # Bottom-face centre (points nearest the upward camera): the same physical feature in every view,
    # unlike the median, which side views pull up the plug walls and so bias the camera tilt.
    centers_cam = np.stack([extreme_band_centroid(plug_cam_views[i], axis=2, side="min") for i in idx])
    tips_world = np.stack([np.asarray(fingertip_poses[i])[:3, 3] for i in idx]).astype(np.float64)
    d_cam = centers_cam - centers_cam.mean(axis=0)
    d_world = tips_world - tips_world.mean(axis=0)
    u, _, vt = np.linalg.svd(d_cam.T @ d_world)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    residual_mm = np.linalg.norm(d_cam @ rot.T - d_world, axis=1) * 1e3
    print(
        "[script] Estimated aux camera rotation (world_from_aux, xyz euler deg): "
        f"{np.round(R.from_matrix(rot).as_euler('xyz', degrees=True), 1).tolist()} "
        f"(configured: {np.round(R.from_matrix(fallback_rot).as_euler('xyz', degrees=True), 1).tolist()}); "
        f"per-view residual {np.round(residual_mm, 1).tolist()} mm"
    )
    if residual_mm.max() > 10.0:
        print("[script] WARNING: large residual; the plug segmentation may be picking up non-plug points.")
    return rot


def prescan_plug(robot, record: dict) -> None:
    # AGOS capture_plug_bottom_view: carry the held plug over the upward-looking auxiliary
    # camera, keep the held orientation, capture from several fingertip positions, segment
    # the (green) plug, and store the fused cloud in the fingertip frame.
    print("[script] Pre-scanning the plug with the auxiliary camera...")
    time.sleep(1)
    cfg = robot.config
    return_pose = np.asarray(robot._robot_ik_controller.eef_pose, dtype=np.float64).copy()
    photo_rot = return_pose[:3, :3].copy()
    if cfg.plug_photo_vertical:
        # Capture with the plug vertical (no tilt), then return to the tilted start pose. The cloud is
        # stored in the fingertip frame, so it stays valid at any pose; views still only translate.
        axis = np.asarray(record["aligned_rot"])[:, 2] if "aligned_rot" in record else np.array([0.0, 0.0, -1.0])
        photo_rot = untilted_rotation(photo_rot, axis)
        tilt_deg = np.rad2deg(np.arccos(np.clip(np.dot(return_pose[:3, 2], photo_rot[:, 2]), -1.0, 1.0)))
        print(f"[script] Plug scan with the plug vertical (removing {tilt_deg:.1f} deg of tilt, keeping the spin)")
    auxiliary_camera_name, auxiliary_camera = get_auxiliary_zed_camera(robot)
    auxiliary_k, _ = get_zed_intrinsics_and_baseline(auxiliary_camera)
    world_from_aux = np.eye(4, dtype=np.float64)
    world_from_aux[:3, :3] = R.from_euler("xyz", cfg.aux_cam_rot_euler_deg, degrees=True).as_matrix()
    world_from_aux[:3, 3] = np.asarray(cfg.aux_cam_pos_world, dtype=np.float64)

    plug_cam_views, plug_color_views, fingertip_poses = [], [], []
    for view_idx, offset in enumerate(cfg.plug_photo_views):
        view_pos = np.asarray(cfg.plug_photo_pos, dtype=np.float64) + np.asarray(offset, dtype=np.float64)
        move_to_pose_interpolated(robot, view_pos, photo_rot, num_steps=rotation_steps(robot, photo_rot, 5))
        robot._control_refined(view_pos, photo_rot)

        auxiliary_images = read_zed_stereo_rgb(auxiliary_camera)
        auxiliary_depth = compute_foundation_stereo_depth(
            auxiliary_images, auxiliary_camera, scale=cfg.plug_photo_stereo_scale
        )
        clean_depth = np.asarray(auxiliary_depth, dtype=np.float32).copy()
        if cfg.plug_depth_edge_rel_thresh > 0:
            clean_depth[~depth_edge_mask(clean_depth, cfg.plug_depth_edge_rel_thresh, cfg.plug_depth_edge_window)] = 0.0
        points_cam, colors = depth_rgb_to_camera_point_cloud(
            clean_depth,
            auxiliary_images["left"],
            auxiliary_k,
            stride=1,
            max_depth_m=cfg.plug_photo_max_depth_m,
        )
        keep = green_mask(
            colors,
            hue_range_deg=cfg.plug_green_hue_range_deg,
            min_saturation=cfg.plug_green_min_saturation,
            min_value=cfg.plug_green_min_value,
        )
        num_green = int(keep.sum())
        keep[keep] = largest_cluster_mask(
            points_cam[keep], cfg.plug_cluster_eps_m, cfg.plug_cluster_min_points, cfg.plug_cluster_voxel_m
        )
        # Measured fingertip pose at capture time (sim uses the reached pose, not the target).
        fingertip_pose = np.asarray(robot._robot_ik_controller.eef_pose, dtype=np.float64).copy()
        plug_cam_views.append(points_cam[keep])
        plug_color_views.append(colors[keep])
        fingertip_poses.append(fingertip_pose)
        print(f"[script] Plug view {view_idx + 1}/{len(cfg.plug_photo_views)} offset={list(offset)}: "
              f"{num_green} green / {len(keep)} points, {int(keep.sum())} in the largest cluster")
        if view_idx == 0:
            robot._last_auxiliary_stereo_rgb = auxiliary_images
            robot._last_auxiliary_left_rgb = auxiliary_images["left"]
            robot._last_auxiliary_depth = auxiliary_depth

    if cfg.aux_cam_estimate_rotation:
        world_from_aux[:3, :3] = estimate_aux_camera_rotation(
            plug_cam_views, fingertip_poses, fallback_rot=world_from_aux[:3, :3]
        )
    plug_local_views = _views_to_fingertip(plug_cam_views, fingertip_poses, world_from_aux)
    err_initial = multiview_alignment_error(plug_local_views)
    err_rotation = err_initial
    if cfg.plug_refine_rotation_icp:
        refined = refine_aux_rotation_icp(plug_cam_views, fingertip_poses, world_from_aux)
        refined_views = _views_to_fingertip(plug_cam_views, fingertip_poses, refined)
        refined_err = multiview_alignment_error(refined_views)
        if refined_err < err_initial:  # keep only if the views actually overlap better
            world_from_aux, plug_local_views, err_rotation = refined, refined_views, refined_err
    spread_before = tip_spread_mm(plug_local_views)
    if cfg.plug_align_tips:
        plug_local_views, tip_shifts = align_view_tips(
            plug_local_views, band_m=cfg.plug_tip_band_m, max_shift_m=cfg.plug_tip_max_shift_m
        )
        print(f"[script] Plug tip alignment shifts (mm): {np.round(tip_shifts * 1e3, 2).tolist()}")
    if cfg.plug_per_view_icp:
        plug_local_views = refine_views_icp(
            plug_local_views,
            max_dist=cfg.plug_icp_max_dist_m,
            max_translation=cfg.plug_icp_max_translation_m,
            max_rotation_deg=cfg.plug_icp_max_rotation_deg,
        )
    err_final = multiview_alignment_error(plug_local_views)
    print(f"[script] Plug tip height spread across views: {spread_before:.2f} mm -> "
          f"{tip_spread_mm(plug_local_views):.2f} mm")
    print(
        "[script] Plug multi-view overlap (mean distance to other views): "
        f"{err_initial * 1e3:.2f} mm -> {err_rotation * 1e3:.2f} mm (rotation ICP) "
        f"-> {err_final * 1e3:.2f} mm (per-view ICP); refined aux rotation "
        f"{np.round(R.from_matrix(world_from_aux[:3, :3]).as_euler('xyz', degrees=True), 2).tolist()} deg"
    )
    fingertip_pose = fingertip_poses[-1]
    plug_points_fingertip = np.concatenate(plug_local_views, axis=0).astype(np.float32)
    plug_colors = np.concatenate(plug_color_views, axis=0).astype(np.uint8)
    if cfg.plug_outlier_nb_neighbors > 0 and plug_points_fingertip.shape[0] > cfg.plug_outlier_nb_neighbors:
        import open3d as o3d

        plug_pcd = o3d.geometry.PointCloud()
        plug_pcd.points = o3d.utility.Vector3dVector(plug_points_fingertip.astype(np.float64))
        _, inlier_idx = plug_pcd.remove_statistical_outlier(
            nb_neighbors=int(cfg.plug_outlier_nb_neighbors), std_ratio=float(cfg.plug_outlier_std_ratio)
        )
        plug_points_fingertip = plug_points_fingertip[inlier_idx]
        plug_colors = plug_colors[inlier_idx]
    fused_keep = largest_cluster_mask(
        plug_points_fingertip, cfg.plug_cluster_eps_m, cfg.plug_cluster_min_points, cfg.plug_cluster_voxel_m
    )
    plug_points_fingertip = plug_points_fingertip[fused_keep]
    plug_colors = plug_colors[fused_keep]
    if plug_points_fingertip.shape[0] == 0:
        raise RuntimeError(
            f"[script] Plug pre-scan with {auxiliary_camera_name} found no green points; "
            "check plug_photo_* / plug_green_* settings."
        )
    print(f"[script] Fused plug cloud: {plug_points_fingertip.shape[0]} points (fingertip frame)")
    robot._plug_points_fingertip = plug_points_fingertip
    robot._plug_colors = plug_colors
    record["init_plug_points_fingertip"] = plug_points_fingertip
    record["init_plug_colors"] = plug_colors
    record["init_plug_photo_fingertip_pose"] = fingertip_pose
    visualize_open3d_point_cloud(
        plug_points_fingertip,
        plug_colors,
        "Plug point cloud (fingertip frame, multi-view pre-scan)",
        center_on_cloud=True,
    )

    # Back to the pose held before the photo.
    move_to_pose_interpolated(robot, return_pose[:3, 3], return_pose[:3, :3],
                              num_steps=rotation_steps(robot, return_pose[:3, :3], 5))
    robot._control_refined(return_pose[:3, 3], return_pose[:3, :3])

    # Debug: canonical plug view at the start pose (+ overlay with the socket if scanned).
    plug_view = render_plug_canonical(
        plug_points_fingertip, plug_colors, robot._robot_ik_controller.eef_pose
    )
    save_depth_vis(plug_view["depth"], "plug_canonical_depth")
    if "init_socket_points_world" in record:
        socket_pcd = np.asarray(record["init_socket_points_world"])
        views = canonical_views(
            plug_points_fingertip, plug_colors, robot._robot_ik_controller.eef_pose,
            socket_pcd[:, :3], socket_pcd[:, 3:6], shared_center=False,
        )
        save_rgb(views["overlay"], "canonical_overlay")
        save_rgb(paper_overlay(views["plug"], views["socket"]), "canonical_overlay_paper")


def show_socket_virtual_image(socket_view: dict) -> None:
    """Show the socket virtual RGB and depth images (as in the May check figures).

    Depth is shown exactly as the policy receives it (AGOS CanonicalNormalizer): nearest surface
    black, farther lighter, empty pixels white. Saved to outputs/ and opened in the image viewer.
    """
    from PIL import Image, ImageDraw

    h, w = socket_view["mask"].shape
    depth_norm = canonical_normalize(socket_view["depth"])
    depth_img = np.repeat((depth_norm * 255).astype(np.uint8)[..., None], 3, axis=-1)
    rgb = np.asarray(socket_view["rgb"], dtype=np.float32)
    rgb_img = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    gap = np.full((h, 8, 3), 255, dtype=np.uint8)
    panel = np.concatenate([rgb_img, gap, depth_img], axis=1)
    panel = np.repeat(np.repeat(panel, 2, axis=0), 2, axis=1)  # 2x for readability
    canvas = Image.new("RGB", (panel.shape[1], panel.shape[0] + 28), "white")
    canvas.paste(Image.fromarray(panel), (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 8), "Socket virtual RGB image", fill="black")
    draw.text((2 * w + 16 + 6, 8), "Socket virtual depth image (policy input: near = dark, empty = white)", fill="black")
    path = "/home/yinongh/automate/lerobot/outputs/socket_virtual_image_check.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path)
    print(f"[script] Socket virtual images saved to {path}")
    try:
        canvas.show(title="Socket virtual images")
    except Exception as exc:
        print(f"[script] Could not open an image viewer ({exc}); open the file above to check.")


def save_agos_episode_visualization(robot, episode_index) -> None:
    """Write the episode's AGOS visualization (figure, mp4, per-frame PNGs) and clear the buffer."""
    frames = getattr(robot, "_agos_vis_frames", None) or []
    robot._agos_vis_frames = []
    robot._agos_vis_frames_count = 0
    if not frames:
        return
    from PIL import Image

    out_dir = os.path.join(robot.config.agos_vis_dir, f"episode_{int(episode_index or 0):03d}")
    os.makedirs(out_dir, exist_ok=True)
    wrists = [f["wrist"] for f in frames]
    overlays = [f["overlay"] for f in frames]
    for i, (wrist, overlay) in enumerate(zip(wrists, overlays)):
        Image.fromarray(overlay).save(os.path.join(out_dir, f"overlay_{i:04d}.png"))
        if wrist is not None:
            Image.fromarray(np.asarray(wrist, dtype=np.uint8)).save(os.path.join(out_dir, f"wrist_{i:04d}.png"))
    fig_path = agos_figure(wrists, overlays, os.path.join(out_dir, "agos_figure.png"),
                           num_columns=robot.config.agos_vis_columns, title=f"Episode {episode_index}")
    try:
        import imageio.v2 as imageio

        with imageio.get_writer(os.path.join(out_dir, "agos_rollout.mp4"), fps=5) as writer:
            for wrist, overlay in zip(wrists, overlays):
                h = overlay.shape[0]
                if wrist is not None:
                    scale_w = int(round(wrist.shape[1] * h / wrist.shape[0]))
                    wrist_img = np.asarray(Image.fromarray(np.asarray(wrist, dtype=np.uint8)).resize((scale_w, h)))
                    panel = np.concatenate([wrist_img, overlay], axis=1)
                else:
                    panel = overlay
                writer.append_data(panel[: panel.shape[0] // 2 * 2, : panel.shape[1] // 2 * 2])
    except Exception as exc:  # video is a convenience; the figure and PNGs are already written
        print(f"[script] Could not write AGOS rollout video: {exc}")
    print(f"[script] AGOS visualization ({len(frames)} frames) saved to {out_dir} ({os.path.basename(fig_path)})")


def run_scripted_grasp_sequence(robot):
    # visualize_world_and_wrist_camera_frames(robot)
    # exit(0)
    # return
    record = {}
    command_gripper(robot,action = -1.0, label="open")

    skip_grasping = False
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
        print("Placing Plug into the socket, press enter to continue...")
        input()
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

                robot._control_refined(cur_pos, target_rot)

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
        # Initial pose: AGOS distribution (translation + full RPY / axial-spin rotation noise).
        target_pos, target_rot, init_pose_sample = sample_agos_initial_pose(robot, aligned_pos, aligned_rot)
        record["init_pose_sample"] = init_pose_sample
        
    skip_initialization = False
    if not skip_initialization:
        # Socket capture as in May: from just above the aligned pose with the aligned (vertical)
        # orientation, during the straight lift; one wrist capture per view, clouds concatenated.
        print("[script] Capturing the socket point cloud from above the aligned pose...")
        capture_rot = np.asarray(record["aligned_rot"], dtype=np.float64).copy()
        wrist_camera = robot.cameras["cam_wrist"]
        wrist_k, _ = get_zed_intrinsics_and_baseline(wrist_camera)
        cam_to_gripper = np.array(
                [
                    [-0.00768086, -0.94557934, -0.32530096, 0.07294499],
                    [0.99995759, -0.00891583, 0.00230583, -0.03177615],
                    [-0.00508067, -0.32526946, 0.94560772, -0.08727812],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
        socket_top = np.asarray(record["aligned_pos"], dtype=np.float64).copy()
        socket_top[2] -= robot.config.socket_scan_plug_tip_offset
        z_low, z_high = robot.config.socket_scan_z_range
        crop_radius = float(robot.config.socket_scan_crop_radius)
        move_steps = int(robot.config.socket_scan_move_steps)
        view_points, view_colors, view_depths, view_cam_poses = [], [], [], []
        for view_idx, offset in enumerate(robot.config.socket_scan_views):
            # Translate to each view at the capture height, keeping the aligned orientation, so the
            # socket is seen from different angles around the held plug.
            view_pos = socket_top + np.asarray(offset, dtype=np.float64)
            view_rot = capture_rot
            view_steps = rotation_steps(robot, view_rot, move_steps)
            for i in range(view_steps):
                current_rot = robot._robot_ik_controller.eef_pose[:3,:3]
                current_pos = robot._robot_ik_controller.eef_pose[:3,3]
                next_tgt_pos = (view_pos - current_pos) / (view_steps - i) + current_pos
                next_tgt_rot, _, _, _ = robot._interpolate_rotation_matrix(
                    current_rot,
                    view_rot,
                    max_angle_step_deg=float(getattr(robot.config, "script_rot_max_angle_step_deg", 0.3)),
                )
                robot._robot_ik_controller.control(
                        target_pos=next_tgt_pos,
                        target_rot=next_tgt_rot,
                        grasping_action=getattr(robot, "_last_gripper_action", robot.config.gripper_open_action),
                        wait_times=100,
                        joint_threshold=float(getattr(robot.config, "script_joint_solution_threshold", 0.5)),
                    )
            # Settle at the view pose so the capture is not blurred and the pose is accurate.
            robot._control_refined(view_pos, view_rot)

            wrist_images = read_zed_stereo_rgb(wrist_camera)
            # Full-resolution stereo + every pixel: the socket covers a small part of the image.
            wrist_depth = compute_foundation_stereo_depth(
                wrist_images, wrist_camera, scale=robot.config.socket_scan_stereo_scale
            )
            # Same cleanup as the plug scan: drop stereo flying pixels at depth edges.
            wrist_depth = np.asarray(wrist_depth, dtype=np.float32).copy()
            if robot.config.socket_scan_edge_filter and robot.config.plug_depth_edge_rel_thresh > 0:
                wrist_depth[~depth_edge_mask(wrist_depth, robot.config.plug_depth_edge_rel_thresh,
                                             robot.config.plug_depth_edge_window)] = 0.0
            points_cam, colors = depth_rgb_to_camera_point_cloud(
                    wrist_depth,
                    wrist_images["left"],
                    wrist_k,
                    stride=robot.config.socket_scan_stride,
                    max_depth_m=0.5,
                )
            world_from_gripper = np.asarray(robot._robot_ik_controller.eef_pose, dtype=np.float64)
            points_world = transform_points(points_cam, world_from_gripper @ cam_to_gripper)
            keep = (points_world[:, 2] >= z_low) & (points_world[:, 2] <= z_high)
            if crop_radius > 0:
                keep &= np.linalg.norm(points_world[:, :2] - socket_top[:2], axis=1) <= crop_radius
            # Keep the socket body only (drops detached speckles that also skew the bbox centre).
            in_band = int(keep.sum())
            keep[keep] = largest_cluster_mask(
                points_world[keep], robot.config.plug_cluster_eps_m, robot.config.plug_cluster_min_points,
                robot.config.plug_cluster_voxel_m,
            )
            print(f"[script] Socket view {view_idx + 1}: largest cluster kept {int(keep.sum())} / {in_band} "
                  f"points in the z band / crop")
            print(f"[script] Socket view {view_idx + 1}/{len(robot.config.socket_scan_views)} "
                  f"offset={list(offset)}: kept {int(keep.sum())} / {len(keep)} points")
            view_points.append(points_world[keep])
            view_depths.append(wrist_depth)
            view_cam_poses.append(world_from_gripper @ cam_to_gripper)
            view_colors.append(colors[keep])

        if robot.config.socket_free_space_carving and len(view_points) > 1:
            view_points, view_colors = carve_free_space(
                view_points, view_colors, view_depths, view_cam_poses, wrist_k,
                margin_m=robot.config.socket_carving_margin_m,
            )
        wrist_points_world = np.concatenate(view_points, axis=0)
        wrist_colors = np.concatenate(view_colors, axis=0)
        if wrist_points_world.shape[0] == 0:
            raise RuntimeError("[script] Socket pre-scan produced no points; check socket_scan_* settings.")
        if robot.config.socket_hole_from_color:
            hole = socket_hole_mask_from_color(wrist_points_world, wrist_colors, robot.config)
            if hole.any():
                record["socket_hole_center_xy"] = np.median(wrist_points_world[hole, :2], axis=0)
                print(f"[script] Socket hole from colour: removed {int(hole.sum())} dark points, centre "
                      f"{np.round(record['socket_hole_center_xy'] * 1e3, 1).tolist()} mm")
                wrist_points_world = wrist_points_world[~hole]
                wrist_colors = wrist_colors[~hole]
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
            "Socket point cloud (multi-view pre-scan)",
            center_on_cloud=True,
        )
        record["init_socket_pcd"] = robot._initial_wrist_points_world_colored
        # AGOS: render the fused socket cloud once, top-down in world axes, bbox centre.
        socket_view = render_socket_canonical(wrist_points_world, wrist_colors)
        record["init_socket_points_world"] = robot._initial_wrist_points_world_colored
        record["socket_canonical_view"] = socket_view
        record["socket_canonical_depth"] = socket_view["depth"]
        record["socket_canonical_center_xy"] = np.asarray(socket_view["center_xy"], dtype=np.float32)
        save_depth_vis(socket_view["depth"], "socket_canonical_depth")
        if robot.config.socket_check_pause:
            show_socket_virtual_image(socket_view)
            input("[script] Check the socket virtual image, then press Enter to continue initialization...")


        total_init_steps = 150
        num_lift_steps = 25
        lift_start = float(robot._robot_ik_controller.eef_pose[2, 3] - record["aligned_pos"][2])
        for i in range(total_init_steps):
            current_rot = robot._robot_ik_controller.eef_pose[:3,:3]
            current_pos = robot._robot_ik_controller.eef_pose[:3,3]
            # Next tgt pos is the interpolation between current pos and target pos, with a small step size to ensure smooth movement and better IK convergence
            if i < num_lift_steps:
                # Straight extraction along the socket axis, keeping the aligned orientation.
                next_tgt_pos = record["aligned_pos"].copy()
                # continue from the socket-capture height up to the extraction height
                next_tgt_pos[2] += lift_start + (robot.config.init_lift_height - lift_start) * (i + 1) / num_lift_steps
                next_tgt_rot = record["aligned_rot"].copy()
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
                    wait_times=20,
                    joint_threshold=0.0025,
                )
        
        # Compensate gravity-induced pose error at the sampled pose (full 6-DoF, no projection).
        robot._control_refined(target_pos, target_rot)
        init_pos = np.asarray(target_pos, dtype=np.float64).copy()
        init_rot = np.asarray(target_rot, dtype=np.float64).copy()
        # save_rgb(init_socket_rgb, "initial_socket_rgb")
        # save_depth_vis(init_socket_depth, "initial_socket_depth")
        # print("Inspect the initial socket RGB and depth captures, then press Enter to continue...")
        # input()
    skip_plug_photo = False
    if not skip_plug_photo:
        prescan_plug(robot, record)

    # Random Offset from 0.003 ~ 0.005 in x & y direction
    # Random yaw rotation ranging from 5~90 degrees.
    np.random.seed(int(time.time_ns() % (2**32)))
    x_offset = np.random.uniform(0.003, 0.005)
    y_offset = np.random.uniform(0.003, 0.005)
    yaw_rotation = np.random.uniform(5, 90)
    # use the offset to disturb the aligned_pos and aligned_rot, create disturbed_pos and disturbed_rot
    disturbed_pos = record["aligned_pos"].copy()
    disturbed_pos[0] += x_offset
    disturbed_pos[1] += y_offset
    yaw_rad = np.deg2rad(yaw_rotation)  
    yaw_quat = R.from_euler('z', yaw_rad).as_quat()  # in xyzw format
    aligned_quat = R.from_matrix(record["aligned_rot"]).as_quat()
    disturbed_quat = R.from_quat(yaw_quat) * R.from_quat(aligned_quat)
    disturbed_rot = disturbed_quat.as_matrix()
    record["disturbed_pos"] = disturbed_pos
    record["disturbed_rot"] = disturbed_rot

    record["init_EEF_pose"] = robot._robot_ik_controller.eef_pose

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
    events["pause"] = False

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
            elif key == keyboard.Key.space:
                events["pause"] = not events.get("pause", False)
                print("Space pressed:", "PAUSE requested (robot will stop commanding)" if events["pause"]
                      else "RESUME requested")
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
    frame_index = 0
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
        episode_index = None
        if dataset is not None:
            if dataset.episode_buffer is None:
                episode_index = dataset.meta.total_episodes
            else:
                episode_index = dataset.episode_buffer["episode_index"]
        print("Episode Index, Frame Index: ", episode_index, frame_index)
        # Space toggles a manual pause (e.g. to hand-guide the arm and test recovery): no commands and no
        # recorded frames while paused; on resume the robot re-syncs to its measured pose.
        if events is not None and events.get("pause", False):
            if not getattr(robot, "_manual_paused", False):
                robot._manual_paused = True
                if has_method(robot, "on_manual_pause"):
                    robot.on_manual_pause()
            if events.get("exit_early", False):  # arrow keys / Esc still work while paused
                events["exit_early"] = False
                break
            time.sleep(0.05)
            timestamp = time.perf_counter() - start_episode_t
            continue
        if getattr(robot, "_manual_paused", False):
            robot._manual_paused = False
            if has_method(robot, "on_manual_resume"):
                robot.on_manual_resume()
        step_data = robot.teleop_step(record_data=True, insert_meta_data=insert_meta_data, episode_index=episode_index, frame_index=frame_index)
        if step_data is not None:
            observation, action = step_data
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

        frame_index += 1
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
    dataset: LeRobotDataset, robot: Robot, fps: int, use_videos: bool, extra_features: dict | None = None
) -> None:
    expected_features = get_features_from_robot(robot, use_videos)
    if extra_features:
        expected_features = {**expected_features, **extra_features}

    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, expected_features),
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
