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
import rerun as rr
import torch
from deepdiff import DeepDiff
from termcolor import colored
import pytorch3d.transforms as transforms

from lerobot.common.datasets.image_writer import safe_stop_image_writer
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import get_features_from_robot
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device, has_method
from lerobot.common.utils.aloha_utils import ALOHA_CONFIGURATION, ALOHA_MODEL, VIRTUAL_CAMERA_MAPPING, forward_kinematics, render_and_overlay, setup_renderer
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


def save_auxiliary_stereo_images(
    images: dict[str, np.ndarray],
    camera_name: str,
    output_dir: str = "outputs/scripted_grasp_auxiliary",
) -> dict[str, str]:
    """Save the auxiliary stereo images locally for quick visual inspection."""
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    saved_paths = {}

    for image_name, image in images.items():
        path = os.path.join(output_dir, f"{timestamp}_{camera_name}_{image_name}.png")
        Image.fromarray(image).save(path)

        latest_path = os.path.join(output_dir, f"latest_{image_name}.png")
        Image.fromarray(image).save(latest_path)
        saved_paths[image_name] = path

    return saved_paths


def save_foundation_stereo_depth(
    images: dict[str, np.ndarray],
    camera,
    camera_name: str,
    output_dir: str = "outputs/scripted_grasp_auxiliary",
    max_depth: float = 0.35,
) -> dict[str, str | np.ndarray]:
    """Run FoundationStereo on the auxiliary ZED pair and save depth products."""
    import cv2

    os.makedirs(output_dir, exist_ok=True)
    fs_input_dir = os.path.join(output_dir, "foundationstereo_input")
    fs_output_dir = os.path.join(output_dir, "foundationstereo_output")
    os.makedirs(fs_input_dir, exist_ok=True)
    os.makedirs(fs_output_dir, exist_ok=True)

    k, baseline_m = get_zed_intrinsics_and_baseline(camera)
    left_rgb = images["left"]
    right_rgb = images["right"]

    cv2.imwrite(os.path.join(fs_input_dir, "left.png"), cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(fs_input_dir, "right.png"), cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR))
    with open(os.path.join(fs_input_dir, "K.txt"), "w") as f:
        f.write(
            f"{k[0, 0]} {k[0, 1]} {k[0, 2]} "
            f"{k[1, 0]} {k[1, 1]} {k[1, 2]} "
            f"{k[2, 0]} {k[2, 1]} {k[2, 2]}\n"
        )
        f.write(f"{baseline_m}\n")

    fs_depth = get_foundation_stereo_depth()
    depth = fs_depth.infer_depth(
        left_rgb,
        right_rgb,
        fx=float(k[0, 0]),
        baseline_m=baseline_m,
        remove_invisible=True,
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    depth_path = os.path.join(fs_output_dir, f"{timestamp}_{camera_name}_depth_meter.npy")
    depth_vis_path = os.path.join(fs_output_dir, f"{timestamp}_{camera_name}_depth_contrast_rgb.png")
    rgb_path = os.path.join(fs_output_dir, f"{timestamp}_{camera_name}_rgb.png")

    depth_vis_rgb = make_contrast_depth_vis(depth, max_depth=max_depth)
    if depth_vis_rgb.shape[:2] != left_rgb.shape[:2]:
        depth_vis_rgb = cv2.resize(
            depth_vis_rgb,
            (left_rgb.shape[1], left_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    np.save(depth_path, depth)
    cv2.imwrite(depth_vis_path, cv2.cvtColor(depth_vis_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(rgb_path, cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR))

    latest_depth_path = os.path.join(fs_output_dir, "latest_depth_meter.npy")
    latest_depth_vis_path = os.path.join(fs_output_dir, "latest_depth_contrast_rgb.png")
    latest_rgb_path = os.path.join(fs_output_dir, "latest_rgb.png")
    np.save(latest_depth_path, depth)
    cv2.imwrite(latest_depth_vis_path, cv2.cvtColor(depth_vis_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(latest_rgb_path, cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR))

    return {
        "depth": depth_path,
        "depth_vis": depth_vis_path,
        "rgb": rgb_path,
        "depth_array": depth,
    }


def command_gripper(robot, action, label, ticks=5, sleep_s=0.2):
    """Franka gripper convention: negative opens, nonnegative closes."""
    print(f"Commanding gripper {label}...")
    for tick in range(ticks):
        robot.robot_interface.gripper_control(action)
        print(f"  {label} command {tick + 1}/{ticks}")
        time.sleep(sleep_s)




def run_scripted_grasp_sequence(robot):
    # auxiliary_camera_name, auxiliary_camera = get_auxiliary_zed_camera(robot)
    # auxiliary_images = read_zed_stereo_rgb(auxiliary_camera)
    # robot._last_auxiliary_stereo_rgb = auxiliary_images
    # saved_paths = save_auxiliary_stereo_images(auxiliary_images, auxiliary_camera_name)
    # depth_paths = save_foundation_stereo_depth(auxiliary_images, auxiliary_camera, auxiliary_camera_name)
    # robot._last_auxiliary_depth = depth_paths["depth_array"]
    # print(
    #     f"[script] Read {auxiliary_camera_name} stereo RGB images: "
    #     f"left={auxiliary_images['left'].shape} right={auxiliary_images['right'].shape}"
    # )
    # print(
    #     "[script] Saved auxiliary stereo images: "
    #     f"left={saved_paths['left']} right={saved_paths['right']}"
    # )
    # print(
    #     "[script] Saved FoundationStereo depth: "
    #     f"depth={depth_paths['depth']} left_vis={depth_paths['depth_vis']} rgb={depth_paths['rgb']}"
    # )
    # return
    target_quat = np.array(robot.config.target_quat, dtype=np.float64)
    target_pos = np.array(robot.config.approach_pos, dtype=np.float64)
    target_rot = transforms.quaternion_to_matrix(
            torch.tensor([target_quat[3], target_quat[0], target_quat[1], target_quat[2]], dtype=torch.float64)
        ).numpy()
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
    input()
    aligned_pose = robot._robot_ik_controller.eef_pose
    aligned_pos = aligned_pose[:3,3]
    aligned_rot = aligned_pose[:3,:3]
    print("aligned_pose:\n", aligned_pose)
    print("aligned_pos:\n", aligned_pos)
    print("aligned_rot:\n", aligned_rot)
    record = {
        "aligned_pose": aligned_pose,
        "aligned_pos": aligned_pos,
        "aligned_rot": aligned_rot,
    }
    # Lift Up the Gripper
    target_pos = aligned_pos.copy()
    target_pos[2] += 0.02
    target_pos[1] += np.random.uniform(-0.01, 0.01)
    target_pos[0] += np.random.uniform(-0.01, 0.01)
    for i in range(50):
        current_pos = robot._robot_ik_controller.eef_pose[:3,3]
        # Next tgt pos is the interpolation between current pos and target pos, with a small step size to ensure smooth movement and better IK convergence
        next_tgt_pos = (target_pos - current_pos) / (50-i) + current_pos
        robot._robot_ik_controller.control(
                target_pos=next_tgt_pos,
                target_rot=aligned_rot,
                grasping_action=getattr(robot, "_last_gripper_action", robot.config.gripper_open_action),
                wait_times=100,
                joint_threshold=float(getattr(robot.config, "script_joint_solution_threshold", 0.5)),
            )
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

        if teleoperate:
            if not getattr(robot, "_scripted_grasp_sequence_done", False):
                robot._scripted_grasp_sequence_done = True
                run_scripted_grasp_sequence(robot)

            # observation, action = robot.teleop_step(record_data=True, insert_meta_data=insert_meta_data)
            if robot.use_eef:
                observation["observation.right_eef_pose"] = add_eef_pose(robot, observation['observation.state'])
                action["action.right_eef_pose"] = add_eef_pose(robot, action['action'])
        else:
            observation = robot.capture_observation()
            if robot.use_eef:
                observation["observation.right_eef_pose"] = add_eef_pose(robot, observation['observation.state'])
            action = None

            if policy is not None:
                # Pretty ugly, but moving this code inside the policy makes it uglier to visualize
                # the goal_gripper_proj key.
                camera_names = get_camera_names_from_observation(observation)
                observation = get_phantomized_observation(policy, policy.config, camera_names, observation)
                observation = compute_goal_prediction(policy, policy.config, single_task, observation)

                observation["task"] = single_task
                pred_action, pred_action_eef = predict_action(
                    observation, policy, get_safe_torch_device(policy.config.device), policy.config.use_amp
                )
                # Action can eventually be clipped using `max_relative_target`,
                # so action actually sent is saved in the dataset.
                action = robot.send_action(pred_action)
                action = {"action": action}
                if robot.use_eef:
                    action["action.right_eef_pose"] = pred_action_eef

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

        # TODO(Steven): This should be more general (for RemoteRobot instead of checking the name, but anyways it will change soon)
        if (display_data and not is_headless()) or (display_data and robot.robot_type.startswith("lekiwi")):
            if action is not None:
                for k, v in action.items():
                    for i, vv in enumerate(v):
                        rr.log(f"sent_{k}_{i}", rr.Scalar(vv.numpy()))

                if "action.right_eef_pose" in action:
                    eef_pose = action['action.right_eef_pose']
                    eef_rot, eef_trans = transforms.rotation_6d_to_matrix(eef_pose[:6]), eef_pose[6:9]
                    # Log EEF pose as a 3D coordinate frame
                    origin = eef_trans.numpy()
                    axes = eef_rot.numpy() @ (np.eye(3) * 0.1)
                    rr.log("high_level/eef_frame", rr.Arrows3D(
                        origins=[origin] * 3,
                        vectors=axes,
                        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                    ))

            image_keys = [key for key in observation if "image" in key]
            for key in image_keys:
                rr.log(key, rr.Image(observation[key].numpy()), static=True)

            # Add point cloud visualization from high-level model
            if policy is not None and hasattr(policy, 'high_level'):
                hl_wrapper = policy.high_level

                white_bg = False
                if white_bg:
                    # Set white background for 3D view using blueprint
                    blueprint = rr.blueprint.Blueprint(
                        rr.blueprint.Spatial3DView(
                            origin="high_level",
                            background=[255, 255, 255]  # White background
                        )
                    )
                    rr.send_blueprint(blueprint)

                if hl_wrapper.last_pcd_xyz is not None:
                    pcd_rgb = ((hl_wrapper.last_pcd_rgb + 1) * 255 / 2).astype(np.uint8)
                    # Scene point cloud with colors
                    rr.log("high_level/scene_pointcloud", rr.Points3D(hl_wrapper.last_pcd_xyz, colors=pcd_rgb))

                # Gripper point cloud
                if hl_wrapper.last_gripper_pcd is not None:
                    rr.log("high_level/gripper_pointcloud",
                           rr.Points3D(hl_wrapper.last_gripper_pcd, colors=[0, 255, 0]))

                if hl_wrapper.last_goal_prediction is not None:
                    # Goal prediction
                    rr.log("high_level/goal_prediction",
                        rr.Points3D(hl_wrapper.last_goal_prediction, colors=[255, 0, 0], radii=0.01))

                # Goal gripper mesh
                if hl_wrapper.last_goal_gripper_mesh is not None:
                    mesh = hl_wrapper.last_goal_gripper_mesh
                    LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)
                    rr.log("high_level/goal_gripper_mesh", rr.Mesh3D(
                        vertex_positions=mesh.vertices,
                        triangle_indices=mesh.faces,
                        vertex_normals=mesh.vertex_normals,
                        vertex_colors=np.tile(LIGHT_PURPLE, (len(mesh.vertices), 1))
                    ))

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
