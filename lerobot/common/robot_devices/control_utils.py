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


import json
import logging
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

# Default location of the per-camera calibration JSON used to transform the
# Kinect point cloud into the robot/base frame for rerun visualization.
# Convention from polaris/utils_/data_utils.py: extrinsic is cam-to-base (4x4).
_DEFAULT_CAM_CALIB_PATH = os.environ.get(
    "LEROBOT_CAM_CALIB_JSON",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "polaris", "PolaRiS-Hub", "put_red_cup_no_curtain", "cam_calibration.json",
    ),
)


@cache
def _load_cam_calibration(path: str) -> dict | None:
    """Load camera calibration JSON. Returns None on any failure (visualization
    falls back to camera-frame point cloud)."""
    try:
        with open(path, "r") as f:
            calib = json.load(f)
        return {
            cam_key: np.asarray(cam_data["extrinsic"], dtype=np.float32)
            for cam_key, cam_data in calib.items()
            if "extrinsic" in cam_data
        }
    except Exception as e:
        print(f"[rerun viz] could not load camera calibration {path}: {e}")
        return None


def _eef_pose_to_axes(eef_pose: torch.Tensor, axis_len: float = 0.1):
    """Convert an EEF pose tensor [rot6d(6), pos(3), gripper(1)] to (origins, vectors)
    suitable for rr.Arrows3D. Returns three arrows (X red, Y green, Z blue)."""
    rot = transforms.rotation_6d_to_matrix(eef_pose[:6]).cpu().numpy()
    pos = eef_pose[6:9].cpu().numpy()
    axes = rot @ (np.eye(3) * axis_len)
    return [pos] * 3, axes


def _log_eef_arrows(path: str, eef_pose: torch.Tensor, colors=None, axis_len: float = 0.1):
    if colors is None:
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # X=R, Y=G, Z=B
    origins, vectors = _eef_pose_to_axes(eef_pose, axis_len=axis_len)
    rr.log(path, rr.Arrows3D(origins=origins, vectors=vectors, colors=colors))


_PCD_DEBUG_PRINTED = {"once": False, "missing_key": False, "missing_calib": False}


def _log_kinect_pointcloud(observation: dict, cam_key: str, calib_cam_key: str,
                           subsample_stride: int = 4, max_depth_m: float = 3.0,
                           rerun_path: str = "world/pointcloud_left"):
    """Log a kinect point cloud (in base frame if calibration available, else camera frame)."""
    pcd_obs_key = f"observation.images.{cam_key}.point_cloud"
    if pcd_obs_key not in observation:
        if not _PCD_DEBUG_PRINTED["missing_key"]:
            relevant = [k for k in observation if cam_key in k]
            print(
                f"[rerun viz] '{pcd_obs_key}' not in observation -- pointcloud will not be logged.\n"
                f"            Did you set use_depth=true and use_point_cloud=true in --robot.cameras for '{cam_key}'?\n"
                f"            Keys present for this camera: {relevant}"
            )
            _PCD_DEBUG_PRINTED["missing_key"] = True
        return
    pcd = observation[pcd_obs_key]
    if isinstance(pcd, torch.Tensor):
        pcd = pcd.cpu().numpy()
    # Kinect SDK (pyk4a.depth_point_cloud) returns int16 in MILLIMETERS.
    # Convert to float32 meters before any spatial filtering.
    if pcd.dtype != np.float32:
        pcd = pcd.astype(np.float32) / 1000.0              # (H_d, W_d, 3) meters at DEPTH resolution

    # Pcd lives at the depth camera's native resolution (e.g. 640x576), which
    # does NOT match the color camera's resolution (e.g. 1280x720). Prefer
    # `transformed_color` (color resampled+aligned to depth grid by the SDK).
    # Fall back to resizing raw color to depth dims for an approximate match.
    transformed_color_key = f"observation.images.{cam_key}.transformed_color"
    color_obs_key = f"observation.images.{cam_key}.color"
    rgb = None
    if transformed_color_key in observation:
        rgb = observation[transformed_color_key]
    elif color_obs_key in observation:
        rgb = observation[color_obs_key]
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.cpu().numpy()

    if rgb is not None and rgb.shape[:2] != pcd.shape[:2]:
        # Resolution mismatch (most common: color is at color-cam res, pcd is at depth-cam res).
        # Approximate alignment by resizing color to depth dims using cv2.
        try:
            import cv2
            rgb = cv2.resize(
                rgb, (pcd.shape[1], pcd.shape[0]), interpolation=cv2.INTER_AREA
            )
        except Exception as e:
            print(f"[rerun viz] color resize failed ({e}); pointcloud will be uncolored.")
            rgb = None

    # Subsample for performance (apply AFTER resolution match so they stride together)
    if subsample_stride > 1:
        pcd = pcd[::subsample_stride, ::subsample_stride, :]
        if rgb is not None:
            rgb = rgb[::subsample_stride, ::subsample_stride, :]

    pts = pcd.reshape(-1, 3)
    rgb_flat = rgb.reshape(-1, 3) if rgb is not None else None

    # Filter invalid / too-far points
    valid = np.isfinite(pts).all(axis=1) & (np.linalg.norm(pts, axis=1) > 1e-3)
    valid &= np.linalg.norm(pts, axis=1) < max_depth_m
    pts = pts[valid]
    if rgb_flat is not None:
        rgb_flat = rgb_flat[valid]

    if pts.shape[0] == 0:
        if not _PCD_DEBUG_PRINTED["once"]:
            print(f"[rerun viz] pointcloud has 0 valid points after filter "
                  f"(max_depth_m={max_depth_m}). Check your camera position/scene.")
            _PCD_DEBUG_PRINTED["once"] = True
        return

    # Transform camera frame -> base frame using cam-to-base extrinsic
    calib = _load_cam_calibration(_DEFAULT_CAM_CALIB_PATH)
    if calib is None or calib_cam_key not in calib:
        if not _PCD_DEBUG_PRINTED["missing_calib"]:
            print(
                f"[rerun viz] no calibration for '{calib_cam_key}' at {_DEFAULT_CAM_CALIB_PATH}. "
                f"Pointcloud will be logged in CAMERA frame (will not align with EEF arrows). "
                f"Set LEROBOT_CAM_CALIB_JSON or fix the file to enable cam->base transform."
            )
            _PCD_DEBUG_PRINTED["missing_calib"] = True
    else:
        E = calib[calib_cam_key]                          # (4, 4) cam-to-base
        pts = pts @ E[:3, :3].T + E[:3, 3]

    if not _PCD_DEBUG_PRINTED["once"]:
        print(f"[rerun viz] logging pointcloud '{rerun_path}': {pts.shape[0]} points "
              f"(stride={subsample_stride}, max_depth_m={max_depth_m}).")
        _PCD_DEBUG_PRINTED["once"] = True

    if rgb_flat is None:
        rr.log(rerun_path, rr.Points3D(pts))
    else:
        rr.log(rerun_path, rr.Points3D(pts, colors=rgb_flat.astype(np.uint8)))


def _preview_pred_action_in_rerun(observation: dict, pred_action_eef: torch.Tensor,
                                  robot, display_data: bool):
    """Log the about-to-be-sent action + current state + pcd to rerun BEFORE
    the action is actually sent. Used by step-pause mode so the user can
    inspect what the policy is about to command."""
    if not display_data:
        return
    try:
        if (not is_headless()) or robot.robot_type.startswith("lekiwi"):
            _log_eef_arrows("world/predicted_eef", pred_action_eef)
            if "observation.right_eef_pose" in observation:
                _log_eef_arrows(
                    "world/current_eef",
                    observation["observation.right_eef_pose"],
                    colors=[[150, 80, 80], [80, 150, 80], [80, 80, 150]],
                )
            for key in observation:
                if "image" not in key or key.endswith(".point_cloud"):
                    continue
                v = observation[key]
                if isinstance(v, torch.Tensor):
                    v = v.numpy()
                if v.ndim == 3 and v.shape[-1] == 3:
                    rr.log(key, rr.Image(v), static=True)
                elif v.ndim == 2:
                    rr.log(key, rr.DepthImage(v), static=True)
            _log_kinect_pointcloud(
                observation,
                cam_key="cam_azure_kinect_left",
                calib_cam_key="cam1",
                subsample_stride=4,
                rerun_path="world/pointcloud_left",
            )
    except Exception as e:
        print(f"[step-pause] preview log failed: {e}")


def _wait_for_step(events: dict) -> bool:
    """Block until the user signals to advance. Returns True if the user
    requested continuous mode ('c'), False otherwise. Also returns immediately
    if events['exit_early'] is set."""
    events["step_advance"] = False
    events["step_continue"] = False
    print("[step-pause] waiting... (space=next, c=continuous, ←=exit)", flush=True)
    while not events.get("step_advance") and not events.get("exit_early"):
        time.sleep(0.02)
    return bool(events.get("step_continue"))

from lerobot.common.datasets.image_writer import safe_stop_image_writer
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import get_features_from_robot
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device, has_method
from lerobot.common.utils.aloha_utils import ALOHA_CONFIGURATION, ALOHA_MODEL, VIRTUAL_CAMERA_MAPPING, forward_kinematics, render_and_overlay, setup_renderer

def add_eef_pose(robot, real_joints):
    if robot.robot_type == "aloha":
        eef_pose, eef_pose_se3 = forward_kinematics(ALOHA_CONFIGURATION, real_joints)
        eef_pose = torch.cat([eef_pose, real_joints[-1][None]], axis=0).float()
    elif robot.robot_type == "droid":
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
    if robot.robot_type not in ["stretch", "droid", "dummy", "franka_leap"]:
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
    # Drop visualization-only camera streams (raw depth and point clouds) — the
    # policy never consumes these and the image-prep loop below would choke on
    # their shape. `.transformed_depth` is kept here because some policies
    # (e.g. GhostClient with use_map_anything=False) need the camera-native
    # depth to reach select_action; image-prep handles its (H, W, 1) uint16
    # layout correctly via the "image" branch (→ (1, 1, H, W) float metres).
    for _k in list(observation.keys()):
        if (
            _k.endswith(".point_cloud")
            or _k.endswith(".depth")
        ):
            observation.pop(_k)
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

    events["step_advance"] = False
    events["step_continue"] = False

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
                events["step_advance"] = True
            elif hasattr(key, "char") and key.char == "c":
                print("'c' pressed. Switching to continuous mode.")
                events["step_continue"] = True
                events["step_advance"] = True
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
            robot_kwargs={"observation.state": observation["observation.state"], "observation.right_eef_pose": observation["observation.right_eef_pose"]}
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

    # Step-pause mode: pause before send_action on each fresh policy prediction.
    # Enable via LEROBOT_STEP_PAUSE=1. Press space to advance one prediction,
    # 'c' to switch to continuous, right-arrow to exit episode early.
    step_pause_enabled = bool(int(os.environ.get("LEROBOT_STEP_PAUSE", "0")))
    if step_pause_enabled and (is_headless() or events is None or "step_advance" not in events):
        print("[step-pause] Disabled (headless or keyboard listener not initialized).")
        step_pause_enabled = False
    if step_pause_enabled:
        print("[step-pause] ENABLED. Space=advance one prediction, 'c'=continuous, right-arrow=exit episode.")

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if teleoperate:
            observation, action = robot.teleop_step(record_data=True)
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

                # Pause-before-send on fresh predictions only (AMPLIFY: _actions_done == 1
                # right after a fresh inference; buffered ticks have it > 1).
                fresh_prediction = getattr(policy, "_actions_done", 1) == 1
                if step_pause_enabled and fresh_prediction:
                    _preview_pred_action_in_rerun(
                        observation, pred_action_eef, robot, display_data,
                    )
                    if _wait_for_step(events):
                        # User pressed 'c' -> drop out of pause mode for the rest of this episode
                        step_pause_enabled = False
                    if events.get("exit_early"):
                        break

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

        # Generic per-step observation passthrough: a policy can set
        # `policy._extra_observation = {key: tensor, ...}` inside select_action to
        # surface synthesized obs (e.g. server-produced depth) back here, since
        # predict_action shallow-copies observation and its mutations to the inner
        # dict never reach the dataset-validating frame builder below.
        extra_obs = getattr(policy, "_extra_observation", None) if policy is not None else None
        if extra_obs:
            for _k, _v in extra_obs.items():
                observation[_k] = _v

        if dataset is not None:
            # Drop visualization-only camera streams (raw depth and point clouds).
            # These exist because we enabled use_depth/use_point_cloud on the Kinect for
            # rerun viz, but the dataset feature schema doesn't include them and
            # validate_frame would reject the frame as having "extra features".
            persisted_obs = {
                k: v for k, v in observation.items()
                if not (k.endswith(".point_cloud") or k.endswith(".depth") or k.endswith(".transformed_depth"))
            }
            frame = {**persisted_obs, **action, "task": single_task}
            dataset.add_frame(frame)

        # TODO(Steven): This should be more general (for RemoteRobot instead of checking the name, but anyways it will change soon)
        if (display_data and not is_headless()) or (display_data and robot.robot_type.startswith("lekiwi")):
            if action is not None:
                for k, v in action.items():
                    for i, vv in enumerate(v):
                        rr.log(f"sent_{k}_{i}", rr.Scalar(vv.numpy()))

                if "action.right_eef_pose" in action:
                    # Predicted (commanded) EEF pose this tick (X=R, Y=G, Z=B)
                    _log_eef_arrows("world/predicted_eef", action["action.right_eef_pose"])

            # Current EEF state of the robot (dimmer colors so it's distinguishable from predicted)
            if "observation.right_eef_pose" in observation:
                _log_eef_arrows(
                    "world/current_eef",
                    observation["observation.right_eef_pose"],
                    colors=[[150, 80, 80], [80, 150, 80], [80, 80, 150]],
                )

            image_keys = [key for key in observation if "image" in key and not key.endswith(".point_cloud")]
            for key in image_keys:
                v = observation[key]
                if isinstance(v, torch.Tensor):
                    v = v.numpy()
                if v.ndim == 3 and v.shape[-1] == 3:
                    rr.log(key, rr.Image(v), static=True)
                elif v.ndim == 2:
                    # depth or other single-channel
                    rr.log(key, rr.DepthImage(v), static=True)

            # Kinect point cloud transformed into the base frame, if available.
            # cam_azure_kinect_left = device_id 1 -> "cam1" entry in cam_calibration.json
            _log_kinect_pointcloud(
                observation,
                cam_key="cam_azure_kinect_left",
                calib_cam_key="cam1",
                subsample_stride=4,
                rerun_path="world/pointcloud_left",
            )

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
    # Only diff the features the robot declares — the dataset may have additional
    # features (policy viz channels registered mid-recording, goal-conditioning
    # projections, etc.) that aren't the robot's responsibility, and demanding
    # exact equality would block any resume after such features were added.
    robot_features = get_features_from_robot(robot, use_videos)
    dataset_features_subset = {k: dataset.features[k] for k in robot_features if k in dataset.features}

    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset_features_subset, robot_features),
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
