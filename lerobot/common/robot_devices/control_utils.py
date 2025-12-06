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

from lerobot.common.datasets.image_writer import safe_stop_image_writer
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import get_features_from_robot
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device, has_method
from lerobot.common.utils.aloha_utils import ALOHA_CONFIGURATION, ALOHA_MODEL, VIRTUAL_CAMERA_MAPPING, forward_kinematics, render_and_overlay, setup_renderer

def add_eef_pose(real_joints):
    eef_pose, eef_pose_se3 = forward_kinematics(ALOHA_CONFIGURATION, real_joints)
    eef_pose = torch.cat([eef_pose, real_joints[-1][None]], axis=0).float()
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
    if not robot.robot_type.startswith("stretch"):
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
            observation, action = robot.teleop_step(record_data=True)
            if robot.use_eef:
                observation["observation.right_eef_pose"] = add_eef_pose(observation['observation.state'])
                action["action.right_eef_pose"] = add_eef_pose(action['action'])
        else:
            observation = robot.capture_observation()
            if robot.use_eef:
                observation["observation.right_eef_pose"] = add_eef_pose(observation['observation.state'])
            action = None

            if policy is not None:
                # Pretty ugly, but moving this code inside the policy makes it uglier to visualize
                # the goal_gripper_proj key.

                if hasattr(policy.config, "enable_goal_conditioning") and policy.config.enable_goal_conditioning:
                    # Generate new goal prediction when queue is empty
                    # This code is specific to diffusion policy / kinect :(
                    if hasattr(policy, "_queues") and len(policy._queues[policy.act_key]) == 0:
                        # Gather observations from all configured cameras
                        camera_obs = {}
                        state = observation["observation.state"].numpy()

                        # Setup renderer once for all cameras if using phantomize
                        if hasattr(policy.config, "phantomize") and policy.config.phantomize:
                            if policy.renderer is None:
                                intrinsics_txts, extrinsics_txts, virtual_camera_names = [], [], []
                                for cam_name in policy.high_level.camera_names:
                                    intrinsics_txts.append(f"lerobot/scripts/{policy.high_level.calibration_data[cam_name]['intrinsics']}")
                                    extrinsics_txts.append(f"lerobot/scripts/{policy.high_level.calibration_data[cam_name]['extrinsics']}")
                                    virtual_camera_names.append(VIRTUAL_CAMERA_MAPPING[cam_name])

                                # Get image dimensions from first camera
                                first_cam = policy.high_level.camera_names[0]
                                rgb_key = f"observation.images.{first_cam}.color"
                                height, width, _ = observation[rgb_key].numpy().shape

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

                            if hasattr(policy.config, "phantomize") and policy.config.phantomize:
                                # Overlay RGB with rendered robot
                                render = render_and_overlay(
                                    policy.renderer,
                                    ALOHA_MODEL,
                                    state,
                                    camera_obs[cam_name]["rgb"].copy(),
                                    policy.downsample_factor,
                                    VIRTUAL_CAMERA_MAPPING[cam_name],
                                )
                                camera_obs[cam_name]["rgb"] = render
                        # import matplotlib.pyplot as plt, cv2, os
                        # os.makedirs("phantomize_inference_viz", exist_ok=True)
                        # img = cv2.hconcat([camera_obs[cam_name]["rgb"] for cam_name in policy.high_level.camera_names])
                        # plt.imsave(f"phantomize_inference_viz/{time.time()}.png", img)


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

        if dataset is not None:
            frame = {**observation, **action, "task": single_task}
            dataset.add_frame(frame)

        # TODO(Steven): This should be more general (for RemoteRobot instead of checking the name, but anyways it will change soon)
        if (display_data and not is_headless()) or (display_data and robot.robot_type.startswith("lekiwi")):
            if action is not None:
                for k, v in action.items():
                    for i, vv in enumerate(v):
                        rr.log(f"sent_{k}_{i}", rr.Scalar(vv.numpy()))

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


def reset_environment(robot, events, reset_time_s, fps):
    # TODO(rcadene): refactor warmup_record and reset_environment
    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()

    control_loop(
        robot=robot,
        control_time_s=reset_time_s,
        events=events,
        fps=fps,
        teleoperate=True,
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
