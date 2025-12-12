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
from PIL import Image
from termcolor import cprint

from lerobot.common.datasets.image_writer import safe_stop_image_writer
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import get_features_from_robot
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device, has_method
from lerobot.common.utils.aloha_utils import ALOHA_CONFIGURATION, ALOHA_MODEL, VIRTUAL_CAMERA_MAPPING, forward_kinematics, render_and_overlay, setup_renderer
from lerobot.scripts.yufei_policy_utils import compute_pcd, get_gripper_4_points_from_sriram_data, \
    get_4_points_from_gripper_pos_orient, infer_multitask_high_level_model, low_level_policy_infer, \
    get_aloha_future_eef_poses_from_delta_actions, rotation_transfer_6D_to_matrix
from matplotlib import pyplot as plt

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
    policy=None,
    # policy: PreTrainedPolicy = None,
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
    if policy is not None and has_method(policy, "reset"):
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

            ### reset to a predefined rest pose
            if policy is None:
                # state = observation['observation.state']
                action = torch.tensor([90.0000, 192.8320, 193.1836, 176.3965, 176.5723,   6.1523,  19.3359,
                    -3.5156,  -4.2650,  90.9668, 153.8965, 154.5117, 109.5996, 109.9512,
                    -4.6582,  94.9219,  -3.4277,  70]).float()
                robot.send_action(action.squeeze(0))
                

            if policy is not None and type(policy) != dict:
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

            ### NOTE: write my own inference of the high and low-level policy
            if type(policy) == dict:
                obs_queue = policy['obs_queue']
                debug_queue = policy['debug_queue']
                high_level_policy, low_level_policy = policy['high_level'], policy['low_level']
                action_queue = policy['action_queue']

                # import pdb; pdb.set_trace()
                ### NOTE: get the scene pcd from the depth camera
                depth_keys = ["observation.images.cam_azure_kinect_front.transformed_depth", "observation.images.cam_azure_kinect_back.transformed_depth"]
                # import pdb; pdb.set_trace()
                all_cam_depth_images = []
                for depth_key in depth_keys:
                    depth = Image.fromarray(observation[depth_key].numpy()[:, :, 0])
                    depth = np.asarray(depth)
                    all_cam_depth_images.append(depth)
                
                # import pdb; pdb.set_trace()
                # fig, axes = plt.subplots(1, 2)
                # axes = axes.reshape(-1)
                # axes[0].imshow(all_cam_depth_images[0])    
                # axes[1].imshow(all_cam_depth_images[1])
                # plt.show()    
                # plt.close("all")
                
                # import pdb; pdb.set_trace()
                scene_pcd, scene_pcd_in_table_center = compute_pcd(all_cam_depth_images, num_points=4500)
                
                
                # import pdb; pdb.set_trace()
                ### NOTE: get the gripper 4 points
                right_eef_pose = observation['observation.right_eef_pose']
                cprint(f"current eef position: { right_eef_pose[6:9]}", "red")
                # cprint(f"current eef position: { right_eef_pose[6:9]}", "red")
                # cprint(f"current eef position: { right_eef_pose[6:9]}", "red")
                # cprint(f"current eef position: { right_eef_pose[6:9]}", "red")
                eef_pos, eef_rot_6d, eef_gripper_width, eef_pos_robot_base, eef_rot_matrix_robot_base, eef_rot_6d_robot_base, eef_gripper_width_franka = get_gripper_4_points_from_sriram_data(right_eef_pose)
                agent_pos = np.array([*eef_pos_robot_base, *eef_rot_6d_robot_base, *eef_gripper_width_franka])
                cprint(f"eef gripper width (franka): {eef_gripper_width_franka}", "yellow")
                eef_4_points = get_4_points_from_gripper_pos_orient(eef_pos_robot_base, eef_rot_matrix_robot_base, eef_gripper_width_franka)
                x_dir = eef_rot_matrix_robot_base[:, 0]
                y_dir = eef_rot_matrix_robot_base[:, 1]
                z_dir = eef_rot_matrix_robot_base[:, 2]
                eef_x_end = eef_pos_robot_base + 0.1 * eef_rot_matrix_robot_base[:, 0]
                eef_y_end = eef_pos_robot_base + 0.1 * eef_rot_matrix_robot_base[:, 1]
                eef_z_end = eef_pos_robot_base + 0.1 * eef_rot_matrix_robot_base[:, 2]

                obs_queue.append({
                    'scene_pcd': scene_pcd,
                    'eef_4_points': eef_4_points,
                    'agent_pos': agent_pos
                })
                debug_queue.append({
                    "eef_pos": eef_pos,
                    "scene_pcd": scene_pcd,
                })

                if len(action_queue) == 0:
                    high_level_input_np = np.concatenate([scene_pcd, eef_4_points], axis=0)  # (N, 3)
                    high_level_input = torch.from_numpy(high_level_input_np).float().unsqueeze(0).cuda()

                    # import pdb; pdb.set_trace()
                    with torch.no_grad():
                        high_level_predict = infer_multitask_high_level_model(
                            high_level_input, high_level_policy, policy['cat_embedding'], policy['high_level_args'].articubot
                        )  # 1, 4, 3 ## torch tensor

                   
                    ### TODO: figure out how to use the history here
                    # import pdb; pdb.set_trace()
                    if len(obs_queue) == 1:
                    # if True:
                        scene_pcd_history = torch.from_numpy(obs_queue[0]['scene_pcd']).float().to('cuda').unsqueeze(0).unsqueeze(1).repeat(1,2,1,1)
                        agent_pos_history = torch.from_numpy(obs_queue[0]['agent_pos']).float().to('cuda').unsqueeze(0).unsqueeze(1).repeat(1,2,1)
                        eef_4_points_history = torch.from_numpy(obs_queue[0]['eef_4_points']).float().to('cuda').unsqueeze(0).unsqueeze(1).repeat(1,2,1,1)
                    else:
                        scene_pcd_history = torch.from_numpy(np.stack([obs_queue[i]['scene_pcd'] for i in range(-2,0)], axis=0)).float().to('cuda').unsqueeze(0)
                        agent_pos_history = torch.from_numpy(np.stack([obs_queue[i]['agent_pos'] for i in range(-2,0)], axis=0)).float().to('cuda').unsqueeze(0)
                        eef_4_points_history = torch.from_numpy(np.stack([obs_queue[i]['eef_4_points'] for i in range(-2,0)], axis=0)).float().to('cuda').unsqueeze(0)
                        # import pdb; pdb.set_trace()

                    with torch.no_grad():
                        low_level_action = low_level_policy_infer(
                            scene_pcd_history,
                            agent_pos_history, 
                            high_level_predict.float().to('cuda').unsqueeze(1).repeat(1, 2, 1, 1),
                            eef_4_points_history,
                            low_level_policy,
                            cat_idx=policy['cat_idx']
                        )

                        low_level_action_np = low_level_action.cpu().numpy().reshape(-1, 10)
                        # delta_pos_mag = np.linalg.norm(low_level_action_np[:, :3], axis=1)
                        # print("delta pos magnitude is: ", delta_pos_mag)

                    
                    ### convert this low-level action back to the aloha eef pose
                    ### TODO: note this max_relative_target thing
                    # import pdb; pdb.set_trace()
                    aloha_world_eef_pos, aloha_world_eef_orient_6d, aloha_gripper_widths, robot_base_eef_pos = get_aloha_future_eef_poses_from_delta_actions(
                        low_level_action, 
                        eef_pos_robot_base, 
                        eef_rot_matrix_robot_base, 
                        eef_gripper_width_franka
                    )

                    

                    ### 3d plot the scene pcd
                    if False:
                        fig = plt.figure(figsize=(10, 5))

                        ax = fig.add_subplot(1, 2, 1, projection='3d')
                        ax.scatter(scene_pcd[:,0], scene_pcd[:,1], scene_pcd[:,2], s=1, color='grey')
                        # ax.scatter(eef_4_points[:,0], eef_4_points[:,1], eef_4_points[:,2], s=50, color='green')
                        ax.scatter(eef_4_points[[0],0], eef_4_points[[0],1], eef_4_points[[0],2], s=50, color='red')
                        ax.scatter(eef_4_points[[1],0], eef_4_points[[1],1], eef_4_points[[1],2], s=50, color='green')
                        ax.scatter(eef_4_points[[2],0], eef_4_points[[2],1], eef_4_points[[2],2], s=50, color='blue')
                        ax.scatter(eef_4_points[[3],0], eef_4_points[[3],1], eef_4_points[[3],2], s=50, color='black')
                        ### plot the coordinate frame of the current eef
                        # ax.plot([eef_pos_robot_base[0], eef_x_end[0]], [eef_pos_robot_base[1], eef_x_end[1]], [eef_pos_robot_base[2], eef_x_end[2]], color='red', linewidth=2)
                        # ax.plot([eef_pos_robot_base[0], eef_y_end[0]], [eef_pos_robot_base[1], eef_y_end[1]], [eef_pos_robot_base[2], eef_y_end[2]], color='green', linewidth=2)
                        # ax.plot([eef_pos_robot_base[0], eef_z_end[0]], [eef_pos_robot_base[1], eef_z_end[1]], [eef_pos_robot_base[2], eef_z_end[2]], color='blue', linewidth=2)

                        high_level_predict_np = high_level_predict.cpu().numpy()[0]
                        ax.scatter(high_level_predict_np[[0],0], high_level_predict_np[[0],1], high_level_predict_np[[0],2], s=75, color='red')
                        ax.scatter(high_level_predict_np[[1],0], high_level_predict_np[[1],1], high_level_predict_np[[1],2], s=75, color='green')
                        ax.scatter(high_level_predict_np[[2],0], high_level_predict_np[[2],1], high_level_predict_np[[2],2], s=75, color='blue')
                        ax.scatter(high_level_predict_np[[3],0], high_level_predict_np[[3],1], high_level_predict_np[[3],2], s=75, color='black')
                        ax.plot(robot_base_eef_pos[:,0], robot_base_eef_pos[:,1], robot_base_eef_pos[:,2], color='blue', linewidth=4)
                        ax.set_xlabel("X")
                        ax.set_ylabel("Y")
                        ax.set_zlabel("Z")
                        ax.scatter([0], [0], [0], color='red', s=100)
                        ax.set_xlim([0, 1])
                        ax.set_ylim([-0.5, 0.5])
                        ax.set_zlim([-0.3, 0.7])

                        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
                        ax2.scatter(scene_pcd_in_table_center[:,0], scene_pcd_in_table_center[:,1], scene_pcd_in_table_center[:,2], s=1, color='grey')
                        ax2.plot(aloha_world_eef_pos[:,0], aloha_world_eef_pos[:,1], aloha_world_eef_pos[:,2], color='blue', linewidth=4)
                        ### plot the coordinate frame of the current eef

                        debug_queue[-1].update({
                            "commanded_aloha_world_eef_pos": aloha_world_eef_pos,
                            }
                        )

                        for idx in range(4):
                            aloha_eef_pos_current = aloha_world_eef_pos[idx]
                            aloha_world_eef_orient_matrix = rotation_transfer_6D_to_matrix(aloha_world_eef_orient_6d[idx])
                            aloha_x_end = aloha_eef_pos_current + 0.1 * aloha_world_eef_orient_matrix[:, 0]
                            aloha_y_end = aloha_eef_pos_current + 0.1 * aloha_world_eef_orient_matrix[:, 1]
                            aloha_z_end = aloha_eef_pos_current + 0.1 * aloha_world_eef_orient_matrix[:, 2]

                            ax2.plot([aloha_eef_pos_current[0], aloha_x_end[0]], [aloha_eef_pos_current[1], aloha_x_end[1]], [aloha_eef_pos_current[2], aloha_x_end[2]], color='red', linewidth=2)
                            ax2.plot([aloha_eef_pos_current[0], aloha_y_end[0]], [aloha_eef_pos_current[1], aloha_y_end[1]], [aloha_eef_pos_current[2], aloha_y_end[2]], color='green', linewidth=2)
                            ax2.plot([aloha_eef_pos_current[0], aloha_z_end[0]], [aloha_eef_pos_current[1], aloha_z_end[1]], [aloha_eef_pos_current[2], aloha_z_end[2]], color='blue', linewidth=2)

                        ax2.set_xlabel("X")
                        ax2.set_ylabel("Y")
                        ax2.set_zlabel("Z")
                        ax2.scatter([0], [0], [0], color='red', s=100)
                        ax2.set_ylim([-0.5, 0.5])
                        ax2.set_xlim([-0.5, 0.5])
                        ax2.set_zlim([-0.3, 0.7])
                        plt.show()
                        plt.close("all")

                    # if len(debug_queue) >=4:
                    #     plt.close("all")
                    #     commanded_aloha_pos = debug_queue[-5]['commanded_aloha_world_eef_pos']
                    #     commanded_scene_pcd = debug_queue[-5]['scene_pcd']
                    #     commanded_eef_points = debug_queue[-5]['eef_4_points']
                    #     acutal_eef_pos = [debug_queue[i]['eef_pos'] for i in range(-4, 0)]
                    #     fig = plt.figure(figsize=(5, 5))
                    #     ax3 = fig.add_subplot(1, 1, 1, projection='3d')
                    #     ax3.scatter(commanded_scene_pcd[:,0], commanded_scene_pcd[:,1], commanded_scene_pcd[:,2], s=1, color='grey')
                    #     ax3.plot(commanded_aloha_pos[:,0], commanded_aloha_pos[:,1], commanded_aloha_pos[:,2], color='blue', linewidth=3)
                    #     ax3.plot([acutal_eef_pos[i][0] for i in range(4)], [acutal_eef_pos[i][1] for i in range(4)], [acutal_eef_pos[i][2] for i in range(4)], color='red', linewidth=3)
                    #     ax3.scatter(commanded_eef_points[:, 0], commanded_eef_points[:, 1], commanded_eef_points[:, 2], s=40, color='black')
                    #     ax3.set_xlabel("X")
                    #     ax3.set_ylabel("Y")
                    #     ax3.set_zlabel("Z")
                    #     ax3.scatter([0], [0], [0], color='red', s=100)
                    #     ax3.set_ylim([-0.5, 0.5])
                    #     ax3.set_xlim([-0.5, 0.5])
                    #     ax3.set_zlim([-0.3, 0.7])
                    
                    
                    
                    
                    ### TODO: figure out what command is sent to the robot
                    # import pdb; pdb.set_trace()
                    all_aloha_eef_poses = []
                    for  pos, orient_6d, gripper_width in zip(aloha_world_eef_pos, aloha_world_eef_orient_6d, aloha_gripper_widths):
                        eef_pose = [*orient_6d, *pos, gripper_width]
                        eef_pose = torch.tensor(eef_pose).float()
                        all_aloha_eef_poses.append(eef_pose)

                    # import pdb; pdb.set_trace()
                    tmp_all_aloha_eef_poses = [right_eef_pose] + all_aloha_eef_poses
                    diff_positions = []
                    diff_orientations = []
                    for idx in range(1, len(tmp_all_aloha_eef_poses)):
                        diff_pos = tmp_all_aloha_eef_poses[idx][6:9] - tmp_all_aloha_eef_poses[idx-1][6:9]
                        orient_6d_now = tmp_all_aloha_eef_poses[idx][0:6]
                        orient_6d_prev = tmp_all_aloha_eef_poses[idx-1][0:6]

                        rot_matrix_now = rotation_transfer_6D_to_matrix(orient_6d_now.numpy())
                        rot_matrix_prev = rotation_transfer_6D_to_matrix(orient_6d_prev.numpy())

                        from scipy.spatial.transform import Rotation as R
                        quat_now = R.from_matrix(rot_matrix_now).as_quat()
                        quat_prev = R.from_matrix(rot_matrix_prev).as_quat()

                        dot = np.abs(np.dot(quat_now, quat_prev))
                        dot = np.clip(dot, -1.0, 1.0)
                        diff_orient = 2 * np.arccos(dot)

                        diff_orientations.append(diff_orient)
                        diff_positions.append(np.linalg.norm(diff_pos))

                    cprint(f"diff positions between steps: {diff_positions}", "yellow")
                    cprint(f"diff orientations between steps: {np.rad2deg(diff_orientations)}", "yellow")
                    cprint(f"commanded future eef position: { aloha_world_eef_pos}", "green")


                
                    # import pdb; pdb.set_trace()
                    robot_adapter = policy.get('robot_adapter', None)
                    state = observation['observation.state']
                    # import pdb; pdb.set_trace()
                    aloha_joint_actions = []
                    initialization_state = state

                    # cprint("current aloha position: {}".format(right_eef_pose[6:9]), "blue")
                    # all_aloha_eef_poses = []
                    # for i in range(4):
                    #     future_eef_pos = right_eef_pose[6:9] + (i+1) * 0.02 * np.array([0, 0, 1])
                    #     all_aloha_eef_poses.append(torch.tensor([*right_eef_pose[:6].numpy(), *future_eef_pos, right_eef_pose[9]]).float())
                    # cprint("future aloha position: {}".format(all_aloha_eef_poses), "blue")

                    for idx, pose in enumerate(all_aloha_eef_poses):
                        solved_joint = robot_adapter.transform_action(pose, initialization_state)
                        print("diff in joint space is: ", np.abs(solved_joint - initialization_state).max())
                        aloha_joint_actions.append(solved_joint)
                        initialization_state = solved_joint 

                    for aja in aloha_joint_actions:
                        action_queue.append(aja)

                    ### render the robot in simulation for visualization
                    # intrinsics_txts = ["/data/yufei/lerobot/lerobot/scripts/aloha_calibration/intrinsics_000259921812.txt", "/data/yufei/lerobot/lerobot/scripts/aloha_calibration/intrinsics_000003493812.txt"]
                    # extrinsics_txts = ["/data/yufei/lerobot/lerobot/scripts/aloha_calibration/T_world_from_camera_front_1208.txt", "/data/yufei/lerobot/lerobot/scripts/aloha_calibration/T_world_from_camera_back_v1_1020.txt"]
                    # virtual_camera_names = ["teleoperator_pov", "collaborator_pov"]

                    # first_cam = "cam_azure_kinect_front"
                    # rgb_key = f"observation.images.{first_cam}.color"
                    # height, width, _ = observation[rgb_key].numpy().shape

                    # Setup renderer with all cameras at once
                    # renderer = setup_renderer(
                    #     ALOHA_MODEL,
                    #     intrinsics_txts,
                    #     extrinsics_txts,
                    #     0.25,
                    #     width,
                    #     height,
                    #     virtual_camera_names
                    # )

                    # # Gather camera observations and apply phantomize if needed
                    # to_render_joint_angles = [state] + aloha_joint_actions
                    # all_images = []
                    # for idx, aja in enumerate(to_render_joint_angles):
                    #     print("rendering for future step: ", idx)
                    #     renders = []
                    #     for cam_name in ["cam_azure_kinect_front"]:
                    #         rgb_key = f"observation.images.{cam_name}.color"
                    #         depth_key = f"observation.images.{cam_name}.transformed_depth"
                            
                    #         # Overlay RGB with rendered robot
                    #         render = render_and_overlay(
                    #             renderer,
                    #             ALOHA_MODEL,
                    #             aja,
                    #             observation[rgb_key].numpy(),
                    #             0.25,
                    #             VIRTUAL_CAMERA_MAPPING[cam_name],
                    #         )
                    #         renders.append(render)
                            
                    #     image = renders[0]
                    #     all_images.append(image)

                    # image = np.concatenate(all_images, axis=1)
                    # plt.imshow(image)
                    # plt.show()
                        
                action_to_send = action_queue.popleft()


                observation_new = robot.capture_observation()
                # import pdb; pdb.set_trace()
                joint_state = observation_new['observation.state'][9:]
                diff = np.abs(action_to_send[9:] - joint_state).mean()
                control_t = 0
                while diff > 0.02 and control_t < 4:
                    action_joint = robot.send_action(action_to_send)
                    time.sleep(0.2)
                    observation_new = robot.capture_observation()
                    joint_state = observation_new['observation.state'][9:]
                    diff = np.abs(action_to_send[9:] - joint_state).mean()
                    control_t += 1
                    print("waiting for the robot to reach the target joint state, current diff is: ", diff)
                    
                action = {"action": action_joint}
                if robot.use_eef:
                    action["action.right_eef_pose"] = np.zeros(10, dtype=np.float32) ### TODO: record the true eef pose

                
               
        if dataset is not None:
            frame = {**observation, **action, "task": single_task}
            dataset.add_frame(frame)

        # TODO(Steven): This should be more general (for RemoteRobot instead of checking the name, but anyways it will change soon)
        if (display_data and not is_headless()) or (display_data and robot.robot_type.startswith("lekiwi")):
            # if action is not None:
            #     for k, v in action.items():
            #         for i, vv in enumerate(v):
            #             rr.log(f"sent_{k}_{i}", rr.Scalar(vv.numpy()))

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
                            eef_pos_robot_base="high_level",
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
                           rr.Points3D(hl_wrapper.last_gripper_pcd, colors=[255, 0, 0]))

                if hl_wrapper.last_goal_prediction is not None:
                    # Goal prediction
                    rr.log("high_level/goal_prediction",
                        rr.Points3D(hl_wrapper.last_goal_prediction, colors=[0, 255, 0], radii=0.005))

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
        teleoperate=False,
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
