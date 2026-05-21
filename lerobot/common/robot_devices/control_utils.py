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


def record_droid_insertion_pose(robot, path: str = "outputs/scripted_insertion_pose.json") -> dict:
    """Record the current insertion pose in robot and IK frames.

    Use the ``ik_grip_site`` pose for later delta-action interpolation and IK.
    The ``robot_eef`` pose is kept for readability/debugging against deoxys.
    """
    final_rot, final_pos = robot.robot_interface.last_eef_rot_and_pos
    final_pos = final_pos.squeeze()
    final_rot_6d = transforms.matrix_to_rotation_6d(
        torch.from_numpy(final_rot[None])
    ).squeeze()

    joints = robot._get_franka_joints()
    ik_rot, ik_pos = droid_ik_model_eef_pose(joints)
    ik_rot_6d = transforms.matrix_to_rotation_6d(
        torch.from_numpy(ik_rot[None])
    ).squeeze()

    try:
        gripper_width = float(robot._get_gripper_width())
    except Exception:
        gripper_width = None

    record = {
        "recommended_policy_frame": "ik_grip_site",
        "timestamp_unix_s": time.time(),
        "ik_grip_site": {
            "position": [float(v) for v in ik_pos.tolist()],
            "rotation_matrix": [[float(v) for v in row] for row in ik_rot.tolist()],
            "rotation_6d": [float(v) for v in ik_rot_6d.tolist()],
        },
        "robot_eef": {
            "position": [float(v) for v in final_pos.tolist()],
            "rotation_matrix": [[float(v) for v in row] for row in final_rot.tolist()],
            "rotation_6d": [float(v) for v in final_rot_6d.tolist()],
        },
        "state": {
            "joints": [float(v) for v in joints.tolist()],
            "gripper_width": gripper_width,
        },
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(record, f, indent=2)

    print(f"Recorded insertion pose to {path}")
    print(
        "Insertion pose for scripted policy (ik_grip_site pos + rot_6d): "
        f"{np.round(ik_pos, 6).tolist()} + "
        f"{[round(v, 6) for v in ik_rot_6d.tolist()]}"
    )
    print(
        "Debug robot_eef pose (pos + rot_6d): "
        f"{np.round(final_pos, 6).tolist()} + "
        f"{[round(v, 6) for v in final_rot_6d.tolist()]}"
    )
    print(
        "Frame offset ik_grip_site - robot_eef: "
        f"{np.round(ik_pos - final_pos, 6).tolist()}"
    )
    return record


def lift_from_recorded_insertion_pose(
    robot,
    record: dict,
    lift_m: float = 0.03,
    num_steps: int = 60,
    sleep_s: float = 0.02,
):
    """Lift straight up using the same robot-frame smooth pose primitive as keyboard tuning."""
    from deoxys.utils import transform_utils

    recorded_start_pos = np.asarray(record["ik_grip_site"]["position"], dtype=np.float64)
    start_ik_rot, start_ik_pos = droid_ik_model_eef_pose(robot._get_franka_joints())
    robot_rot, robot_pos = robot.robot_interface.last_eef_rot_and_pos
    robot_pos = robot_pos.squeeze()
    target_robot_pos = robot_pos.copy()
    target_robot_pos[2] += lift_m
    target_quat = transform_utils.mat2quat(robot_rot)

    print(
        "Lifting with smooth_pose_move_to: "
        f"robot_start={np.round(robot_pos, 6).tolist()} "
        f"robot_target={np.round(target_robot_pos, 6).tolist()} "
        f"ik_start={np.round(start_ik_pos, 6).tolist()} "
        f"recorded_start={np.round(recorded_start_pos, 6).tolist()} "
        f"lift_z={lift_m:.6f}"
    )

    smooth_pose_move_to(
        robot,
        target_pos=target_robot_pos,
        target_quat=target_quat,
        step_m=0.001,
        num_steps_per_waypoint=10,
        num_additional_steps=0,
    )
    _, final_ik_pos = droid_ik_model_eef_pose(robot._get_franka_joints())
    _, final_robot_pos = robot.robot_interface.last_eef_rot_and_pos
    final_robot_pos = final_robot_pos.squeeze()
    print(
        "After smooth lift: "
        f"robot_pos={np.round(final_robot_pos, 6).tolist()} "
        f"robot_error={np.round(target_robot_pos - final_robot_pos, 6).tolist()} "
        f"ik_pos={np.round(final_ik_pos, 6).tolist()} "
        f"ik_delta={np.round(final_ik_pos - start_ik_pos, 6).tolist()}"
    )

def run_scripted_grasp_sequence(robot):
    target_quat = np.array(robot.config.target_quat, dtype=np.float64)
    approach_pos = np.array(robot.config.approach_pos, dtype=np.float64)

    print("Moving to the approach pose")
    # smooth_pose_move_to(
    #     robot,
    #     target_pos=approach_pos,
    #     target_quat=target_quat,
    #     step_m=0.01,
    #     num_steps_per_waypoint=20,
    #     num_additional_steps=0,
    # )

    if not hasattr(robot, "_robot_ik_controller") or robot._robot_ik_controller is None:
        raise RuntimeError("RobotIKController is required for the approach lift test.")
    for _ in range(10):
        before_joints = robot._get_franka_joints()
        before_pos, before_quat = robot._robot_ik_controller.bullet_ik_wrapper.forward_kinematics(before_joints)
        before_franka_pose = robot._robot_ik_controller.eef_pose
        print("EEF pose: ", robot._robot_ik_controller.eef_pose)
        print("Before Pose: ", before_pos)
        # before_pos = np.asarray(before_pos, dtype=np.float64)
        before_pos = np.asarray(before_franka_pose[:3, 3], dtype=np.float64)
        target_pos = before_pos.copy()
        target_pos[1] +=-0.001
        target_rot = transforms.quaternion_to_matrix(
            torch.tensor([before_quat[3], before_quat[0], before_quat[1], before_quat[2]], dtype=torch.float64)
        ).numpy()

        print(
            "[approach_lift_test] command "
            f"before_pybullet_pos={np.round(before_pos, 6).tolist()} "
            f"target_pybullet_pos={np.round(target_pos, 6).tolist()} "
            "delta=[0.0, 0.0, 0.001]"
        )

        success = robot._robot_ik_controller.control(
            target_pos=target_pos,
            target_rot=target_rot,
            grasping_action=getattr(robot, "_last_gripper_action", robot.config.gripper_open_action),
            # wait_times=int(getattr(robot.config, "script_joint_wait_times", 50)),
            wait_times=100,
            joint_threshold=float(getattr(robot.config, "script_joint_solution_threshold", 0.5)),
        )

        after_joints = robot._get_franka_joints()
        after_pos, _ = robot._robot_ik_controller.bullet_ik_wrapper.forward_kinematics(after_joints)
        after_pos = np.asarray(after_pos, dtype=np.float64)
        after_pos_real_franka = robot._robot_ik_controller.eef_pose[:3, 3]
        target_joints = getattr(robot._robot_ik_controller, "last_joint_target", None)
        if target_joints is None:
            joint_error = None
        else:
            joint_error = np.asarray(target_joints, dtype=np.float64) - after_joints

        print(
            "[approach_lift_test] result\n"
            f"  success={success}\n"
            f"  after_pybullet_pos={np.round(after_pos, 6).tolist()}\n"
            f"  actual_delta={np.round(after_pos - before_pos, 6).tolist()}\n"
            f"  target_error={np.round(target_pos - after_pos_real_franka, 6).tolist()}\n"
            f"  target_joints={None if target_joints is None else np.round(target_joints, 6).tolist()}\n"
            f"  after_joints={np.round(after_joints, 6).tolist()}\n"
            f"  joint_error={None if joint_error is None else np.round(joint_error, 6).tolist()}\n"
            f"  joint_error_norm={None if joint_error is None else float(np.linalg.norm(joint_error)):.6f}"
        )
        time.sleep(0.5)
    return {"debug_lift_only": True}

    # _, actual_approach_pos = robot.robot_interface.last_eef_rot_and_pos
    # plane_z = actual_approach_pos.squeeze()[2]
    # approach_pos[2] = plane_z
    # print(
    #     "Tune grasp XY at the current approach height. "
    #     f"Keyboard plane robot_frame_z={plane_z:.4f}. Press Enter to finish tuning."
    # )
    # while True:
    #     approach_pos[2] = plane_z
    #     key = input("w/a/s/d to adjust XY, Enter to finish: ").strip().lower()

    #     if key in ("", "enter"):
    #         break

    #     if key == "a":
    #         approach_pos[2] -= 0.001   # y -
    #     elif key == "d":
    #         approach_pos[2] += 0.001   # y +
    #     elif key == "w":
    #         approach_pos[0] -= 0.001   # x -
    #     elif key == "s":
    #         approach_pos[0] += 0.001   # x +
    #     else:
    #         continue

    #     approach_pos[2] = plane_z
    #     grasp_pos[:2] = approach_pos[:2]
    #     print(
    #         "Adjusted grasp XY: "
    #         f"approach={np.round(approach_pos, 6).tolist()} "
    #         f"grasp={np.round(grasp_pos, 6).tolist()}"
    #     )
    #     _, before_robot_pos = robot.robot_interface.last_eef_rot_and_pos
    #     before_robot_pos = before_robot_pos.squeeze()
    #     _, before_ik_pos = droid_ik_model_eef_pose(robot._get_franka_joints())
    #     before_ik_offset = before_ik_pos - before_robot_pos
    #     target_ik_pos = approach_pos + before_ik_offset
    #     smooth_pose_move_to(
    #         robot,
    #         target_pos=approach_pos,
    #         target_quat=target_quat,
    #         step_m=0.001,
    #         num_steps_per_waypoint=5,
    #         num_additional_steps=0,
    #     )
    #     _, after_robot_pos = robot.robot_interface.last_eef_rot_and_pos
    #     after_robot_pos = after_robot_pos.squeeze()
    #     _, after_ik_pos = droid_ik_model_eef_pose(robot._get_franka_joints())
    #     print(
    #         "[keyboard:drift] "
    #         f"target_robot={np.round(approach_pos, 6).tolist()} "
    #         f"before_robot={np.round(before_robot_pos, 6).tolist()} "
    #         f"after_robot={np.round(after_robot_pos, 6).tolist()} "
    #         f"robot_actual_delta={np.round(after_robot_pos - before_robot_pos, 6).tolist()} "
    #         f"robot_target_error={np.round(approach_pos - after_robot_pos, 6).tolist()}"
    #     )
    #     print(
    #         "[keyboard:drift] "
    #         f"target_ik_est={np.round(target_ik_pos, 6).tolist()} "
    #         f"before_ik={np.round(before_ik_pos, 6).tolist()} "
    #         f"after_ik={np.round(after_ik_pos, 6).tolist()} "
    #         f"ik_actual_delta={np.round(after_ik_pos - before_ik_pos, 6).tolist()} "
    #         f"ik_target_error={np.round(target_ik_pos - after_ik_pos, 6).tolist()} "
    #         f"ik_xy_error_norm={np.linalg.norm(target_ik_pos[:2] - after_ik_pos[:2]):.6f}"
    #     )

    # grasp_pos[:2] = approach_pos[:2]
    # print(
    #     "Finished XY tuning: "
    #     f"approach={np.round(approach_pos, 6).tolist()} "
    #     f"grasp={np.round(grasp_pos, 6).tolist()}"
    # )

    # slow_close_gripper(robot)
    # print("Press Enter when the gripper is at the insertion pose to record it...")
    # input()
    record = record_droid_insertion_pose(robot)
    # if hasattr(robot, "_recorded_insertion_pose"):
    #     robot._recorded_insertion_pose = record
    #     robot._pose_target_pos = None
    #     robot._pose_target_origin_pos = None
    #     robot._pose_target_rot = None
    # lift_m = float(np.random.uniform(0.02, 0.04))
    # print(
    #     "Randomized post-insertion lift: "
    #     f"z={lift_m:.4f} m"
    # )
    # lift_from_recorded_insertion_pose(robot, record, lift_m=lift_m)
    # print("Ready for Collecting Data. press enter to continue")
    # input()
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
                print("Script Step")
                insert_meta_data = run_scripted_grasp_sequence(robot)
                if insert_meta_data.get("debug_lift_only", False):
                    print("Approach lift debug complete; stopping before teleop_step commands.")
                    break

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
