"""ArticuBot manipulation environment wrapper for lerobot.

Wraps the ArticuBot PyBullet-based articulated object manipulation environment
(Franka Panda robot opening doors, faucets, etc.) into a gymnasium.Env that
exposes observations in lerobot's format (pixels, depth, agent_pos).

Modeled after LiberoEnv.  The camera/observation logic is ported from
RobogenImageWrapper (which is NOT copied into this repo).
"""

import os
import copy
import yaml
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

from lerobot.common.envs.articubot.utils import (
    _resolve_config_dir,
    depth_buffer_to_metric,
    proj_matrix_to_intrinsics,
    rotation_transfer_6D_to_matrix,
    rotation_transfer_matrix_to_6D,
    save_env,
    load_env,
)


class ArticuBotEnv(gym.Env):
    """Gymnasium wrapper for ArticuBot articulated-object manipulation environments.

    Internally creates a PyBullet ``articulated`` env (door/faucet/etc.) and
    sets up 2 fixed cameras + 1 wrist camera (ported from RobogenImageWrapper).

    Observation dict (lerobot format)::

        {
            "pixels": {
                "cam0": (H, W, 3) uint8,   # fixed view 1
                "cam1": (H, W, 3) uint8,   # fixed view 2
                "wrist": (H, W, 3) uint8,  # wrist camera
            },
            "depth": {
                "cam0": (H, W) float32,     # metric depth
                "cam1": (H, W) float32,
                "wrist": (H, W) float32,
            },
            "agent_pos": (10,) float32,     # EEF pos(3) + 6D rot(6) + gripper(1)
        }

    Action space: ``Box(-1, 1, shape=(10,), dtype=float32)``
        action[0:3]  – EEF position delta
        action[3:9]  – 6D rotation delta
        action[9]    – gripper joint delta
    """

    metadata = {"render_fps": 30}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        task_config: str = "",
        task_name: str = "articulated",
        object_name: str = "storagefurniture",
        link_name: str = "link_0",
        init_angle: Optional[float] = None,
        init_state_file: str = "",
        camera_height: int = 256,
        camera_width: int = 256,
        obs_type: str = "pixels_agent_pos",
        max_episode_steps: int = 600,
        render_mode: Optional[str] = None,
        env_idx: int = 0,
        **kwargs,
    ):
        super().__init__()

        self.task_config = task_config
        self.task_name_str = task_name
        self.object_name = object_name.lower()
        self.link_name = link_name
        self.init_angle = init_angle
        self.init_state_file = init_state_file if init_state_file else None
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.obs_type = obs_type
        self._max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.env_idx = env_idx

        # ---- build inner env (articulated) ----
        self._env = self._build_inner_env()

        # ---- cameras (ported from RobogenImageWrapper.__init__) ----
        self.depth_near = 0.01
        self.depth_far = 100.0
        self._setup_cameras()

        # ---- spaces ----
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32
        )

        obs_dict: dict[str, Any] = {}
        if "pixels" in self.obs_type:
            obs_dict["pixels"] = spaces.Dict({
                "cam0": spaces.Box(0, 255, (camera_height, camera_width, 3), np.uint8),
                "cam1": spaces.Box(0, 255, (camera_height, camera_width, 3), np.uint8),
                "wrist": spaces.Box(0, 255, (camera_height, camera_width, 3), np.uint8),
            })
            obs_dict["depth"] = spaces.Dict({
                "cam0": spaces.Box(0, np.inf, (camera_height, camera_width), np.float32),
                "cam1": spaces.Box(0, np.inf, (camera_height, camera_width), np.float32),
                "wrist": spaces.Box(0, np.inf, (camera_height, camera_width), np.float32),
            })
        if "agent_pos" in self.obs_type:
            obs_dict["agent_pos"] = spaces.Box(
                -np.inf, np.inf, (10,), np.float32
            )
            obs_dict["robot_data"] = spaces.Dict({
                "ee_pos": spaces.Box(-np.inf, np.inf, (3,), np.float64),
                "ee_quat": spaces.Box(-np.inf, np.inf, (4,), np.float64),
                "gripper_angle": spaces.Box(-np.inf, np.inf, (), np.float64),
            })
        self.observation_space = spaces.Dict(obs_dict)

        self.current_step = 0

        # Intrinsics (for potential downstream use)
        self.intrinsics = np.array([
            proj_matrix_to_intrinsics(pm, self.camera_width, self.camera_height)
            for pm in (self.project_matrices + [self.wrist_project_matrix])
        ])

    # ------------------------------------------------------------------
    # Inner env construction
    # ------------------------------------------------------------------

    def _build_inner_env(self):
        """Create the ``articulated`` PyBullet env."""
        from lerobot.common.envs.articubot.envs.articulated import articulated

        # Resolve relative config paths against the articubot config directory
        task_config_path = self.task_config
        if task_config_path and not os.path.isabs(task_config_path):
            resolved = os.path.join(_resolve_config_dir(), task_config_path)
            if os.path.exists(resolved):
                task_config_path = resolved
        self.task_config = task_config_path

        config = yaml.safe_load(open(self.task_config, "r"))
        # Extract object_name, link_name, init_angle from config
        for cfg_dict in config:
            if "name" in cfg_dict:
                self.object_name = cfg_dict["name"].lower()
            if "link_name" in cfg_dict:
                self.link_name = cfg_dict["link_name"]
            if "init_angle" in cfg_dict and self.init_angle is None:
                self.init_angle = cfg_dict["init_angle"]

        env_kwargs: Dict[str, Any] = {
            "gui": False,
            "config_path": self.task_config,
            "task_name": self.task_name_str,
            "object_name": self.object_name,
            "link_name": self.link_name,
            "init_angle": self.init_angle,
            "restore_state_file": self.init_state_file,
        }
        env = articulated(**env_kwargs)
        return env

    # ------------------------------------------------------------------
    # Camera setup (ported from RobogenImageWrapper.__init__)
    # ------------------------------------------------------------------

    def _setup_cameras(self):
        """Set up 2 fixed cameras + wrist projection (from RobogenImageWrapper)."""
        # Find the first non-robot, non-plane object to center cameras on
        for name in self._env.urdf_ids:
            if name in ("robot", "plane", "init_table"):
                continue
            obj_id = self._env.urdf_ids[name]
            min_aabb, max_aabb = self._env.get_aabb(obj_id)
            center = (min_aabb + max_aabb) / 2
            self.mean_camera_target = center
            self.mean_distance = np.linalg.norm(max_aabb - min_aabb) * 0.9
            break

        rpy_mean_list = [[-10, 0, -45], [-10, 0, -135]]
        self.view_matrices = []
        self.project_matrices = []
        for rpy_mean in rpy_mean_list:
            rpy = np.array(rpy_mean, dtype=float)
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=self.mean_camera_target,
                distance=self.mean_distance,
                yaw=rpy[2], pitch=rpy[0], roll=rpy[1],
                upAxisIndex=2,
                physicsClientId=self._env.id,
            )
            project_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=1,
                nearVal=self.depth_near, farVal=self.depth_far,
                physicsClientId=self._env.id,
            )
            self.view_matrices.append(view_matrix)
            self.project_matrices.append(project_matrix)

        self.wrist_project_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=1,
            nearVal=self.depth_near, farVal=self.depth_far,
            physicsClientId=self._env.id,
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)

        self._env.reset()
        self._env._get_info()
        self.current_step = 0

        obs = self._build_observation()
        info = {
            "task_description": f"articubot_{self.object_name}",
            "task_name": self.task_name_str,
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)

        # ---- action processing (from RobogenImageWrapper.step) ----
        pos, orient = self._env.robot.get_pos_orient(
            self._env.robot.right_end_effector
        )

        # new_pos = current_pos + delta
        new_pos = pos + np.array(action[:3])

        # new_orient = current_orient @ delta_orient (6D)
        current_rot = np.array(p.getMatrixFromQuaternion(orient)).reshape(3, 3)
        delta_rot = rotation_transfer_6D_to_matrix(action[3:9])
        after_rot = current_rot @ delta_rot
        new_orient_quat = R.from_matrix(after_rot).as_quat()
        euler = p.getEulerFromQuaternion(new_orient_quat)

        # new finger = current + delta
        cur_joint_angle = p.getJointState(
            self._env.robot.body,
            self._env.robot.right_gripper_indices[0],
            physicsClientId=self._env.id,
        )
        target_joint_angle = action[9] + cur_joint_angle[0]

        inner_action = list(new_pos) + list(euler) + [target_joint_angle]
        self._env.take_direct_action(inner_action)

        reward, success = self._env.compute_reward()
        info = self._env._get_info()

        self.current_step += 1
        truncated = self.current_step >= self._max_episode_steps
        terminated = False

        obs = self._build_observation()

        info["is_success"] = bool(success)
        return obs, float(reward), terminated, truncated, info

    def render(self):
        """Return RGB from the first fixed camera."""
        if not self.view_matrices:
            return None
        w, h, img, _, _ = p.getCameraImage(
            self.camera_width, self.camera_height,
            self.view_matrices[0], self.project_matrices[0],
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self._env.id,
        )
        return np.reshape(img, (h, w, 4))[:, :, :3]

    def close(self):
        if hasattr(self, "_env"):
            self._env.close()

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    def _build_observation(self) -> Dict[str, Any]:
        """Capture images from all cameras and build lerobot-format obs dict."""
        rgbs, depths = self._take_images()
        obs: Dict[str, Any] = {}

        if "pixels" in self.obs_type:
            obs["pixels"] = {
                "cam0": rgbs[0].astype(np.uint8),
                "cam1": rgbs[1].astype(np.uint8),
                "wrist": rgbs[2].astype(np.uint8),
            }
            metric_depths = depth_buffer_to_metric(
                np.array(depths, dtype=np.float32),
                self.depth_near,
                self.depth_far,
            )
            obs["depth"] = {
                "cam0": metric_depths[0].astype(np.float32),
                "cam1": metric_depths[1].astype(np.float32),
                "wrist": metric_depths[2].astype(np.float32),
            }

        if "agent_pos" in self.obs_type:
            pos, orient = self._env.robot.get_pos_orient(
                self._env.robot.right_end_effector
            )
            rot_mat = p.getMatrixFromQuaternion(orient)
            orient_6d = rotation_transfer_matrix_to_6D(rot_mat)

            cur_left = p.getJointState(
                self._env.robot.body,
                self._env.robot.right_gripper_indices[0],
                physicsClientId=self._env.id,
            )
            cur_right = p.getJointState(
                self._env.robot.body,
                self._env.robot.right_gripper_indices[1],
                physicsClientId=self._env.id,
            )
            gripper_angle = cur_left[0] + cur_right[0]

            agent_pos = np.concatenate([
                pos, orient_6d, [gripper_angle]
            ]).astype(np.float32)
            obs["agent_pos"] = agent_pos

            obs["robot_data"] = {
                "ee_pos": np.array(pos, dtype=np.float64),
                "ee_quat": np.array(orient, dtype=np.float64),
                "gripper_angle": np.float64(gripper_angle),
            }

        return obs

    # ------------------------------------------------------------------
    # Camera image capture (ported from RobogenImageWrapper)
    # ------------------------------------------------------------------

    def _take_images(self):
        """Capture RGB + depth from 2 fixed cameras and 1 wrist camera.

        Ported from ``RobogenImageWrapper.take_images_around_object()``.
        """
        rgbs = []
        depths = []

        for view_matrix, project_matrix in zip(
            self.view_matrices, self.project_matrices
        ):
            w, h, img, depth, _ = p.getCameraImage(
                self.camera_width, self.camera_height,
                view_matrix, project_matrix,
                flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
                physicsClientId=self._env.id,
            )
            rgbs.append(np.reshape(img, (h, w, 4))[:, :, :3])
            depths.append(np.reshape(depth, (h, w)))

        # Wrist camera
        wrist_view = self._get_wrist_view_matrix()
        w, h, img, depth, _ = p.getCameraImage(
            self.camera_width, self.camera_height,
            wrist_view, self.wrist_project_matrix,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self._env.id,
        )
        rgbs.append(np.reshape(img, (h, w, 4))[:, :, :3])
        depths.append(np.reshape(depth, (h, w)))

        return rgbs, depths

    def _get_wrist_view_matrix(self):
        """Compute wrist camera view matrix (ported from RobogenImageWrapper)."""
        hand_state = p.getLinkState(self._env.robot.body, 7)
        hand_pos = np.array(hand_state[0])
        hand_ori = np.array(hand_state[1])

        local_pos_offset = np.array([0.0, -0.1, -0.1])
        local_quat = R.from_euler("xyz", [-20, 0, 45], degrees=True).as_quat()

        hand_rot = R.from_quat(hand_ori)
        global_eye = hand_pos + hand_rot.apply(
            R.from_quat(local_quat).apply(local_pos_offset)
        )
        local_look_dir = np.array([0, 0, 1.0])
        combined_rot = hand_rot * R.from_quat(local_quat)
        global_look_dir = combined_rot.apply(local_look_dir)
        global_target = global_eye + global_look_dir
        global_up = combined_rot.apply(np.array([0, -1, 0]))

        return p.computeViewMatrix(global_eye, global_target, global_up)

    # ------------------------------------------------------------------
    # Properties for compatibility
    # ------------------------------------------------------------------

    @property
    def task_description(self) -> str:
        return f"articubot_{self.object_name}"
