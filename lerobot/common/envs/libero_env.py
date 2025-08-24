import os
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import robosuite.utils.transform_utils as T
from robosuite.utils.camera_utils import get_real_depth_map


class LiberoEnv(gym.Env):
    """Gymnasium wrapper for LIBERO environments."""
    
    def __init__(
        self,
        task_suite_name: str = "libero_object",
        task_id: int = 0,
        camera_heights: int = 128,
        camera_widths: int = 128,
        obs_type: str = "pixels_agent_pos",
        max_episode_steps: int = 300,
        render_mode: Optional[str] = None,
        env_idx: int = 0,  # For vectorized envs - each should start with different init state
        **kwargs
    ):
        super().__init__()
        
        self.task_suite_name = task_suite_name
        self.task_id = task_id
        self.camera_heights = camera_heights
        self.camera_widths = camera_widths
        self.obs_type = obs_type
        self._max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        
        # Initialize LIBERO benchmark
        benchmark_dict = benchmark.get_benchmark_dict()
        self.task_suite = benchmark_dict[task_suite_name]()
        
        # Get task info
        self.libero_task = self.task_suite.get_task(task_id)
        self.task_name = self.libero_task.name
        self.task_description = self.libero_task.language
        self.task_bddl_file = os.path.join(
            get_libero_path("bddl_files"), 
            self.libero_task.problem_folder, 
            self.libero_task.bddl_file
        )
        
        # Initialize environment
        env_args = {
            "bddl_file_name": self.task_bddl_file,
            "camera_heights": camera_heights,
            "camera_widths": camera_widths,
            "camera_depths": True,
        }
        self.env = OffScreenRenderEnv(**env_args)
        
        # Get init states for benchmarking
        self.init_states = self.task_suite.get_task_init_states(task_id)

        # Set up action and observation spaces
        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),  # LIBERO uses 7-DOF delta actions
            dtype=np.float32
        )
        
        # Define observation space to match preprocess_observation expectations
        if obs_type == "pixels":
            self.observation_space = spaces.Dict({
                "pixels": spaces.Dict({
                    "agentview": spaces.Box(
                        low=0,
                        high=255,
                        shape=(camera_heights, camera_widths, 3),
                        dtype=np.uint8
                    ),
                    "wristview": spaces.Box(
                        low=0,
                        high=255,
                        shape=(camera_heights, camera_widths, 3),
                        dtype=np.uint8
                    )
                }),
                "depth": spaces.Dict({
                    "agentview": spaces.Box(
                        low=0,
                        high=np.inf,
                        shape=(camera_heights, camera_widths),
                        dtype=np.float32
                    )
                })
            })
        elif obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict({
                "pixels": spaces.Dict({
                    "agentview": spaces.Box(
                        low=0,
                        high=255,
                        shape=(camera_heights, camera_widths, 3),
                        dtype=np.uint8
                    ),
                    "wristview": spaces.Box(
                        low=0,
                        high=255,
                        shape=(camera_heights, camera_widths, 3),
                        dtype=np.uint8
                    )
                }),
                "depth": spaces.Dict({
                    "agentview": spaces.Box(
                        low=0,
                        high=np.inf,
                        shape=(camera_heights, camera_widths),
                        dtype=np.float32
                    )
                }),
                "agent_pos": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(8,),
                    dtype=np.float32
                ),
                "robot_data": spaces.Dict({
                    "ee_pos": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(3,),
                        dtype=np.float32
                    ),
                    "ee_quat": spaces.Box(
                        low=-1.0,
                        high=1.0,
                        shape=(4,),
                        dtype=np.float32
                    ),
                    "gripper_angle": spaces.Box(
                        low=0.0,
                        high=0.04,
                        shape=(),
                        dtype=np.float32
                    )
                })
            })
        else:
            raise ValueError(f"Unsupported obs_type: {obs_type}")
        
        self.current_step = 0
        # Start each env in vector with different init state for diversity
        self.current_init_state_id = env_idx % len(self.init_states)
        
        # Add metadata for compatibility with eval script  
        self.metadata = {"render_fps": 30}  # Use default fps
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        if seed is not None:
            self.env.seed(seed)
        
        self.env.reset()
        
        # Use different init states for evaluation diversity
        if options and "init_state_id" in options:
            init_state_id = options["init_state_id"]
        else:
            init_state_id = self.current_init_state_id % len(self.init_states)
            self.current_init_state_id += 1
        
        self.env.set_init_state(self.init_states[init_state_id])
        self.current_step = 0
        
        # Get initial observation
        obs = self._get_observation()
        info = {
            "task_description": self.task_description,
            "task_name": self.task_name,
            "init_state_id": init_state_id
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        # Ensure action is the right shape and type
        action = np.array(action, dtype=np.float32)
        if action.shape != (7,):
            raise ValueError(f"Action must have shape (7,), got {action.shape}")
        
        # Step environment
        raw_obs, reward, done, info = self.env.step(action.tolist())
        
        self.current_step += 1
        
        # Check if episode is done due to max steps
        truncated = self.current_step >= self._max_episode_steps
        terminated = bool(done)
        
        # Format observation to match preprocess_observation expectations
        obs = {}
        
        if "pixels" in self.obs_type:
            # Return pixels as dict with camera names as keys
            obs["pixels"] = {
                "agentview": raw_obs["agentview_image"].astype(np.uint8),
                "wristview": raw_obs["robot0_eye_in_hand_image"].astype(np.uint8)
            }
            # Add depth information for agentview
            # Convert to metric depth
            metric_depth = get_real_depth_map(self.env.sim, raw_obs["agentview_depth"])
            depth_data = metric_depth.astype(np.float32)
            # Ensure depth has correct shape (H, W) - squeeze if needed
            if depth_data.ndim == 3 and depth_data.shape[-1] == 1:
                depth_data = depth_data.squeeze(-1)
            obs["depth"] = {
                "agentview": depth_data
            }
        
        if "agent_pos" in self.obs_type:
            # Match dataset format: ee_pos (3D) + ee_ori (3D) + gripper_states (2D) = 8D
            ee_pos = raw_obs["robot0_eef_pos"]  # 3D position
            ee_quat = raw_obs["robot0_eef_quat"]  # 4D quaternion
            ee_ori = T.quat2axisangle(ee_quat)  # Convert to 3D axis-angle
            gripper_states = raw_obs["robot0_gripper_qpos"]  # 2D gripper positions
            # Concatenate: pos(3) + ori(3) + gripper(2) = 8D to match dataset
            obs["agent_pos"] = np.concatenate([ee_pos, ee_ori, gripper_states]).astype(np.float32)

            obs["robot_data"] = {
                "ee_pos": ee_pos,
                "ee_quat": ee_quat,
                "gripper_angle": gripper_states[0]
            }

        
        # Add success indicator for evaluation
        # LIBERO tasks give sparse reward=1.0 for success
        info["is_success"] = bool(reward == 1.0)
        
        return obs, float(reward), terminated, truncated, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        # Get observation by taking a dummy step
        dummy_action = [0.0] * 7
        raw_obs, _, _, _ = self.env.step(dummy_action)

        obs = {}
        
        if "pixels" in self.obs_type:
            # Return pixels as dict with camera names as keys
            obs["pixels"] = {
                "agentview": raw_obs["agentview_image"].astype(np.uint8),
                "wristview": raw_obs["robot0_eye_in_hand_image"].astype(np.uint8)
            }
            # Add depth information for agentview only
            # Convert to metric depth
            metric_depth = get_real_depth_map(self.env.sim, raw_obs["agentview_depth"])
            depth_data = metric_depth.astype(np.float32)
            # Ensure depth has correct shape (H, W) - squeeze if needed
            if depth_data.ndim == 3 and depth_data.shape[-1] == 1:
                depth_data = depth_data.squeeze(-1)
            obs["depth"] = {
                "agentview": depth_data
            }
        
        if "agent_pos" in self.obs_type:
            # Match dataset format: ee_pos (3D) + ee_ori (3D) + gripper_states (2D) = 8D
            ee_pos = raw_obs["robot0_eef_pos"]  # 3D position
            ee_quat = raw_obs["robot0_eef_quat"]  # 4D quaternion
            ee_ori = T.quat2axisangle(ee_quat)  # Convert to 3D axis-angle
            gripper_states = raw_obs["robot0_gripper_qpos"]  # 2D gripper positions
            # Concatenate: pos(3) + ori(3) + gripper(2) = 8D to match dataset
            obs["agent_pos"] = np.concatenate([ee_pos, ee_ori, gripper_states]).astype(np.float32)

            obs["robot_data"] = {
                "ee_pos": ee_pos,
                "ee_quat": ee_quat,
                "gripper_angle": gripper_states[0]
            }

        return obs
    
    def render(self):
        if self.render_mode == "rgb_array":
            obs = self._get_observation()
            if "pixels" in obs and "agentview" in obs["pixels"]:
                return obs["pixels"]["agentview"]
        return None
    
    def close(self):
        if hasattr(self, 'env'):
            self.env.close()
    
    @property
    def task(self) -> str:
        """For compatibility with lerobot env interface"""
        return f"{self.task_suite_name}_{self.task_id}"
    
    def task_description_method(self) -> str:
        """For compatibility with lerobot env interface"""
        return self.task_description
