from abc import ABC, abstractmethod
from typing import Dict
import torch
import pytorch3d.transforms as transforms

class RobotAdapter(ABC):
    """Abstract interface for robot-specific logic"""

    @abstractmethod
    def get_obs_key(self) -> str:
        pass

    @abstractmethod
    def get_act_key(self) -> str:
        pass

    @abstractmethod
    def transform_action(self, action: torch.Tensor, state: torch.Tensor, reference_eef: torch.Tensor | None = None) -> torch.Tensor:
        """Transform policy output to robot-executable action"""
        pass

    @abstractmethod
    def get_eef_action(self, action: torch.Tensor, state: torch.Tensor, reference_eef: torch.Tensor | None = None) -> torch.Tensor:
        """Get eef action info """
        pass

    @abstractmethod
    def compute_relative_actions(self, batch: dict) -> dict:
        """Compute relative actions. Override in subclass if needed."""
        pass

class AlohaAdapter(RobotAdapter):
    def __init__(self, action_space: str):
        assert action_space in ["right_eef", "joint", "right_eef_relative"]
        self.action_space = action_space
        from lerobot.common.utils.aloha_utils import ALOHA_CONFIGURATION, ALOHA_REST_STATE
        self.config = ALOHA_CONFIGURATION
        self.rest_state = ALOHA_REST_STATE

    def get_obs_key(self) -> str:
        if self.action_space in ["right_eef", "right_eef_relative"]:
            return "observation.right_eef_pose"
        return "observation.state"

    def get_act_key(self) -> str:
        if self.action_space == "right_eef":
            return "action.right_eef_pose"
        elif self.action_space == "right_eef_relative":
            return "action.right_eef_pose_relative"
        return "action"

    def _relative_to_absolute_eef(self, relative_action: torch.Tensor, current_eef: torch.Tensor) -> torch.Tensor:
        """Convert relative EEF action to absolute
        Args:
            relative_action: (1, 10) [6D_rot, xyz, gripper] relative delta
            current_eef: (10,) [6D_rot, xyz, gripper] current absolute pose
        Returns:
            absolute_action: (1, 10) [6D_rot, xyz, gripper] absolute pose
        """
        rel_rot6d, rel_pos, rel_gripper = relative_action[0, :6], relative_action[0, 6:9], relative_action[0, 9:10]
        obs_rot6d, obs_pos, obs_gripper = current_eef[:6], current_eef[6:9], current_eef[9:10]

        R_obs = transforms.rotation_6d_to_matrix(obs_rot6d[None]).squeeze()  # (3, 3)
        R_rel = transforms.rotation_6d_to_matrix(rel_rot6d[None]).squeeze()  # (3, 3)
        R_abs = torch.matmul(R_obs, R_rel)  # (3, 3)

        abs_rot6d = transforms.matrix_to_rotation_6d(R_abs[None]).squeeze()  # (6,)
        abs_pos = obs_pos + rel_pos  # (3,)
        abs_gripper = obs_gripper + rel_gripper  # (1,)

        return torch.cat([abs_rot6d, abs_pos, abs_gripper])[None]  # (1, 10)

    def compute_relative_actions(self, batch: dict) -> dict:
        """Compute relative actions (delta from current observation).
        Args:
            batch: Dictionary containing observations and actions
                - observation.right_eef_pose: (B, n_obs_steps, 10) absolute EEF poses [6D_rot, xyz, gripper]
                - action.right_eef_pose: (B, horizon, 10) absolute EEF poses [6D_rot, xyz, gripper]

        Returns:
            batch: Modified batch with new relative action key
                - action.right_eef_pose_relative: (B, horizon, 10) relative deltas from current obs
        """
        obs_key = self.get_obs_key()
        source_act_key = "action.right_eef_pose"
        target_act_key = self.get_act_key()

        # Use the last observation as the reference point
        current_obs = batch[obs_key][:, -1, :]  # (B, 10)
        actions = batch[source_act_key]  # (B, horizon, 10)

        B, horizon, _ = actions.shape

        obs_rot6d, obs_pos, obs_gripper = current_obs[:, :6], current_obs[:, 6:9], current_obs[:, 9:10]
        act_rot6d, act_pos, act_gripper = actions[:, :, :6], actions[:, :, 6:9], actions[:, :, 9:10]

        # Convert 6D rotations to rotation matrices
        R_obs = transforms.rotation_6d_to_matrix(obs_rot6d)  # (B, 3, 3)
        R_act = transforms.rotation_6d_to_matrix(act_rot6d.reshape(B * horizon, 6))  # (B*horizon, 3, 3)
        R_act = R_act.reshape(B, horizon, 3, 3)  # (B, horizon, 3, 3)

        # Compute relative rotation: R_relative = R_obs^T @ R_action
        R_obs_inv = R_obs.transpose(-2, -1)  # (B, 3, 3) - transpose = inverse for rotation
        R_relative = torch.matmul(
            R_obs_inv.unsqueeze(1),  # (B, 1, 3, 3) - broadcast across horizon
            R_act  # (B, horizon, 3, 3)
        )  # (B, horizon, 3, 3)

        # Convert back to 6D
        relative_rot6d = transforms.matrix_to_rotation_6d(R_relative.reshape(B * horizon, 3, 3))
        relative_rot6d = relative_rot6d.reshape(B, horizon, 6)  # (B, horizon, 6)
        relative_pos = act_pos - obs_pos.unsqueeze(1)  # (B, horizon, 3)
        relative_gripper = act_gripper - obs_gripper.unsqueeze(1)  # (B, horizon, 1)

        # Concatenate to form relative action
        batch[target_act_key] = torch.cat(
            [relative_rot6d, relative_pos, relative_gripper], dim=-1
        )  # (B, horizon, 10)
        return batch

    def transform_action(self, action: torch.Tensor, state: torch.Tensor, reference_eef: torch.Tensor | None = None) -> torch.Tensor:
        """Transform policy output to robot-executable action.
        Args:
            action: Policy output action
                - For "right_eef": (10,) absolute EEF pose
                - For "right_eef_relative": (10,) relative EEF delta
                - For "joint": (18,) joint positions
            state: Current robot state (joint positions)
            reference_eef: Reference EEF pose for relative actions (10,) [6D_rot, xyz, gripper]
        Returns:
            Joint positions to execute on robot
        """
        from lerobot.common.utils.aloha_utils import inverse_kinematics

        if self.action_space in ["right_eef", "right_eef_relative"]:
            if self.action_space == "right_eef_relative":
                # For relative actions, convert to absolute using reference EEF
                action = self._relative_to_absolute_eef(action, reference_eef)

            # IK to get joint action
            action_joint = inverse_kinematics(self.config, action.squeeze())[None].float()
            # Force left arm to rest at predefined pose
            action_joint[:, :9] = self.rest_state[:, :9]
            return action_joint
        return action

    def get_eef_action(self, action: torch.Tensor, state: torch.Tensor, reference_eef: torch.Tensor | None = None) -> torch.Tensor:
        if self.action_space == "right_eef":
            return action  # eef action is primary
        elif self.action_space == "right_eef_relative":
            # For relative actions, convert to absolute using reference EEF
            return self._relative_to_absolute_eef(action, reference_eef)

        # For joint space, return dummy eef
        return torch.zeros(10, device=action.device, dtype=action.dtype)

class LiberoFrankaAdapter(RobotAdapter):
    """Libero Franka adapter"""
    def __init__(self, obs_key: str, act_key: str):
        self.obs_key = obs_key
        self.act_key = act_key

    def get_obs_key(self) -> str:
        return self.obs_key

    def get_act_key(self) -> str:
        return self.act_key

    def transform_action(self, action: torch.Tensor, state: torch.Tensor, reference_eef: torch.Tensor | None = None) -> torch.Tensor:
        return action  # No transformation

    def get_eef_action(self, action: torch.Tensor, state: torch.Tensor, reference_eef: torch.Tensor | None = None) -> torch.Tensor:
        return action  # Return same action as auxiliary

    def compute_relative_actions(self, batch: dict) -> dict:
        return batch  # No-op: return unchanged
