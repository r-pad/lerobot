from abc import ABC, abstractmethod
from typing import Dict
import torch

class RobotAdapter(ABC):
    """Abstract interface for robot-specific logic"""

    @abstractmethod
    def get_obs_key(self) -> str:
        pass

    @abstractmethod
    def get_act_key(self) -> str:
        pass

    @abstractmethod
    def transform_action(self, action: torch.Tensor, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Transform policy output to robot-executable action"""
        pass

    @abstractmethod
    def get_eef_action(self, action: torch.Tensor) -> torch.Tensor:
        """Get eef action info """
        pass

class AlohaAdapter(RobotAdapter):
    def __init__(self, action_space: str):
        assert action_space in ["right_eef", "joint"]
        self.action_space = action_space
        from lerobot.common.utils.aloha_utils import ALOHA_CONFIGURATION, ALOHA_REST_STATE
        self.config = ALOHA_CONFIGURATION
        self.rest_state = ALOHA_REST_STATE

    def get_obs_key(self) -> str:
        if self.action_space == "right_eef":
            return "observation.right_eef_pose"
        return "observation.state"

    def get_act_key(self) -> str:
        if self.action_space == "right_eef":
            return "action.right_eef_pose"
        return "action"

    def transform_action(self, action: torch.Tensor, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        from lerobot.common.utils.aloha_utils import forward_kinematics, inverse_kinematics

        if self.action_space == "right_eef":
            # FK on current state
            state = obs['observation.state']
            forward_kinematics(self.config, state[0])

            # IK to get joint action
            action_joint = inverse_kinematics(self.config, action.squeeze())[None].float()
            # Force left arm to rest at predefined pose
            action_joint[:, :9] = self.rest_state[:, :9]
            return action_joint
        return action

    def get_eef_action(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_space == "right_eef":
            return action  # eef action is primary
        # For joint space, return dummy eef
        return torch.zeros(10, device=action.device, dtype=action.dtype)

class GenericAdapter(RobotAdapter):
    """Generic adapter"""
    def __init__(self, obs_key: str, act_key: str):
        self.obs_key = obs_key
        self.act_key = act_key

    def get_obs_key(self) -> str:
        return self.obs_key

    def get_act_key(self) -> str:
        return self.act_key

    def transform_action(self, action: torch.Tensor, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return action  # No transformation

    def get_eef_action(self, action: torch.Tensor) -> torch.Tensor:
        return action  # Return same action as auxiliary
