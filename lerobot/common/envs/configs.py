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

import abc
from dataclasses import dataclass, field

import draccus

from lerobot.common.constants import ACTION, OBS_ENV, OBS_IMAGE, OBS_IMAGES, OBS_ROBOT
from lerobot.configs.types import FeatureType, PolicyFeature


@dataclass
class EnvConfig(draccus.ChoiceRegistry, abc.ABC):
    task: str | None = None
    fps: int = 30
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @abc.abstractproperty
    def gym_kwargs(self) -> dict:
        raise NotImplementedError()


@EnvConfig.register_subclass("aloha")
@dataclass
class AlohaEnv(EnvConfig):
    task: str = "AlohaInsertion-v0"
    fps: int = 50
    episode_length: int = 400
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_ROBOT,
            "top": f"{OBS_IMAGE}.top",
            "pixels/top": f"{OBS_IMAGES}.top",
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels":
            self.features["top"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3))
        elif self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(14,))
            self.features["pixels/top"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
        }


@EnvConfig.register_subclass("pusht")
@dataclass
class PushtEnv(EnvConfig):
    task: str = "PushT-v0"
    fps: int = 10
    episode_length: int = 300
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    visualization_width: int = 384
    visualization_height: int = 384
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
            "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_ROBOT,
            "environment_state": OBS_ENV,
            "pixels": OBS_IMAGE,
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels_agent_pos":
            self.features["pixels"] = PolicyFeature(type=FeatureType.VISUAL, shape=(384, 384, 3))
        elif self.obs_type == "environment_state_agent_pos":
            self.features["environment_state"] = PolicyFeature(type=FeatureType.ENV, shape=(16,))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "visualization_width": self.visualization_width,
            "visualization_height": self.visualization_height,
            "max_episode_steps": self.episode_length,
        }


@EnvConfig.register_subclass("xarm")
@dataclass
class XarmEnv(EnvConfig):
    task: str = "XarmLift-v0"
    fps: int = 15
    episode_length: int = 200
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    visualization_width: int = 384
    visualization_height: int = 384
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
            "pixels": PolicyFeature(type=FeatureType.VISUAL, shape=(84, 84, 3)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_ROBOT,
            "pixels": OBS_IMAGE,
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(4,))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "visualization_width": self.visualization_width,
            "visualization_height": self.visualization_height,
            "max_episode_steps": self.episode_length,
        }


@EnvConfig.register_subclass("libero")
@dataclass
class LiberoEnv(EnvConfig):
    task: str = "libero_object_0"  # Format: {suite_name}_{task_id}
    task_suite_name: str = "libero_object"
    task_id: int = 0

    fps: int = 30
    episode_length: int = 300
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    camera_heights: int = 256
    camera_widths: int = 256
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_ROBOT,
            "pixels": OBS_IMAGE,
        }
    )

    def __post_init__(self):
        # Parse task suite and id from task string
        parts = self.task.split("_")
        assert len(parts) == 3
        self.task_suite_name = "_".join(parts[:-1])
        self.task_id = int(parts[-1])

        if self.obs_type == "pixels":
            self.features["pixels"] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(self.camera_heights, self.camera_widths, 3)
            )
        elif self.obs_type == "pixels_agent_pos":
            self.features["pixels"] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(self.camera_heights, self.camera_widths, 3)
            )
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(8,))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "task_suite_name": self.task_suite_name,
            "task_id": self.task_id,
            "camera_heights": self.camera_heights,
            "camera_widths": self.camera_widths,
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
        }


@EnvConfig.register_subclass("articubot")
@dataclass
class ArticuBotEnv(EnvConfig):
    task: str = "articubot_storagefurniture"
    fps: int = 30
    episode_length: int = 600
    obs_type: str = "pixels_agent_pos"
    camera_height: int = 256
    camera_width: int = 256
    task_config: str = ""  # path to YAML task config
    init_state_file: str = ""  # path to initial state pkl
    object_name: str = "storagefurniture"
    link_name: str = "link_0"
    init_angle: float | None = None
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(10,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_ROBOT,
            "pixels/cam0": f"{OBS_IMAGES}.cam0",
            "pixels/cam1": f"{OBS_IMAGES}.cam1",
            "pixels/wrist": f"{OBS_IMAGES}.wrist",
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(
                type=FeatureType.STATE, shape=(10,)
            )
            self.features["pixels/cam0"] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(self.camera_height, self.camera_width, 3),
            )
            self.features["pixels/cam1"] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(self.camera_height, self.camera_width, 3),
            )
            self.features["pixels/wrist"] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(self.camera_height, self.camera_width, 3),
            )

    @property
    def gym_kwargs(self) -> dict:
        return {
            "task_config": self.task_config,
            "init_state_file": self.init_state_file,
            "object_name": self.object_name,
            "link_name": self.link_name,
            "init_angle": self.init_angle,
            "camera_height": self.camera_height,
            "camera_width": self.camera_width,
            "obs_type": self.obs_type,
            "max_episode_steps": self.episode_length,
        }
