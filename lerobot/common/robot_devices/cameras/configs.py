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
from dataclasses import dataclass

import draccus


@dataclass
class CameraConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@CameraConfig.register_subclass("opencv")
@dataclass
class OpenCVCameraConfig(CameraConfig):
    """
    Example of tested options for Intel Real Sense D405:

    ```python
    OpenCVCameraConfig(0, 30, 640, 480)
    OpenCVCameraConfig(0, 60, 640, 480)
    OpenCVCameraConfig(0, 90, 640, 480)
    OpenCVCameraConfig(0, 30, 1280, 720)
    ```
    """

    camera_index: int
    fps: int | None = None
    width: int | None = None
    height: int | None = None
    color_mode: str = "rgb"
    channels: int | None = None
    rotation: int | None = None
    mock: bool = False

    def __post_init__(self):
        if self.color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"`color_mode` is expected to be 'rgb' or 'bgr', but {self.color_mode} is provided."
            )

        self.channels = 3

        if self.rotation not in [-90, None, 90, 180]:
            raise ValueError(f"`rotation` must be in [-90, None, 90, 180] (got {self.rotation})")


@CameraConfig.register_subclass("intelrealsense")
@dataclass
class IntelRealSenseCameraConfig(CameraConfig):
    """
    Example of tested options for Intel Real Sense D405:

    ```python
    IntelRealSenseCameraConfig(128422271347, 30, 640, 480)
    IntelRealSenseCameraConfig(128422271347, 60, 640, 480)
    IntelRealSenseCameraConfig(128422271347, 90, 640, 480)
    IntelRealSenseCameraConfig(128422271347, 30, 1280, 720)
    IntelRealSenseCameraConfig(128422271347, 30, 640, 480, use_depth=True)
    IntelRealSenseCameraConfig(128422271347, 30, 640, 480, rotation=90)
    ```
    """

    name: str | None = None
    serial_number: int | None = None
    fps: int | None = None
    width: int | None = None
    height: int | None = None
    color_mode: str = "rgb"
    channels: int | None = None
    use_depth: bool = False
    force_hardware_reset: bool = True
    rotation: int | None = None
    mock: bool = False

    def __post_init__(self):
        # bool is stronger than is None, since it works with empty strings
        if bool(self.name) and bool(self.serial_number):
            raise ValueError(
                f"One of them must be set: name or serial_number, but {self.name=} and {self.serial_number=} provided."
            )

        if self.color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"`color_mode` is expected to be 'rgb' or 'bgr', but {self.color_mode} is provided."
            )

        self.channels = 3

        at_least_one_is_not_none = self.fps is not None or self.width is not None or self.height is not None
        at_least_one_is_none = self.fps is None or self.width is None or self.height is None
        if at_least_one_is_not_none and at_least_one_is_none:
            raise ValueError(
                "For `fps`, `width` and `height`, either all of them need to be set, or none of them, "
                f"but {self.fps=}, {self.width=}, {self.height=} were provided."
            )

        if self.rotation not in [-90, None, 90, 180]:
            raise ValueError(f"`rotation` must be in [-90, None, 90, 180] (got {self.rotation})")

@CameraConfig.register_subclass("azurekinect")
@dataclass
class AzureKinectCameraConfig(CameraConfig):
    """
    Example of options for Azure Kinect:
    ```python
    AzureKinectCameraConfig(0, 30, 1920, 1080)
    AzureKinectCameraConfig(0, 30, 1280, 720)
    AzureKinectCameraConfig(0, 15, 1920, 1080, use_depth=True)
    AzureKinectCameraConfig(0, 30, 1920, 1080, rotation=90)
    ```
    """
    device_id: int
    fps: int | None = None
    width: int | None = None
    height: int | None = None
    color_mode: str = "rgb"
    channels: int | None = None
    use_depth: bool = False
    use_ir: bool = False
    use_transformed_depth: bool = False
    use_point_cloud: bool = False
    use_transformed_color: bool = False
    rotation: int | None = None
    mock: bool = False

    def __post_init__(self):
        if self.color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"`color_mode` is expected to be 'rgb' or 'bgr', but {self.color_mode} is provided."
            )

        self.channels = 3

        # Validate fps for Azure Kinect constraints
        if self.fps is not None and self.fps not in [5, 15, 30]:
            raise ValueError(f"Azure Kinect fps must be 5, 15, or 30, got {self.fps}")

        # Similar to IntelRealSense: either all of fps/width/height are set, or none
        at_least_one_is_not_none = self.fps is not None or self.width is not None or self.height is not None
        at_least_one_is_none = self.fps is None or self.width is None or self.height is None
        if at_least_one_is_not_none and at_least_one_is_none:
            raise ValueError(
                "For `fps`, `width` and `height`, either all of them need to be set, or none of them, "
                f"but {self.fps=}, {self.width=}, {self.height=} were provided."
            )

        # Validate resolution combinations when provided
        if self.width is not None and self.height is not None:
            valid_resolutions = [
                (1280, 720),   # 720p
                (1920, 1080),  # 1080p
                (2560, 1440),  # 1440p
                (2048, 1536),  # 1536p
                (3840, 2160),  # 3072p (4K)
            ]

            if (self.width, self.height) not in valid_resolutions:
                raise ValueError(
                    f"Invalid resolution {self.width}x{self.height}. "
                    f"Valid resolutions: {valid_resolutions}"
                )
        if self.rotation not in [-90, None, 90, 180]:
            raise ValueError(f"`rotation` must be in [-90, None, 90, 180] (got {self.rotation})")

        # Additional validation for Azure Kinect specific features
        if self.use_transformed_color and not (self.use_depth or self.use_transformed_depth):
            raise ValueError(
                "use_transformed_color requires depth data. Enable use_depth or use_transformed_depth."
            )

        if self.use_point_cloud and not self.use_depth:
            raise ValueError(
                "use_point_cloud requires depth data. Enable use_depth."
            )