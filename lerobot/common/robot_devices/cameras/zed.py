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

"""
This file contains utilities for recording frames from ZED stereo cameras.

Requires the ZED SDK and its Python API (`pyzed`). See:
https://www.stereolabs.com/docs/get-started-with-zed
"""

import argparse
import concurrent.futures
import logging
import math
import shutil
import threading
import time
from pathlib import Path
from threading import Thread

import numpy as np
from PIL import Image

from lerobot.common.robot_devices.cameras.configs import ZedCameraConfig
from lerobot.common.robot_devices.utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceNotConnectedError,
    busy_wait,
)
from lerobot.common.utils.utils import capture_timestamp_utc


DEPTH_MODE_MAP = {
    "performance": "PERFORMANCE",
    "quality": "QUALITY",
    "ultra": "ULTRA",
    "neural": "NEURAL",
    "neural_plus": "NEURAL_PLUS",
}

RESOLUTION_MAP = {
    (4416, 1242): "HD2K",
    (3840, 1080): "HD1080",
    (2560, 720): "HD720",
    (1344, 376): "VGA",
    # ZED X / ZED X Mini resolutions
    (3856, 2180): "HD1200",
    (1920, 1080): "HD1080",
    (1280, 720): "HD720",
    (1920, 1200): "HD1200",
}


def find_cameras(raise_when_empty=True) -> list[dict]:
    """
    Find the serial numbers and models of ZED cameras connected to the computer.
    """
    import pyzed.sl as sl

    cameras = []
    device_list = sl.Camera.get_device_list()
    for device in device_list:
        cameras.append(
            {
                "serial_number": device.serial_number,
                "camera_model": str(device.camera_model),
            }
        )

    if raise_when_empty and len(cameras) == 0:
        raise OSError(
            "Not a single ZED camera was detected. Try re-plugging, or re-installing the ZED SDK."
        )

    return cameras


def save_image(img_array, serial_number, frame_index, images_dir):
    try:
        img = Image.fromarray(img_array)
        path = images_dir / f"camera_{serial_number}_frame_{frame_index:06d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path), quality=100)
        logging.info(f"Saved image: {path}")
    except Exception as e:
        logging.error(f"Failed to save image for camera {serial_number} frame {frame_index}: {e}")


def save_images_from_cameras(
    images_dir: Path,
    serial_numbers: list[int] | None = None,
    fps=None,
    width=None,
    height=None,
    record_time_s=2,
):
    """
    Initializes all the cameras and saves images to the directory. Useful to visually identify the camera
    associated to a given serial number.
    """
    if serial_numbers is None or len(serial_numbers) == 0:
        camera_infos = find_cameras()
        serial_numbers = [cam["serial_number"] for cam in camera_infos]

    print("Connecting cameras")
    cameras = []
    for cam_sn in serial_numbers:
        print(f"{cam_sn=}")
        config = ZedCameraConfig(serial_number=cam_sn, fps=fps, width=width, height=height)
        camera = ZedCamera(config)
        camera.connect()
        print(
            f"ZedCamera({camera.serial_number}, fps={camera.fps}, width={camera.capture_width}, "
            f"height={camera.capture_height}, color_mode={camera.color_mode})"
        )
        cameras.append(camera)

    images_dir = Path(images_dir)
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving images to {images_dir}")
    frame_index = 0
    start_time = time.perf_counter()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            while True:
                now = time.perf_counter()

                for camera in cameras:
                    image = camera.read() if fps is None else camera.async_read()
                    if isinstance(image, tuple):
                        image = image[0]

                    executor.submit(
                        save_image,
                        image,
                        camera.serial_number,
                        frame_index,
                        images_dir,
                    )

                if fps is not None:
                    dt_s = time.perf_counter() - now
                    busy_wait(1 / fps - dt_s)

                if time.perf_counter() - start_time > record_time_s:
                    break

                print(f"Frame: {frame_index:04d}\tLatency (ms): {(time.perf_counter() - now) * 1000:.2f}")

                frame_index += 1
    finally:
        print(f"Images have been saved to {images_dir}")
        for camera in cameras:
            camera.disconnect()


class ZedCamera:
    """
    The ZedCamera class allows recording images (and optionally depth) from ZED stereo cameras
    (ZED, ZED Mini, ZED 2, ZED 2i, ZED X, ZED X Mini). It relies on the ZED SDK (`pyzed`).

    A ZedCamera instance requires a serial number (e.g. `ZedCamera(config)` with
    `config.serial_number=12345`). To find connected ZED cameras and their serial numbers, run:
    ```bash
    python lerobot/common/robot_devices/cameras/zed.py --images-dir outputs/images_from_zed_cameras
    ```

    When a ZedCamera is instantiated, if no specific config is provided, the default fps, width, height
    and color_mode of the given camera will be used.

    Example of usage:
    ```python
    from lerobot.common.robot_devices.cameras.configs import ZedCameraConfig

    config = ZedCameraConfig(serial_number=12345)
    camera = ZedCamera(config)
    camera.connect()
    color_image = camera.read()
    camera.disconnect()
    ```

    Example of returning depth:
    ```python
    config = ZedCameraConfig(serial_number=12345, use_depth=True)
    camera = ZedCamera(config)
    camera.connect()
    color_image, depth_map = camera.read()
    ```
    """

    def __init__(self, config: ZedCameraConfig):
        self.config = config
        self.serial_number = config.serial_number

        self.capture_width = config.width
        self.capture_height = config.height

        if config.rotation in [-90, 90]:
            self.width = config.height
            self.height = config.width
        else:
            self.width = config.width
            self.height = config.height

        self.fps = config.fps
        self.channels = config.channels
        self.color_mode = config.color_mode
        self.use_depth = config.use_depth
        self.depth_mode = config.depth_mode
        self.mock = config.mock

        self.camera = None
        self.is_connected = False
        self.thread = None
        self.stop_event = None
        self.color_image = None
        self.depth_map = None
        self.logs = {}

        self.rotation = None
        if config.rotation is not None:
            import cv2

            if config.rotation == -90:
                self.rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
            elif config.rotation == 90:
                self.rotation = cv2.ROTATE_90_CLOCKWISE
            elif config.rotation == 180:
                self.rotation = cv2.ROTATE_180

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"ZedCamera({self.serial_number}) is already connected."
            )

        import pyzed.sl as sl

        self.camera = sl.Camera()

        init_params = sl.InitParameters()
        init_params.camera_resolution = self._get_resolution(self.capture_width, self.capture_height)

        if self.fps is not None:
            init_params.camera_fps = self.fps

        if self.serial_number is not None:
            init_params.set_from_serial_number(self.serial_number)

        if self.use_depth:
            depth_mode_name = DEPTH_MODE_MAP.get(self.depth_mode, "NEURAL")
            init_params.depth_mode = getattr(sl.DEPTH_MODE, depth_mode_name)
        else:
            init_params.depth_mode = sl.DEPTH_MODE.NONE

        init_params.coordinate_units = sl.UNIT.MILLIMETER

        status = self.camera.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            camera_infos = find_cameras(raise_when_empty=False)
            serial_numbers = [cam["serial_number"] for cam in camera_infos]
            if self.serial_number is not None and self.serial_number not in serial_numbers:
                raise ValueError(
                    f"`serial_number` is expected to be one of these available cameras {serial_numbers}, "
                    f"but {self.serial_number} is provided instead. "
                    "To find the serial number you should use, run "
                    "`python lerobot/common/robot_devices/cameras/zed.py`."
                )
            raise OSError(
                f"Can't open ZedCamera({self.serial_number}): {status}. "
                "Make sure the ZED SDK is installed and the camera is properly connected."
            )

        runtime_params = sl.RuntimeParameters()
        self._runtime_params = runtime_params

        camera_info = self.camera.get_camera_information()
        actual_fps = camera_info.camera_configuration.fps
        actual_width = camera_info.camera_configuration.resolution.width
        actual_height = camera_info.camera_configuration.resolution.height

        if self.fps is not None and not math.isclose(self.fps, actual_fps, rel_tol=1e-3):
            raise OSError(
                f"Can't set {self.fps=} for ZedCamera({self.serial_number}). Actual value is {actual_fps}."
            )
        if self.capture_width is not None and self.capture_width != actual_width:
            raise OSError(
                f"Can't set {self.capture_width=} for ZedCamera({self.serial_number}). "
                f"Actual value is {actual_width}."
            )
        if self.capture_height is not None and self.capture_height != actual_height:
            raise OSError(
                f"Can't set {self.capture_height=} for ZedCamera({self.serial_number}). "
                f"Actual value is {actual_height}."
            )

        self.fps = round(actual_fps)
        self.capture_width = round(actual_width)
        self.capture_height = round(actual_height)

        if self.serial_number is None:
            self.serial_number = camera_info.serial_number

        self._sl_image = sl.Mat()
        if self.use_depth:
            self._sl_depth = sl.Mat()

        self.is_connected = True

    def _get_resolution(self, width, height):
        import pyzed.sl as sl

        if width is None or height is None:
            return sl.RESOLUTION.AUTO

        resolution_map = {
            (4416, 1242): sl.RESOLUTION.HD2K,
            (3840, 1080): sl.RESOLUTION.HD1080,
            (2560, 720): sl.RESOLUTION.HD720,
            (1344, 376): sl.RESOLUTION.VGA,
            (1920, 1200): sl.RESOLUTION.HD1200,
            (1920, 1080): sl.RESOLUTION.HD1080,
            (1280, 720): sl.RESOLUTION.HD720,
        }

        resolution = resolution_map.get((width, height))
        if resolution is not None:
            return resolution

        return sl.RESOLUTION.AUTO

    def read(self, temporary_color: str | None = None) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Read a frame from the camera returned in the format height x width x channels (e.g. 720 x 1280 x 3)
        of type `np.uint8`.

        When `use_depth=True`, returns a tuple `(color_image, depth_map)` with a depth map in the format
        height x width (e.g. 720 x 1280) of type np.float32 (depth in millimeters).

        Note: Reading a frame is done every `camera.fps` times per second, and it is blocking.
        If you are reading data from other sensors, we advise to use `camera.async_read()` which is the
        non-blocking version of `camera.read()`.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"ZedCamera({self.serial_number}) is not connected. Try running `camera.connect()` first."
            )

        import pyzed.sl as sl

        start_time = time.perf_counter()

        err = self.camera.grab(self._runtime_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise OSError(f"Can't grab frame from ZedCamera({self.serial_number}): {err}")

        self.camera.retrieve_image(self._sl_image, sl.VIEW.LEFT)
        # ZED SDK returns BGRA by default
        color_image = self._sl_image.get_data()[:, :, :3].copy()

        requested_color_mode = self.color_mode if temporary_color is None else temporary_color
        if requested_color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"Expected color values are 'rgb' or 'bgr', but {requested_color_mode} is provided."
            )

        if requested_color_mode == "rgb":
            import cv2

            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        h, w, _ = color_image.shape
        if h != self.capture_height or w != self.capture_width:
            raise OSError(
                f"Can't capture color image with expected height and width "
                f"({self.capture_height} x {self.capture_width}). ({h} x {w}) returned instead."
            )

        if self.rotation is not None:
            import cv2

            color_image = cv2.rotate(color_image, self.rotation)

        self.logs["delta_timestamp_s"] = time.perf_counter() - start_time
        self.logs["timestamp_utc"] = capture_timestamp_utc()

        if self.use_depth:
            self.camera.retrieve_measure(self._sl_depth, sl.MEASURE.DEPTH)
            depth_map = self._sl_depth.get_data().copy()

            h, w = depth_map.shape
            if h != self.capture_height or w != self.capture_width:
                raise OSError(
                    f"Can't capture depth map with expected height and width "
                    f"({self.capture_height} x {self.capture_width}). ({h} x {w}) returned instead."
                )

            if self.rotation is not None:
                import cv2

                depth_map = cv2.rotate(depth_map, self.rotation)

            self.color_image = color_image
            self.depth_map = depth_map
            return color_image, depth_map
        else:
            self.color_image = color_image
            return color_image

    def read_loop(self):
        while not self.stop_event.is_set():
            try:
                if self.use_depth:
                    self.color_image, self.depth_map = self.read()
                else:
                    self.color_image = self.read()
            except Exception as e:
                print(f"Error reading in thread: {e}")

    def async_read(self):
        """Access the latest color image (and depth if enabled)."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"ZedCamera({self.serial_number}) is not connected. Try running `camera.connect()` first."
            )

        if self.thread is None:
            self.stop_event = threading.Event()
            self.thread = Thread(target=self.read_loop, args=())
            self.thread.daemon = True
            self.thread.start()

        num_tries = 0
        while self.color_image is None:
            num_tries += 1
            time.sleep(1 / self.fps)
            if num_tries > self.fps and (self.thread.ident is None or not self.thread.is_alive()):
                raise Exception(
                    "The thread responsible for `self.async_read()` took too much time to start. "
                    "There might be an issue. Verify that `self.thread.start()` has been called."
                )

        if self.use_depth:
            return self.color_image, self.depth_map
        else:
            return self.color_image

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"ZedCamera({self.serial_number}) is not connected. Try running `camera.connect()` first."
            )

        if self.thread is not None and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join()
            self.thread = None
            self.stop_event = None

        self.camera.close()
        self.camera = None
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save a few frames using `ZedCamera` for all ZED cameras connected to the computer, "
        "or a selected subset."
    )
    parser.add_argument(
        "--serial-numbers",
        type=int,
        nargs="*",
        default=None,
        help="List of serial numbers used to instantiate the `ZedCamera`. "
        "If not provided, find and use all available cameras.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Set the number of frames recorded per second for all cameras. "
        "If not provided, use the default fps of each camera.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Set the width for all cameras. If not provided, use the default width of each camera.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Set the height for all cameras. If not provided, use the default height of each camera.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default="outputs/images_from_zed_cameras",
        help="Set directory to save a few frames for each camera.",
    )
    parser.add_argument(
        "--record-time-s",
        type=float,
        default=2.0,
        help="Set the number of seconds used to record the frames. By default, 2 seconds.",
    )
    args = parser.parse_args()
    save_images_from_cameras(**vars(args))
