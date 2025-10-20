"""
This file contains utilities for recording frames from Azure Kinect cameras.
"""

import argparse
import concurrent.futures
import logging
import math
import shutil
import threading
import time
import traceback
from pathlib import Path
from threading import Thread
from typing import Optional, Tuple, Union, Dict, Any

import numpy as np
import cv2
from PIL import Image

from lerobot.common.robot_devices.cameras.configs import AzureKinectCameraConfig
from lerobot.common.robot_devices.utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceNotConnectedError,
    busy_wait,
)
from lerobot.common.utils.utils import capture_timestamp_utc


def find_cameras(raise_when_empty=True, mock=False) -> list[dict]:
    """
    Find the names and the serial numbers of the Azure Kinect cameras
    connected to the computer.
    """
    if mock:
        # Mock implementation for testing
        return [{"serial_number": "000000000012", "name": "Azure Kinect 4K Camera"}]
    
    try:
        import pyk4a
        from pyk4a import Config, PyK4A
    except ImportError:
        raise ImportError(
            "pyk4a is required for Azure Kinect support. Install it with: pip install pyk4a"
        )

    cameras = []
    device_count = pyk4a.connected_device_count()
    
    for device_id in range(device_count):
        try:
            device = PyK4A(device_id=device_id)
            device.open()
            serial_number = device.serial
            device.close()
            cameras.append({
                "device_id": device_id,
                "serial_number": serial_number,
                "name": f"Azure Kinect 4K Camera",
            })
        except Exception as e:
            logging.warning(f"Failed to query device {device_id}: {e}")

    if raise_when_empty and len(cameras) == 0:
        raise OSError(
            "Not a single Azure Kinect camera was detected. Try re-plugging, or re-installing "
            "`pyk4a` and the Azure Kinect SDK, or updating the firmware."
        )

    return cameras


def save_image(img_array, device_id, frame_index, images_dir, image_type="color"):
    try:
        img = Image.fromarray(img_array)
        path = images_dir / f"camera_{device_id}_{image_type}_frame_{frame_index:06d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path), quality=100)
        logging.info(f"Saved image: {path}")
    except Exception as e:
        logging.error(f"Failed to save {image_type} image for camera {device_id} frame {frame_index}: {e}")


def save_images_from_cameras(
    images_dir: Path,
    device_ids: list[int] | None = None,
    fps=None,
    width=None,
    height=None,
    record_time_s=2,
    use_depth=True,
    use_ir=False,
    use_transformed_depth=False,
    use_point_cloud=False,
    use_transformed_color=False,
    wired_sync_modes: list[str] | None = None,
    subordinate_delay_off_master_usec: int = 200,
    mock=False,
):
    """
    Initializes all the cameras and saves images to the directory. Useful to visually identify the camera
    associated to a given device ID.

    For synchronized multi-camera setups:
    - wired_sync_modes must match the length and order of device_ids
    - The order should correspond to the real-world hardware configuration
    - Example: device_ids=[1, 0], wired_sync_modes=["subordinate", "master"] means device 1 is subordinate, device 0 is master
    """
    if device_ids is None or len(device_ids) == 0:
        device_ids = [0]

    if mock:
        import tests.cameras.mock_cv2 as cv2
    else:
        import cv2

    # Validate wired_sync_modes if provided
    if wired_sync_modes is not None and len(wired_sync_modes) != len(device_ids):
        raise ValueError(
            f"If wired_sync_modes is provided, it must match the length of device_ids. "
            f"Got {len(wired_sync_modes)} sync modes for {len(device_ids)} devices."
        )

    print("Initializing cameras")
    cameras = []

    # Create all camera objects
    for i, device_id in enumerate(device_ids):
        # Get the wired sync mode for this camera
        wired_sync_mode = None
        if wired_sync_modes is not None:
            sync_mode_str = wired_sync_modes[i]
            # Handle string "None" or actual None
            if sync_mode_str and sync_mode_str.lower() != "none":
                wired_sync_mode = sync_mode_str

        config = AzureKinectCameraConfig(
            device_id=device_id,
            fps=fps,
            width=width,
            height=height,
            use_depth=use_depth,
            use_ir=use_ir,
            use_transformed_depth=use_transformed_depth,
            use_point_cloud=use_point_cloud,
            use_transformed_color=use_transformed_color,
            wired_sync_mode=wired_sync_mode,
            subordinate_delay_off_master_usec=subordinate_delay_off_master_usec,
            mock=mock
        )
        camera = AzureKinectCamera(config)
        cameras.append(camera)

    for camera in cameras:
        camera.connect(start_cameras=False)

    # Subordinate cameras must start first, then master cameras
    subordinate_cameras = [cam for cam in cameras if cam.wired_sync_mode == "subordinate"]
    master_cameras = [cam for cam in cameras if cam.wired_sync_mode == "master"]
    standalone_cameras = [cam for cam in cameras if cam.wired_sync_mode is None]

    # Start subordinate cameras first
    for cam in subordinate_cameras:
        cam.start()
    # Then start master camera
    for cam in master_cameras:
        cam.start()

    # Or standalone camera
    for cam in standalone_cameras:
        cam.start()

    # Print info after all started
    for camera in cameras:
        sync_info = f", sync_mode={camera.wired_sync_mode}" if camera.wired_sync_mode else ""
        print(
            f"AzureKinectCamera({camera.device_id}, fps={camera.fps}, width={camera.capture_width}, "
            f"height={camera.capture_height}, color_mode={camera.color_mode}, use_depth={camera.use_depth}{sync_info})"
        )

    images_dir = Path(images_dir)
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving images to {images_dir}")
    frame_index = 0
    start_time = time.perf_counter()
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            while True:
                now = time.perf_counter()

                for camera in cameras:
                    # If we use async_read when fps is None, the loop will go full speed
                    if fps is None:
                        result = camera.read()
                    else:
                        result = camera.async_read()
                    
                    if result is not None:
                        # Handle both dict and single array returns
                        if isinstance(result, dict):
                            for data_type, data in result.items():
                                if data is not None:
                                    if data_type == "color":
                                        executor.submit(save_image, data, camera.device_id, frame_index, images_dir, "color")
                                    elif data_type == "depth":
                                        # Convert depth to 8-bit for visualization
                                        depth_8bit = (data / 65535.0 * 255).astype(np.uint8)
                                        executor.submit(save_image, depth_8bit, camera.device_id, frame_index, images_dir, "depth")
                                    elif data_type == "ir":
                                        # Convert IR to 8-bit for visualization
                                        ir_8bit = (data / 65535.0 * 255).astype(np.uint8)
                                        executor.submit(save_image, ir_8bit, camera.device_id, frame_index, images_dir, "ir")
                                    elif data_type == "transformed_depth":
                                        # Convert transformed depth to 8-bit for visualization
                                        trans_depth_8bit = (data / 65535.0 * 255).astype(np.uint8)
                                        executor.submit(save_image, trans_depth_8bit, camera.device_id, frame_index, images_dir, "transformed_depth")
                                    elif data_type == "point_cloud":
                                        # Save point cloud as a .npy file instead of image
                                        pc_path = images_dir / f"camera_{camera.device_id}_point_cloud_frame_{frame_index:06d}.npy"
                                        pc_path.parent.mkdir(parents=True, exist_ok=True)
                                        np.save(str(pc_path), data)
                                    elif data_type == "transformed_color":
                                        executor.submit(save_image, data, camera.device_id, frame_index, images_dir, "transformed_color")
                        else:
                            # Single color image
                            executor.submit(save_image, result, camera.device_id, frame_index, images_dir, "color")

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


class AzureKinectCamera:
    """
    The AzureKinectCamera class provides an interface for Azure Kinect cameras with features:
    - Uses device ID for camera identification
    - Supports color, depth, IR, transformed depth (RGB-aligned), point clouds, and transformed color
    - Supports wired synchronized multi-camera setups (master/subordinate mode)
    - Compatible with the LeRobot camera interface pattern

    To find available cameras, run:
    ```bash
    python lerobot/common/robot_devices/cameras/azurekinect.py --images-dir outputs/images_from_kinect_cameras
    ```

    Basic usage (color only):
    ```python
    from lerobot.common.robot_devices.cameras.configs import AzureKinectCameraConfig

    config = AzureKinectCameraConfig(device_id=0)
    camera = AzureKinectCamera(config)
    camera.connect()
    color_image = camera.read()  # Returns (H, W, 3) numpy array
    camera.disconnect()
    ```

    Advanced usage (multiple data types):
    ```python
    config = AzureKinectCameraConfig(
        device_id=0,
        use_depth=True,
        use_transformed_depth=True,
        use_point_cloud=True
    )
    camera = AzureKinectCamera(config)
    camera.connect()

    result = camera.read()  # Returns dict with keys: 'color', 'depth', 'transformed_depth', 'point_cloud'
    color_image = result['color']              # (H, W, 3) uint8
    depth_map = result['depth']                # (H, W) uint16 in mm
    aligned_depth = result['transformed_depth'] # (H, W) uint16 aligned to color camera
    point_cloud = result['point_cloud']        # (H, W, 3) float32 in meters (X, Y, Z)

    camera.disconnect()
    ```

    Synchronized multi-camera usage:
    ```python
    # Configure master and subordinate cameras
    # Note: The device IDs and sync modes must match your real-world hardware setup
    master_config = AzureKinectCameraConfig(
        device_id=0,  # This device is physically configured as master
        fps=15,
        width=1280,
        height=720,
        wired_sync_mode="master"
    )
    subordinate_config = AzureKinectCameraConfig(
        device_id=1,  # This device is physically configured as subordinate
        fps=15,
        width=1280,
        height=720,
        wired_sync_mode="subordinate",
        subordinate_delay_off_master_usec=200
    )

    master_cam = AzureKinectCamera(master_config)
    subordinate_cam = AzureKinectCamera(subordinate_config)

    # Connect cameras (but don't start yet)
    subordinate_cam.connect(start_cameras=False)
    master_cam.connect(start_cameras=False)

    # IMPORTANT: Start subordinate first, then master (required for proper sync!)
    subordinate_cam.start()
    master_cam.start()

    # Now both cameras are synchronized
    master_img = master_cam.read()
    subordinate_img = subordinate_cam.read()
    ```
    """

    def __init__(self, config: AzureKinectCameraConfig):
        self.config = config
        self.device_id = config.device_id

        # Store the raw (capture) resolution from the config
        self.capture_width = config.width
        self.capture_height = config.height

        # If rotated by ±90, swap width and height
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
        self.use_ir = config.use_ir
        self.use_transformed_depth = config.use_transformed_depth
        self.use_point_cloud = config.use_point_cloud
        self.use_transformed_color = config.use_transformed_color
        self.wired_sync_mode = config.wired_sync_mode
        self.subordinate_delay_off_master_usec = config.subordinate_delay_off_master_usec
        self.mock = config.mock

        self.camera = None
        self.capture = None
        self.is_connected = False
        self.thread = None
        self.stop_event = None
        self.color_image = None
        self.depth_map = None
        self.ir_image = None
        self.transformed_depth = None
        self.point_cloud = None
        self.transformed_color = None
        self.logs = {}

        if self.mock:
            import tests.cameras.mock_cv2 as cv2
        else:
            import cv2

        self.rotation = None
        if config.rotation == -90:
            self.rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
        elif config.rotation == 90:
            self.rotation = cv2.ROTATE_90_CLOCKWISE
        elif config.rotation == 180:
            self.rotation = cv2.ROTATE_180

    def connect(self, start_cameras=True):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"AzureKinectCamera({self.device_id}) is already connected."
            )

        if self.mock:
            # Mock connection for testing
            self.fps = self.fps or 30
            self.capture_width = self.capture_width or 1280
            self.capture_height = self.capture_height or 720
            self.is_connected = True
            return

        try:
            import pyk4a
            from pyk4a import Config, PyK4A, ColorResolution, DepthMode, FPS, WiredSyncMode
        except ImportError:
            raise ImportError(
                "pyk4a is required for Azure Kinect support. Install it with: pip install pyk4a"
            )

        # Map config values to pyk4a enums
        fps_map = {5: FPS.FPS_5, 15: FPS.FPS_15, 30: FPS.FPS_30}
        color_resolution_map = {
            (1280, 720): ColorResolution.RES_720P,
            (1920, 1080): ColorResolution.RES_1080P,
            (2560, 1440): ColorResolution.RES_1440P,
            (2048, 1536): ColorResolution.RES_1536P,
            (3840, 2160): ColorResolution.RES_3072P,
        }

        # Set defaults if not specified
        if self.fps is None:
            self.fps = 30
        if self.capture_width is None or self.capture_height is None:
            self.capture_width, self.capture_height = 1280, 720

        # Get pyk4a enums
        k4a_fps = fps_map.get(self.fps, FPS.FPS_30)
        k4a_color_res = color_resolution_map.get(
            (self.capture_width, self.capture_height), ColorResolution.RES_720P
        )
        
        # Enable depth if any depth-related feature is requested
        needs_depth = self.use_depth or self.use_transformed_depth or self.use_point_cloud or self.use_transformed_color
        k4a_depth_mode = DepthMode.NFOV_UNBINNED if needs_depth else DepthMode.OFF
        
        # Use BGRA32 format if we need transformed color (required by pyk4a)
        if self.use_transformed_color:
            from pyk4a import ImageFormat
            color_format = ImageFormat.COLOR_BGRA32
        else:
            color_format = None

        # Map wired sync mode string to enum
        k4a_wired_sync = None
        if self.wired_sync_mode == "master":
            k4a_wired_sync = WiredSyncMode.MASTER
        elif self.wired_sync_mode == "subordinate":
            k4a_wired_sync = WiredSyncMode.SUBORDINATE

        # Create configuration dictionary
        config_params = {
            "color_resolution": k4a_color_res,
            "depth_mode": k4a_depth_mode,
            "camera_fps": k4a_fps,
            "synchronized_images_only": needs_depth,
        }

        if color_format:
            config_params["color_format"] = color_format

        if k4a_wired_sync is not None:
            config_params["wired_sync_mode"] = k4a_wired_sync
            if self.wired_sync_mode == "subordinate":
                config_params["subordinate_delay_off_master_usec"] = self.subordinate_delay_off_master_usec

        k4a_config = Config(**config_params)

        try:
            self.camera = PyK4A(config=k4a_config, device_id=self.device_id, thread_safe=False)
            self.camera.open()
            if start_cameras:
                self.camera.start()
            is_camera_open = True
        except Exception:
            is_camera_open = False
            traceback.print_exc()

        if not is_camera_open:
            # Verify that the provided device_id is valid
            camera_infos = find_cameras()
            available_device_ids = [cam["device_id"] for cam in camera_infos]
            if self.device_id not in available_device_ids:
                raise ValueError(
                    f"`device_id` is expected to be one of these available cameras {available_device_ids}, "
                    f"but {self.device_id} is provided instead. "
                    "To find the device ID you should use, run `python lerobot/common/robot_devices/cameras/azurekinect.py`."
                )
            raise OSError(f"Can't access AzureKinectCamera({self.device_id}).")

        # Update actual resolution based on selected mode
        resolution_map = {
            ColorResolution.RES_720P: (1280, 720),
            ColorResolution.RES_1080P: (1920, 1080),
            ColorResolution.RES_1440P: (2560, 1440),
            ColorResolution.RES_1536P: (2048, 1536),
            ColorResolution.RES_3072P: (3840, 2160),
        }
        actual_width, actual_height = resolution_map[k4a_color_res]

        if (self.capture_width, self.capture_height) != (actual_width, actual_height):
            logging.warning(
                f"Requested resolution {self.capture_width}x{self.capture_height} "
                f"mapped to {actual_width}x{actual_height}"
            )

        self.capture_width = actual_width
        self.capture_height = actual_height
        self.is_connected = start_cameras

    def start(self):
        """Start the camera streaming. Only needed if connect() was called with start_cameras=False."""
        if self.mock:
            self.is_connected = True
            return

        if self.camera is None:
            raise RobotDeviceNotConnectedError(
                f"AzureKinectCamera({self.device_id}) camera not opened. Call connect() first."
            )

        self.camera.start()
        self.is_connected = True

    def read(self, temporary_color_mode: str | None = None) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Read a frame from the camera. Returns different data based on configuration:
        
        - If only color is requested: returns np.ndarray (H, W, 3)
        - If multiple data types requested: returns Dict[str, np.ndarray] with keys:
          - 'color': (H, W, 3) uint8 array
          - 'depth': (H, W) uint16 array in millimeters  
          - 'ir': (H, W) uint16 array
          - 'transformed_depth': (H, W) uint16 array aligned to color camera
          - 'point_cloud': (H, W, 3) float32 array in meters (X, Y, Z)
          - 'transformed_color': (H, W, 3) uint8 array aligned to depth camera
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"AzureKinectCamera({self.device_id}) is not connected. Try running `camera.connect()` first."
            )

        if self.mock:
            # Mock implementation
            result = {}
            if True:  # Always include color for mock
                result['color'] = np.random.randint(0, 255, (self.capture_height, self.capture_width, 3), dtype=np.uint8)
            if self.use_depth:
                result['depth'] = np.random.randint(0, 5000, (self.capture_height, self.capture_width), dtype=np.uint16)
            if self.use_ir:
                result['ir'] = np.random.randint(0, 1000, (self.capture_height, self.capture_width), dtype=np.uint16)
            if self.use_transformed_depth:
                result['transformed_depth'] = np.random.randint(0, 5000, (self.capture_height, self.capture_width), dtype=np.uint16)
            if self.use_point_cloud:
                result['point_cloud'] = np.random.randn(self.capture_height, self.capture_width, 3).astype(np.float32)
            if self.use_transformed_color:
                result['transformed_color'] = np.random.randint(0, 255, (self.capture_height, self.capture_width, 3), dtype=np.uint8)
            
            # Return single array if only color requested, otherwise return dict
            if len(result) == 1 and 'color' in result:
                return result['color']
            return result

        start_time = time.perf_counter()

        try:
            self.capture = self.camera.get_capture()
        except Exception as e:
            raise OSError(f"Can't capture frame from AzureKinectCamera({self.device_id}): {e}")

        result = {}
        
        # Always get color image
        if self.capture.color is None:
            raise OSError(f"Can't capture color image from AzureKinectCamera({self.device_id}).")

        color_image = self.capture.color
        
        # Handle BGRA vs BGR format
        if color_image.shape[2] == 4:  # BGRA
            color_image = color_image[:, :, :3]  # Remove alpha channel
        
        requested_color_mode = self.color_mode if temporary_color_mode is None else temporary_color_mode
        if requested_color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"Expected color values are 'rgb' or 'bgr', but {requested_color_mode} is provided."
            )

        # Convert color format as needed
        if requested_color_mode == "rgb":
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        h, w, _ = color_image.shape
        if h != self.capture_height or w != self.capture_width:
            raise OSError(
                f"Can't capture color image with expected height and width ({self.capture_height} x {self.capture_width}). "
                f"({h} x {w}) returned instead."
            )

        if self.rotation is not None:
            color_image = cv2.rotate(color_image, self.rotation)

        result['color'] = color_image
        self.color_image = color_image

        # Get additional data types as requested
        if self.use_depth:
            if self.capture.depth is None:
                raise OSError(f"Can't capture depth image from AzureKinectCamera({self.device_id}).")
            depth_map = self.capture.depth
            if self.rotation is not None:
                depth_map = cv2.rotate(depth_map, self.rotation)
            result['depth'] = depth_map
            self.depth_map = depth_map

        if self.use_ir:
            if self.capture.ir is None:
                raise OSError(f"Can't capture IR image from AzureKinectCamera({self.device_id}).")
            ir_image = self.capture.ir
            if self.rotation is not None:
                ir_image = cv2.rotate(ir_image, self.rotation)
            result['ir'] = ir_image
            self.ir_image = ir_image

        if self.use_transformed_depth:
            transformed_depth = self.capture.transformed_depth
            if transformed_depth is None:
                raise OSError(f"Can't capture transformed depth from AzureKinectCamera({self.device_id}).")
            if self.rotation is not None:
                transformed_depth = cv2.rotate(transformed_depth, self.rotation)
            result['transformed_depth'] = transformed_depth
            self.transformed_depth = transformed_depth

        if self.use_point_cloud:
            point_cloud = self.capture.depth_point_cloud
            if point_cloud is None:
                raise OSError(f"Can't generate point cloud from AzureKinectCamera({self.device_id}).")
            # Point cloud is (H, W, 3) with X, Y, Z in meters
            if self.rotation is not None:
                # For point clouds, we need to rotate each of the 3 channels
                for i in range(3):
                    point_cloud[:, :, i] = cv2.rotate(point_cloud[:, :, i], self.rotation)
            result['point_cloud'] = point_cloud
            self.point_cloud = point_cloud

        if self.use_transformed_color:
            transformed_color = self.capture.transformed_color
            if transformed_color is None:
                raise OSError(f"Can't capture transformed color from AzureKinectCamera({self.device_id}).")
            # Convert from BGRA to requested format
            if transformed_color.shape[2] == 4:
                transformed_color = transformed_color[:, :, :3]
            if requested_color_mode == "rgb":
                transformed_color = cv2.cvtColor(transformed_color, cv2.COLOR_BGR2RGB)
            if self.rotation is not None:
                transformed_color = cv2.rotate(transformed_color, self.rotation)
            result['transformed_color'] = transformed_color
            self.transformed_color = transformed_color

        # Log timing information
        self.logs["delta_timestamp_s"] = time.perf_counter() - start_time
        self.logs["timestamp_utc"] = capture_timestamp_utc()

        # Return single array if only color requested, otherwise return dict
        if len(result) == 1 and 'color' in result:
            return result['color']
        return result

    def read_loop(self):
        while not self.stop_event.is_set():
            try:
                result = self.read()
                # Store the result in individual properties for async access
                if isinstance(result, dict):
                    self.color_image = result.get('color')
                    self.depth_map = result.get('depth')
                    self.ir_image = result.get('ir')
                    self.transformed_depth = result.get('transformed_depth')
                    self.point_cloud = result.get('point_cloud')
                    self.transformed_color = result.get('transformed_color')
                else:
                    self.color_image = result
            except Exception as e:
                logging.error(f"Error reading in thread: {e}")

    def async_read(self):
        """Access the latest captured data"""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"AzureKinectCamera({self.device_id}) is not connected. Try running `camera.connect()` first."
            )

        if self.thread is None:
            self.stop_event = threading.Event()
            self.thread = Thread(target=self.read_loop, args=())
            self.thread.daemon = True
            self.thread.start()

        num_tries = 0
        max_tries = self.fps * 2
        while num_tries < max_tries:
            ready = True
            result = {}

            if self.color_image is None: ready = False
            else: result['color'] = self.color_image

            if self.use_depth:
                if self.depth_map is None: ready = False
                else: result['depth'] = self.depth_map[..., None]
                
            if self.use_ir:
                if self.ir_image is None: ready = False
                else: result['ir'] = self.ir_image[..., None]
            
            if self.use_transformed_depth: 
                if self.transformed_depth is None: ready = False
                else: result['transformed_depth'] = self.transformed_depth[..., None]
            
            if self.use_point_cloud: 
                if self.point_cloud is None: ready = False
                else: result['point_cloud'] = self.point_cloud
            
            if self.use_transformed_color: 
                if self.transformed_color is None: ready = False
                else: result['transformed_color'] = self.transformed_color
                
            if ready: return result

            time.sleep(1 / self.fps)
            num_tries += 1
        raise TimeoutError("Timed out waiting for async_read() to start.")

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"AzureKinectCamera({self.device_id}) is not connected. Try running `camera.connect()` first."
            )

        if self.thread is not None and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join()
            self.thread = None
            self.stop_event = None

        if not self.mock and self.camera is not None:
            self.camera.stop()
            self.camera = None

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save a few frames using `AzureKinectCamera` for all cameras connected to the computer, or a selected subset."
    )
    parser.add_argument(
        "--device-ids",
        type=int,
        nargs="*",
        default=None,
        help="List of device IDs used to instantiate the `AzureKinectCamera`. If not provided, find and use all available cameras.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Set the number of frames recorded per seconds for all cameras.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Set the width for all cameras.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Set the height for all cameras.",
    )
    parser.add_argument(
        "--use-depth",
        action="store_true",
        help="Enable depth capture.",
    )
    parser.add_argument(
        "--use-ir",
        action="store_true", 
        help="Enable IR capture.",
    )
    parser.add_argument(
        "--use-transformed-depth",
        action="store_true",
        help="Enable transformed depth (aligned to color camera).",
    )
    parser.add_argument(
        "--use-point-cloud",
        action="store_true",
        help="Enable point cloud generation.",
    )
    parser.add_argument(
        "--use-transformed-color",
        action="store_true",
        help="Enable transformed color (aligned to depth camera).",
    )
    parser.add_argument(
        "--wired-sync-modes",
        type=str,
        nargs="*",
        default=None,
        help="Wired sync mode for each camera (None, 'master', or 'subordinate'). "
             "Must match length and order of --device-ids. The order should match the real-world hardware setup. "
             "Example: --device-ids 1 0 --wired-sync-modes subordinate master (device 1 is subordinate, device 0 is master)",
    )
    parser.add_argument(
        "--subordinate-delay-off-master-usec",
        type=int,
        default=200,
        help="Delay in microseconds for subordinate cameras (default: 200).",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default="outputs/images_from_kinect_cameras",
        help="Set directory to save a few frames for each camera.",
    )
    parser.add_argument(
        "--record-time-s",
        type=float,
        default=2.0,
        help="Set the number of seconds used to record the frames.",
    )
    args = parser.parse_args()
    save_images_from_cameras(**vars(args))
