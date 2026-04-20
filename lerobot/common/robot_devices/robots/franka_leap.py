"""FrankaLeapRobot: Franka Panda + LEAP hand (via deoxys) with GELLO arm + Manus glove teleop."""

import glob
import time

import numpy as np
import torch

from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.robots.configs import FrankaLeapRobotConfig
from typing import List, Dict
import json
import PIL
import torchvision.transforms as transforms
from pytorch3d.ops import sample_farthest_points
from lerobot.common.utils.franka_leap_utils import URDFHandPointCloud

TARGET_SHAPE = 224
depth_preprocess = transforms.Compose(
    [
        transforms.Resize(
            TARGET_SHAPE,
            interpolation=transforms.InterpolationMode.NEAREST,
        ),
        transforms.CenterCrop(TARGET_SHAPE),
    ]
)


def load_calibrations(calibration_config_path: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load camera calibrations from JSON config.

    JSON format:
    {
        "cam_azure_kinect": {
            "intrinsics": "path/to/intrinsics.txt",
            "extrinsics": "path/to/extrinsics.txt"
        }
    }

    Returns: Dict mapping camera names to {"K": intrinsics, "T_world_cam": extrinsics}
    """
    with open(calibration_config_path, 'r') as f:
        config = json.load(f)

    calibrations = {}
    root_dir = "/home/leap/Desktop/lerobot_restore/lerobot/lerobot/scripts/"
    for cam_name, cam_config in config.items():
        calibrations[cam_name] = {
            "K": np.loadtxt(root_dir + cam_config["intrinsics"]),
            "T_world_cam": np.loadtxt(root_dir + cam_config["extrinsics"])
        }

    print(f"Loaded calibrations for {len(calibrations)} camera(s): {list(calibrations.keys())}")
    return calibrations


def get_scaled_intrinsics(K, orig_shape, target_shape):
    """
    Scale camera intrinsics based on image resizing and cropping.

    Args:
        K (np.ndarray): Original 3x3 camera intrinsic matrix.
        orig_shape (tuple): Original image shape (height, width).
        target_shape (int): Target size for resize and crop.

    Returns:
        np.ndarray: Scaled 3x3 intrinsic matrix.
    """
    # Getting scale factor from torchvision.transforms.Resize behaviour
    K_ = K.copy()

    scale_factor = target_shape / min(orig_shape)

    # Apply the scale factor to the intrinsics
    K_[0, 0] *= scale_factor  # fx
    K_[1, 1] *= scale_factor  # fy
    K_[0, 2] *= scale_factor  # cx
    K_[1, 2] *= scale_factor  # cy

    # Adjust the principal point (cx, cy) for the center crop
    crop_offset_x = (orig_shape[1] * scale_factor - target_shape) / 2
    crop_offset_y = (orig_shape[0] * scale_factor - target_shape) / 2

    # Adjust the principal point (cx, cy) for the center crop
    K_[0, 2] -= crop_offset_x  # Adjust cx for crop
    K_[1, 2] -= crop_offset_y  # Adjust cy for crop
    return K_

def compute_pcd(depth, K, num_points, max_depth, cam_to_world):
    """
    Compute a downsampled point cloud from RGB and depth images.

    Args:
        rgb (np.ndarray): RGB image array (H, W, 3). np.uint8
        depth (np.ndarray): Depth image array (H, W). np.uint16
        K (np.ndarray): 3x3 camera intrinsic matrix.
        rgb_preprocess (transforms.Compose): Preprocessing for RGB.
        depth_preprocess (transforms.Compose): Preprocessing for depth.
        device (torch.device): Device for computations.
        rng (np.random.Generator): Random number generator.
        num_points (int): Number of points to sample.
        max_depth (float): Maximum depth threshold.

    Returns:
        np.ndarray: Downsampled point cloud (N, 6) with XYZ and RGB.
    """
    depth_ = (depth.numpy() / 1000.0).squeeze().astype(np.float32)
    depth_ = PIL.Image.fromarray(depth_)
    depth_ = np.asarray(depth_preprocess(depth_))

    height, width = depth_.shape
    # Create pixel coordinate grid
    x = np.arange(width)
    y = np.arange(height)
    x_grid, y_grid = np.meshgrid(x, y)

    # Flatten grid coordinates and depth
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    z_flat = depth_.flatten()

    # Remove points with invalid depth
    valid_depth = np.logical_and(z_flat > 0, z_flat < max_depth)
    x_flat = x_flat[valid_depth]
    y_flat = y_flat[valid_depth]
    z_flat = z_flat[valid_depth]

    # Create homogeneous pixel coordinates
    pixels = np.stack([x_flat, y_flat, np.ones_like(x_flat)], axis=0)

    # Unproject points using K inverse
    K_inv = np.linalg.inv(K)
    points = K_inv @ pixels
    points = points * z_flat
    points = points.T  # Shape: (N, 3)

    pcd_xyz_hom = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)  # (N, 4)
    pcd_xyz_world = (cam_to_world @ pcd_xyz_hom.T).T[:, :3]  # (N, 3)
    # clip to bounding box
    pcd_xyz_world = pcd_xyz_world[pcd_xyz_world[:, 2] >= 0.]
    pcd_xyz_world = pcd_xyz_world[np.logical_and(pcd_xyz_world[:, 0] >= 0.22, pcd_xyz_world[:, 0] <= 1.)]
    pcd_xyz_world = pcd_xyz_world[np.logical_and(pcd_xyz_world[:, 1] >= -0.9, pcd_xyz_world[:, 1] <= 0.1)]

    scene_pcd_pt3d = torch.from_numpy(pcd_xyz_world)
    scene_pcd_downsample, scene_points_idx = sample_farthest_points(
        scene_pcd_pt3d[None], K=num_points, random_start_point=False
    )
    pcd_xyz_world = scene_pcd_downsample.squeeze().numpy()

    # Get corresponding colors at the indices
    return pcd_xyz_world


class FrankaLeapRobot:
    """Robot class for a Franka Panda with a LEAP dexterous hand.

    The follower arm is a Franka Panda controlled through the deoxys C++ controller
    interface over ZMQ. The leader arm is a GELLO (7-DOF Dynamixel). The LEAP hand
    (16-DOF) is teleoperated via a Manus glove through a GeoRT retargeting model.
    """

    robot_type = "franka_leap"

    def __init__(self, config: FrankaLeapRobotConfig | None = None, **kwargs):
        if config is None:
            self.config = FrankaLeapRobotConfig(**kwargs)
        else:
            self.config = config

        self.cameras = make_cameras_from_configs(self.config.cameras)
        self.robot_interface = None
        self.gello = None
        self.leap_hand = None
        self.manus_mocap = None
        self.geort_model = None
        self.is_connected = False
        self.logs = {}
        self.robot_type = self.config.type
        self.use_eef = self.config.use_eef

        # 7 arm joints + 16 hand joints
        self.arm_joint_names = [
            "joint_1", "joint_2", "joint_3", "joint_4",
            "joint_5", "joint_6", "joint_7",
        ]
        self.hand_joint_names = [
            "index_0", "index_1", "index_2", "index_3",
            "middle_0", "middle_1", "middle_2", "middle_3",
            "ring_0", "ring_1", "ring_2", "ring_3",
            "thumb_0", "thumb_1", "thumb_2", "thumb_3",
        ]
        self.joint_names = self.arm_joint_names + self.hand_joint_names

        # load camera calibrations
        calibrations = load_calibrations("/home/leap/Desktop/lerobot_restore/lerobot/lerobot/scripts/franka_leap_calibration/calibration_franka_leap.json")
        camera_names = list(calibrations.keys())

        for cam_name in camera_names:
            K = calibrations[cam_name]["K"]
            scaled_K = get_scaled_intrinsics(K, (720, 1280), TARGET_SHAPE)
            calibrations[cam_name]["scaled_K"] = scaled_K

        self.calibrations = calibrations
        self.hand_model = URDFHandPointCloud(total_points=500)

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            cam_ft.update(cam.config.get_feature_specs(cam_key))
        return cam_ft

    @property
    def motor_features(self) -> dict:
        motor_features = {
            "action": {
                "dtype": "float32",
                "shape": (len(self.joint_names),),
                "names": list(self.joint_names),
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(self.joint_names),),
                "names": list(self.joint_names),
            },
        }

        if self.use_eef:
            eef_names = [
                "rot_6d_0", "rot_6d_1", "rot_6d_2",
                "rot_6d_3", "rot_6d_4", "rot_6d_5",
                "trans_0", "trans_1", "trans_2",
            ] + list(self.hand_joint_names)
            motor_features["observation.right_eef_pose"] = {
                "dtype": "float32",
                "shape": (len(eef_names),),
                "names": eef_names,
            }
            motor_features["action.right_eef_pose"] = {
                "dtype": "float32",
                "shape": (len(eef_names),),
                "names": eef_names,
            }

        return motor_features

    @property
    def features(self):
        return {**self.motor_features, **self.camera_features}

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    def connect(self):
        if self.is_connected:
            raise RuntimeError("FrankaLeapRobot is already connected.")

        # Lazy import deoxys
        from deoxys.franka_interface import FrankaInterface
        from deoxys.utils import YamlConfig

        # Initialize Franka interface
        self.robot_interface = FrankaInterface(
            self.config.deoxys_general_cfg_file,
            use_visualizer=False,
        )
        self.controller_cfg = YamlConfig(
            self.config.deoxys_controller_cfg_file
        ).as_easydict()

        # Wait for state buffer to populate
        print("Waiting for Franka state buffer...")
        timeout = 30.0
        start_t = time.time()
        while len(self.robot_interface._state_buffer) == 0:
            time.sleep(0.1)
            if time.time() - start_t > timeout:
                raise TimeoutError(
                    "Timed out waiting for Franka state buffer. "
                    "Check that the deoxys controller is running."
                )
        print("Franka state buffer ready.")

        # Initialize GELLO leader arm
        self._init_gello()

        # Initialize LEAP hand
        self._init_leap_hand()

        # Initialize Manus glove + GeoRT model for hand teleop
        self._init_manus_glove()

        # Connect cameras
        from threading import Thread

        azure_kinect_cameras = []
        for name, camera in self.cameras.items():
            if camera.__class__.__name__ == "AzureKinectCamera":
                camera.connect(start_cameras=False)
                azure_kinect_cameras.append(camera)
            else:
                camera.connect()

        if len(azure_kinect_cameras) > 0:
            # For wired sync mode, subordinate cameras must start before master
            subordinates = [cam for cam in azure_kinect_cameras if cam.wired_sync_mode == "subordinate"]
            masters = [cam for cam in azure_kinect_cameras if cam.wired_sync_mode == "master"]
            standalone = [cam for cam in azure_kinect_cameras if cam.wired_sync_mode is None]

            for cam in subordinates:
                cam.start()
            for cam in masters:
                cam.start()
            for cam in standalone:
                cam.start()

        self.is_connected = True

        # Run interactive home calibration
        self.run_calibration()

    def _init_gello(self):
        """Initialize the GELLO leader arm."""
        from gello.robots.dynamixel import DynamixelRobot

        port = self.config.gello_port
        if port is None:
            usb_ports = glob.glob("/dev/serial/by-id/*")
            if len(usb_ports) == 0:
                raise ValueError(
                    "No GELLO port found. Please specify gello_port or plug in GELLO."
                )
            port = usb_ports[0]
            print(f"Using GELLO port: {port}")

        self._gello_port = port

        self.gello = DynamixelRobot(
            joint_ids=list(self.config.gello_joint_ids),
            joint_offsets=list(self.config.gello_joint_offsets),
            real=True,
            joint_signs=list(self.config.gello_joint_signs),
            port=port,
            # gripper_config=(
            #     self.config.gello_gripper_joint_id,
            #     self.config.gello_gripper_open_degrees,
            #     self.config.gello_gripper_close_degrees,
            # ),
        )

    def _init_leap_hand(self):
        """Initialize the LEAP dexterous hand via LeapNode."""
        from lerobot.common.robot_devices.robots.leap_hand import LeapNode

        print("Connecting to LEAP hand...")
        self.leap_hand = LeapNode()
        print("LEAP hand connected.")

        # Move to open position
        open_hand = np.zeros(16)
        self._move_hand_to(open_hand)

    def _init_manus_glove(self):
        """Initialize Manus glove and GeoRT retargeting model for hand teleop."""
        from lerobot.common.robot_devices.robots.geort import load_model
        from lerobot.common.robot_devices.robots.geort.mocap import ManusMocap

        print(f"Loading GeoRT model (ckpt_tag={self.config.geort_ckpt_tag})...")
        self.geort_model = load_model(
            tag=self.config.geort_ckpt_tag,
            checkpoint_root=self.config.geort_checkpoint_root,
        )
        print("GeoRT model loaded.")

        print("Connecting to Manus glove (ZMQ)...")
        self.manus_mocap = ManusMocap()
        print("Manus glove connected.")

    def _move_hand_to(self, target_allegro: np.ndarray, steps: int = 50, dt: float = 0.02):
        """Linearly interpolate LEAP hand from current position to target (allegro convention)."""
        current_allegro = self.leap_hand.read_pos() - np.pi
        for i in range(1, steps + 1):
            alpha = i / steps
            self.leap_hand.set_allegro((1 - alpha) * current_allegro + alpha * target_allegro)
            time.sleep(dt)

    def _get_hand_state(self) -> np.ndarray:
        """Read current 16-DOF hand joint positions (allegro convention)."""
        return self.leap_hand.read_pos() - np.pi

    def _smooth_move_to(self, target_joints: np.ndarray, step_rad: float = 0.01):
        """Smoothly interpolate Franka from its current pose to target_joints."""
        franka_current = self._get_franka_joints()
        max_delta = np.max(np.abs(target_joints - franka_current))
        num_steps = max(int(max_delta / step_rad), 1)

        # Use a neutral gripper action for the Franka controller
        gripper_action = 1.0

        for i in range(num_steps):
            alpha = (i + 1) / num_steps
            waypoint = franka_current + alpha * (target_joints - franka_current)
            deoxys_action = list(waypoint) + [gripper_action]
            self.robot_interface.control(
                controller_type=self.config.deoxys_controller_type,
                action=deoxys_action,
                controller_cfg=self.controller_cfg,
            )
        return max_delta, num_steps

    def run_calibration(self):
        """Move Franka to match the GELLO's current pose for absolute control."""
        input(
            "\n[FrankaLeapRobot] Position the GELLO at your desired starting pose.\n"
            "Press Enter when ready — the Franka will move to match..."
        )

        gello_joints = np.array(self.gello.get_joint_state())
        gello_target = gello_joints[:7]
        franka_current = self._get_franka_joints()

        print(f"  Franka current: {np.round(franka_current, 4)}")
        print(f"  GELLO target:   {np.round(gello_target, 4)}")
        print("  Moving Franka to match GELLO...")

        self._smooth_move_to(gello_target)
        print("[FrankaLeapRobot] Franka aligned to GELLO. Ready for teleop.")

    def _get_franka_joints(self) -> np.ndarray:
        """Read current 7-DOF joint positions from Franka."""
        return np.array(self.robot_interface._state_buffer[-1].q)

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise RuntimeError("FrankaLeapRobot is not connected. Run `robot.connect()` first.")

        # --- Arm: read GELLO and command Franka ---
        before_lread_t = time.perf_counter()
        gello_joints = np.array(self.gello.get_joint_state())
        self.logs["read_leader_dt_s"] = time.perf_counter() - before_lread_t

        robot_target = gello_joints[:7]

        # If GELLO has drifted far from Franka, smoothly interpolate
        franka_current = self._get_franka_joints()
        max_delta = np.max(np.abs(robot_target - franka_current))
        if max_delta > self.config.max_safe_joint_delta:
            print(
                f"[FrankaLeapRobot] Large GELLO drift detected (max delta: {max_delta:.4f} rad). "
                "Smoothly interpolating.\n"
                f"  Franka: {np.round(franka_current, 4)}\n"
                f"  GELLO:  {np.round(robot_target, 4)}"
            )
            self._smooth_move_to(robot_target)

        # Send arm joint target to Franka via deoxys (gripper_action=1.0 neutral)
        deoxys_action = list(robot_target) + [1.0]
        before_fwrite_t = time.perf_counter()
        self.robot_interface.control(
            controller_type=self.config.deoxys_controller_type,
            action=deoxys_action,
            controller_cfg=self.controller_cfg,
        )
        self.logs["write_follower_dt_s"] = time.perf_counter() - before_fwrite_t

        # --- Hand: read Manus glove and command LEAP hand ---
        before_hread_t = time.perf_counter()
        hand_target = None
        result = self.manus_mocap.get()
        if result['status'] == 'recording' and result['result'] is not None:
            hand_target = self.geort_model.forward(result['result'])
            self.leap_hand.set_allegro(hand_target)
        self.logs["read_manus_dt_s"] = time.perf_counter() - before_hread_t

        if not record_data:
            return

        # --- Record observation ---
        before_fread_t = time.perf_counter()
        franka_joints = self._get_franka_joints()
        hand_state = self._get_hand_state()
        self.logs["read_follower_dt_s"] = time.perf_counter() - before_fread_t

        # 7 arm + 16 hand
        state = torch.tensor(
            list(franka_joints) + list(hand_state), dtype=torch.float32
        )

        # Action: arm target + hand target (or current hand state if no glove data)
        hand_action = hand_target if hand_target is not None else hand_state
        action_tensor = torch.tensor(
            list(robot_target) + list(hand_action), dtype=torch.float32
        )

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            if type(images[name]) == dict:
                for img_name in images[name].keys():
                    images[name][img_name] = torch.from_numpy(images[name][img_name])
            else:
                images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionaries
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        action_dict["action"] = action_tensor
        for name in self.cameras:
            if type(images[name]) == dict:
                for img_name in images[name].keys():
                    obs_dict[f"observation.images.{name}.{img_name}"] = images[name][img_name]
            else:
                obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict, action_dict

    def capture_observation(self) -> dict:
        """Read Franka joint state + LEAP hand state + camera images."""
        if not self.is_connected:
            raise RuntimeError("FrankaLeapRobot is not connected. Run `robot.connect()` first.")

        before_fread_t = time.perf_counter()
        franka_joints = self._get_franka_joints()
        hand_state = self._get_hand_state()
        self.logs["read_follower_dt_s"] = time.perf_counter() - before_fread_t

        state = torch.tensor(
            list(franka_joints) + list(hand_state), dtype=torch.float32
        )

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            if type(images[name]) == dict:
                for img_name in images[name].keys():
                    images[name][img_name] = torch.from_numpy(images[name][img_name])
            else:
                images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        obs_dict = {}
        obs_dict["observation.state"] = state
        for name in self.cameras:
            if type(images[name]) == dict:
                for img_name in images[name].keys():
                    obs_dict[f"observation.images.{name}.{img_name}"] = images[name][img_name]
            else:
                obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict
    
    def get_pointcloud_obs(self, obs_dict) -> torch.Tensor:
        calibrations = self.calibrations
        camera_names = list(calibrations.keys())
        all_pcd = []
        for cam_name in camera_names:
            scaled_K = calibrations[cam_name]["scaled_K"]
            cam_to_world = calibrations[cam_name]["T_world_cam"]
            depth = obs_dict["observation.images.{}.transformed_depth".format(cam_name)].squeeze()
            pcd = compute_pcd(depth, scaled_K, 4000, 1.5, cam_to_world)
            all_pcd.append(pcd)

        all_pcd = np.concatenate(all_pcd, axis=0)
        scene_pcd_pt3d = torch.from_numpy(all_pcd)
        # print(all_pcd.shape)
        scene_pcd_downsample, scene_points_idx = sample_farthest_points(
            scene_pcd_pt3d[None], K=4000, random_start_point=False
        )
        joint_state = obs_dict['observation.state']
        scene_pcd = scene_pcd_downsample.squeeze().numpy().astype(np.float32)
        hand_pcd = self.hand_model.get_point_cloud(joint_state).astype(np.float32)
        obs_dict["observation.points.point_cloud"] = torch.from_numpy(np.concatenate([scene_pcd, hand_pcd], axis=0))
        obs_dict["observation.points.gripper_pcds"] = torch.from_numpy(self.hand_model.get_hand_skeleton(joint_state).astype(np.float32))
        return obs_dict

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        """Send a 23-value action tensor (7 arm joints + 16 hand joints) to the robot."""
        if not self.is_connected:
            raise RuntimeError("FrankaLeapRobot is not connected. Run `robot.connect()` first.")

        action_list = action.tolist()
        arm_target = np.array(action_list[:7])
        hand_target = np.array(action_list[7:])

        # Send arm joint target to Franka via deoxys (gripper_action=1.0 neutral)
        deoxys_action = list(arm_target) + [1.0]
        self.robot_interface.control(
            controller_type=self.config.deoxys_controller_type,
            action=deoxys_action,
            controller_cfg=self.controller_cfg,
        )

        # Send hand joint target to LEAP hand
        self.leap_hand.set_allegro(hand_target)

        self.logs["write_follower_dt_s"] = 0.0
        return action

    def print_logs(self):
        pass

    def disconnect(self):
        if not self.is_connected:
            return

        if self.robot_interface is not None:
            self.robot_interface.close()
            self.robot_interface = None

        # Move LEAP hand to open before disconnecting
        if self.leap_hand is not None:
            try:
                self._move_hand_to(np.zeros(16))
            except Exception:
                pass

        if self.manus_mocap is not None:
            try:
                self.manus_mocap.close()
            except Exception:
                pass

        self.gello = None
        self.leap_hand = None
        self.manus_mocap = None
        self.geort_model = None

        for name in self.cameras:
            self.cameras[name].disconnect()

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
