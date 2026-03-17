"""DummyRobot: Camera-only recording with no physical robot connection.

Mimics the DroidRobot interface but skips all hardware (Franka, GELLO, Robotiq).
Returns zero-filled state/action tensors so the dataset schema stays compatible
with real Droid datasets.
"""

import time

import torch

from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.robots.configs import DummyRobotConfig


class DummyRobot:
    """Dummy robot that only connects cameras. No robot hardware needed."""

    robot_type = "dummy"

    def __init__(self, config: DummyRobotConfig | None = None, **kwargs):
        if config is None:
            self.config = DummyRobotConfig(**kwargs)
        else:
            self.config = config

        self.cameras = make_cameras_from_configs(self.config.cameras)
        self.is_connected = False
        self.logs = {}
        self.robot_type = self.config.type
        self.use_eef = self.config.use_eef

        self.joint_names = [
            "joint_1", "joint_2", "joint_3", "joint_4",
            "joint_5", "joint_6", "joint_7", "gripper",
        ]

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
                "rot_6d_0", "rot_6d_1", "rot_6d_2", "rot_6d_3", "rot_6d_4", "rot_6d_5",
                "trans_0", "trans_1", "trans_2", "gripper_articulation",
            ]
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
            raise RuntimeError("DummyRobot is already connected.")

        from threading import Thread

        azure_kinect_cameras = []
        for name, camera in self.cameras.items():
            if camera.__class__.__name__ == "AzureKinectCamera":
                camera.connect(start_cameras=False)
                azure_kinect_cameras.append(camera)
            else:
                camera.connect()

        if len(azure_kinect_cameras) > 0:
            def start_camera(cam):
                cam.start()

            threads = [Thread(target=start_camera, args=(cam,)) for cam in azure_kinect_cameras]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        self.is_connected = True
        print("[DummyRobot] Connected cameras only (no robot hardware).")

    def run_calibration(self):
        pass

    def teleop_step(self, record_data=False):
        if not self.is_connected:
            raise RuntimeError("DummyRobot is not connected.")

        if not record_data:
            return

        state = torch.zeros(len(self.joint_names), dtype=torch.float32)
        action_tensor = torch.zeros(len(self.joint_names), dtype=torch.float32)

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
        if not self.is_connected:
            raise RuntimeError("DummyRobot is not connected.")

        state = torch.zeros(len(self.joint_names), dtype=torch.float32)

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

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        return action

    def teleop_safety_stop(self):
        pass

    def disconnect(self):
        if not self.is_connected:
            return
        for name in self.cameras:
            self.cameras[name].disconnect()
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
