"""
Read and print the current robot state and EEF pose.
Use this to verify/adjust the robot's starting position before inference.

Usage:
    pixi run python lerobot/scripts/read_robot_state.py
"""

import argparse
import time
from pathlib import Path

import cv2
import torch
from lerobot.common.robot_devices.robots.configs import DroidRobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.robot_devices.cameras.configs import AzureKinectCameraConfig
from lerobot.common.robot_devices.control_utils import add_eef_pose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="/tmp/robot_state_images")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    robot_config = DroidRobotConfig(
        cameras={
            "cam_azure_kinect_left": AzureKinectCameraConfig(
                device_id=1, fps=30, width=1280, height=720,
                use_transformed_depth=False, wired_sync_mode="master",
            ),
        },
        use_eef=True,
        skip_gello_calibration=True,
    )
    robot = make_robot_from_config(robot_config)
    robot.connect()
    print(f"Robot connected. Images saved to {save_dir}. Ctrl+C to stop.\n")

    frame_idx = 0
    try:
        while True:
            obs = robot.capture_observation()
            state = obs["observation.state"]
            eef = add_eef_pose(robot, state)

            joints  = state[:7].numpy()
            gripper = state[7].item()
            trans   = eef[6:9].numpy()
            rot6d   = eef[0:6].numpy()

            print(f"--- {time.strftime('%H:%M:%S')} ---")
            print(f"  joints  : {[round(float(v), 4) for v in joints]}")
            print(f"  gripper : {float(gripper):.4f}")
            print(f"  trans   : {[round(float(v), 4) for v in trans]}")
            print(f"  rot6d   : {[round(float(v), 4) for v in rot6d]}")
            print()

            # Save left camera image
            img_key = "observation.images.cam_azure_kinect_left.color"
            if img_key in obs:
                img = obs[img_key].numpy()  # (H, W, 3) uint8 RGB
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_path = save_dir / f"frame_{frame_idx:05d}.png"
                cv2.imwrite(str(img_path), img_bgr)
                print(f"  image saved: {img_path}")

            frame_idx += 1
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
