"""
Visualize robot base frame and EEF frame axes projected onto the left camera image.
Saves one annotated image per second to --save_dir.

Usage:
    pixi run python lerobot/scripts/visualize_eef_pose.py --save_dir /tmp/eef_viz
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.common.robot_devices.cameras.configs import AzureKinectCameraConfig
from lerobot.common.robot_devices.control_utils import add_eef_pose
from lerobot.common.robot_devices.robots.configs import DroidRobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.policies.robot_adapters import transforms

CALIB_PATH = Path("/home/haotian/lerobot/polaris/PolaRiS-Hub/put_red_cup_no_curtain/cam_calibration.json")
AXIS_LEN = 0.08  # meters


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(rot6d).float().unsqueeze(0)
    return transforms.rotation_6d_to_matrix(t).squeeze(0).numpy()


def project_points(points_3d: np.ndarray, K: np.ndarray, extrinsic: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Project Nx3 points in robot base frame to image pixels."""
    pts_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])  # (N, 4)
    pts_cam = (extrinsic @ pts_h.T).T[:, :3]                       # (N, 3)
    pts_2d, _ = cv2.projectPoints(
        pts_cam.reshape(-1, 1, 3),
        np.zeros(3), np.zeros(3),  # already in camera frame
        K, dist
    )
    return pts_2d.reshape(-1, 2).astype(int)


def draw_frame(img, origin_px, axes_px, colors, label):
    """Draw 3 axes as arrows from origin, labeled X/Y/Z."""
    axis_names = ["X", "Y", "Z"]
    for i, (end_px, color, name) in enumerate(zip(axes_px, colors, axis_names)):
        cv2.arrowedLine(img, tuple(origin_px), tuple(end_px), color, 3, tipLength=0.2)
        tip = end_px + np.array([4, -4])
        cv2.putText(img, name, tuple(tip), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, name, tuple(tip), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(img, label, tuple(origin_px + np.array([5, -10])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, label, tuple(origin_px + np.array([5, -10])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="/tmp/eef_viz")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load calibration for left camera (cam1 = device_id 1)
    # extrinsic stored as cam→base; invert to get base→camera for projection
    calib = json.loads(CALIB_PATH.read_text())["cam1"]
    K         = np.array(calib["intrinsic"],  dtype=np.float64)
    extrinsic = np.linalg.inv(np.array(calib["extrinsic"], dtype=np.float64))
    dist      = np.array(calib["distortion"], dtype=np.float64)

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
    print(f"Robot connected. Saving to {save_dir}. Ctrl+C to stop.\n")

    frame_idx = 0
    try:
        while True:
            obs = robot.capture_observation()
            obs["observation.right_eef_pose"] = add_eef_pose(robot, obs["observation.state"])
            eef = obs["observation.right_eef_pose"]

            rot6d   = eef[0:6].numpy()
            trans   = eef[6:9].numpy()
            gripper = float(eef[9])

            R_eef = rot6d_to_matrix(rot6d)  # (3,3)

            # Rotation of +45° around world Z-axis
            a = np.deg2rad(-45)
            R_z45 = np.array([[np.cos(a), -np.sin(a), 0],
                               [np.sin(a),  np.cos(a), 0],
                               [0,          0,         1]])
            R_eef_rot = R_z45 @ R_eef

            # --- Base frame axes in robot base coordinates ---
            base_origin = np.zeros((1, 3))
            base_axes = np.eye(3) * AXIS_LEN  # X, Y, Z endpoints

            # --- EEF frame axes in robot base coordinates ---
            eef_origin = trans.reshape(1, 3)
            eef_axes   = trans + R_eef.T * AXIS_LEN  # (3,3): row i = origin + axis_i

            # --- EEF rotated +45° around world Z ---
            eef_rot_axes = trans + R_eef_rot.T * AXIS_LEN

            # Project all points
            all_pts = np.vstack([base_origin, base_axes, eef_origin, eef_axes, eef_rot_axes])
            px = project_points(all_pts, K, extrinsic, dist)

            base_o_px  = px[0]
            base_px    = px[1:4]
            eef_o_px   = px[4]
            eef_px     = px[5:8]
            eef_rot_px = px[8:11]

            raw_img = obs["observation.images.cam_azure_kinect_left.color"].numpy()
            colors     = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
            colors_rot = [(0, 128, 255), (128, 255, 0), (255, 0, 128)]

            def make_annotated(extra_fn):
                bgr = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
                draw_frame(bgr, base_o_px, base_px, colors, "base")
                extra_fn(bgr)
                cv2.putText(bgr, f"trans: [{trans[0]:.3f}, {trans[1]:.3f}, {trans[2]:.3f}]",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(bgr, f"trans: [{trans[0]:.3f}, {trans[1]:.3f}, {trans[2]:.3f}]",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(bgr, f"gripper: {gripper:.3f}",
                            (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(bgr, f"gripper: {gripper:.3f}",
                            (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
                return bgr

            img_eef     = make_annotated(lambda b: draw_frame(b, eef_o_px, eef_px,     colors,     "EEF"))
            img_eef_rot = make_annotated(lambda b: draw_frame(b, eef_o_px, eef_rot_px, colors_rot, "EEF-Z45"))

            cv2.imwrite(str(save_dir / f"frame_{frame_idx:05d}_eef.png"),     img_eef)
            cv2.imwrite(str(save_dir / f"frame_{frame_idx:05d}_eef_z45.png"), img_eef_rot)
            print(f"[{frame_idx:05d}] trans={[round(float(v),4) for v in trans]}  gripper={gripper:.4f}")

            frame_idx += 1
            time.sleep(1.0)

    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
