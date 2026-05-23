"""Build franka_leap calibration txt files from raw_calibration npz outputs.

Background
----------
The raw calibration was produced by RoboGen-sim2real's
`pcd_obs_env_bowen/calibration` workflow. ArUco markers are detected with
DEPTH-camera intrinsics (see marker_detection.py), so each `cam{i}_calibration.npz`
holds T_world_from_DEPTH_cam. `camera_alignments.npz` is the per-camera ICP
correction in world frame (cam1 is the base, so its alignment is identity).

LeRobot's franka_leap pipeline (compute_pcd in franka_leap.py) unprojects
`transformed_depth` using COLOR intrinsics and then applies the loaded extrinsic.
So the extrinsic must be T_world_from_COLOR_cam, not T_world_from_DEPTH_cam.
Mixing those up was the prior depth/color swap bug.

Composition
-----------
    T_world_from_color = align[i] @ raw_T[i] @ T_DEPTH_from_COLOR

T_DEPTH_from_COLOR is queried live from the Azure Kinect SDK calibration
(`get_extrinsic_parameters(COLOR, DEPTH)`).

Mapping (matches replay_debug_syringe_3.py and the existing JSON config):
    raw cam0  -> output cam0  (cam_azure_kinect_front, device_id=0)
    raw cam1  -> output cam2  (cam_azure_kinect_side,  device_id=1)
"""

from pathlib import Path

import numpy as np
from pyk4a import Config, PyK4A
from pyk4a.calibration import CalibrationType
from pyk4a.config import ColorResolution, DepthMode, ImageFormat


REPO = Path("/home/leap/Desktop/lerobot_restore/lerobot")
RAW_DIR = REPO / "raw_calibration"
OUT_DIR = REPO / "lerobot/scripts/franka_leap_calibration"

CAMERAS = [
    {"raw_id": 0, "out": "cam0", "device_id": 0},
    {"raw_id": 1, "out": "cam2", "device_id": 1},
]


def get_color_intrinsics_and_depth_from_color(device_id):
    config = Config(
        color_resolution=ColorResolution.RES_720P,
        color_format=ImageFormat.COLOR_BGRA32,
        depth_mode=DepthMode.NFOV_UNBINNED,
        synchronized_images_only=True,
    )
    k4a = PyK4A(config=config, device_id=device_id, thread_safe=False)
    k4a.open()
    try:
        K_color = k4a.calibration.get_camera_matrix(CalibrationType.COLOR)
        # get_extrinsic_parameters(source, target) returns (R, t) such that
        #   p_target = R @ p_source + t   (translation already in meters in pyk4a)
        # so (COLOR, DEPTH) gives T_DEPTH_from_COLOR.
        R, t = k4a.calibration.get_extrinsic_parameters(
            CalibrationType.COLOR, CalibrationType.DEPTH
        )
        T_depth_from_color = np.eye(4)
        T_depth_from_color[:3, :3] = R
        T_depth_from_color[:3, 3] = t.flatten()
    finally:
        k4a.close()
    return K_color, T_depth_from_color


def main():
    alignments = np.load(RAW_DIR / "camera_alignments.npz")
    np.set_printoptions(precision=4, suppress=True)

    for cam in CAMERAS:
        raw_id, out, device_id = cam["raw_id"], cam["out"], cam["device_id"]
        raw_T = np.load(RAW_DIR / f"cam{raw_id}_calibration.npz")["T"]
        align = alignments[str(raw_id)]

        print(f"\n=== {out}  (raw cam{raw_id}, Kinect device_id={device_id}) ===")
        print("raw_T  (T_world_from_DEPTH, before alignment):")
        print(raw_T)
        print("alignment  (world-frame ICP correction):")
        print(align)

        K_color, T_depth_from_color = get_color_intrinsics_and_depth_from_color(device_id)
        print("color K (1280x720):")
        print(K_color)
        print("T_DEPTH_from_COLOR  (from Kinect SDK):")
        print(T_depth_from_color)

        T_world_color = align @ raw_T @ T_depth_from_color
        print("T_world_from_COLOR  (final extrinsic):")
        print(T_world_color)

        intr_path = OUT_DIR / f"{out}_intrinsics.txt"
        extr_path = OUT_DIR / f"{out}_extrinsics.txt"
        np.savetxt(intr_path, K_color)
        np.savetxt(extr_path, T_world_color)
        print(f"wrote {intr_path}")
        print(f"wrote {extr_path}")


if __name__ == "__main__":
    main()
