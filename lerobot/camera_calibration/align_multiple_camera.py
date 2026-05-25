"""
Align multiple Kinect point clouds in robot base frame using ICP.

Requires per-camera extrinsics from solve_calibration.py. Run after collecting data
and solving each camera individually.

Example:
```bash
python lerobot/camera_calibration/align_multiple_camera.py
```

Running without extra CLI args uses defaults from calibration_common.py.
"""
import copy
import json
import os
import sys

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from lerobot.camera_calibration.calibration_common import (
    CALIB_RESULTS_DIR,
    DEFAULT_CLI_ARGS,
    KINECT_CAMERA_NAMES,
    crop_point_cloud,
    get_colored_point_cloud_in_world,
    load_solved_camera_extrinsics,
)
from lerobot.common.robot_devices.control_configs import ControlPipelineConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.configs import parser

BASE_CAM_NAME = "cam_azure_kinect_back"


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def compute_align_to_target(target_pcd, other_pcds, threshold=0.01, visualize=False, base_cam_name=BASE_CAM_NAME):
    """Compute ICP alignments from other point clouds to the target/base cloud."""
    transforms = {}
    for cam_id, source in other_pcds.items():
        print(f":: Aligning camera {cam_id} with target pcd")
        print(":: Apply point-to-plane ICP")
        trans_init = np.identity(4)
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source,
            target_pcd,
            threshold,
            trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        if visualize:
            draw_registration_result(source, target_pcd, reg_p2p.transformation)
        transforms[cam_id] = reg_p2p.transformation.copy()

    transforms[base_cam_name] = np.identity(4)
    return transforms


def align_pcds(pcds, transforms):
    """Merge transformed point clouds."""
    transformed_pcds = o3d.geometry.PointCloud()
    for cam_name, pcd in pcds.items():
        transformed_pcds += pcd.transform(transforms[cam_name])
    return transformed_pcds


def build_world_point_clouds(observation, extrinsics):
    """Build cropped colored point clouds in robot base frame for each camera."""
    pcds = {}
    for cam_name in KINECT_CAMERA_NAMES:
        if cam_name not in extrinsics:
            print(f"Skipping {cam_name}: no solved extrinsic found.")
            continue
        pcd = get_colored_point_cloud_in_world(observation, cam_name, extrinsics[cam_name])
        pcds[cam_name] = crop_point_cloud(pcd)
        print(f"{cam_name}: {len(pcds[cam_name].points)} points after crop")
    return pcds


def save_alignment_results(transforms, save_dir=os.path.join("data", "calibration")):
    os.makedirs(save_dir, exist_ok=True)

    npz_path = os.path.join(save_dir, "camera_alignments.npz")
    print("Saving alignments to:", npz_path)
    np.savez(npz_path, **{str(cam_id): transform for cam_id, transform in transforms.items()})

    json_content = {}
    for cam_id, transform in transforms.items():
        quat = Rotation.from_matrix(transform[:3, :3]).as_quat()
        json_content[cam_id] = {
            "xyz": transform[:3, 3].tolist(),
            "quaternion": quat.tolist(),
        }
    json_path = os.path.join(save_dir, "camera_alignments.json")
    print("Saving alignments to:", json_path)
    with open(json_path, "w") as f:
        json.dump(json_content, f, indent=2)


@parser.wrap()
def main(cfg: ControlPipelineConfig):
    extrinsics = load_solved_camera_extrinsics(CALIB_RESULTS_DIR)
    missing = [cam for cam in KINECT_CAMERA_NAMES if cam not in extrinsics]
    if missing:
        raise FileNotFoundError(
            "Missing solved extrinsics for: "
            f"{missing}. Run solve_calibration.py for each camera first."
        )

    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    observation = robot.capture_observation()
    pcds = build_world_point_clouds(observation, extrinsics)
    if BASE_CAM_NAME not in pcds:
        raise RuntimeError(f"Base camera {BASE_CAM_NAME} point cloud is empty.")

    transforms = compute_align_to_target(
        pcds[BASE_CAM_NAME],
        pcds,
        threshold=0.01,
        visualize=False,
        base_cam_name=BASE_CAM_NAME,
    )

    ori_pcd = o3d.geometry.PointCloud()
    for pcd in pcds.values():
        ori_pcd += pcd
    print("Showing point clouds before ICP alignment...")
    o3d.visualization.draw_geometries([ori_pcd])

    camera_aligned_pcds = align_pcds(pcds, transforms)
    print("Showing point clouds after ICP alignment...")
    o3d.visualization.draw_geometries([camera_aligned_pcds])

    save_alignment_results(transforms)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(DEFAULT_CLI_ARGS)
    main()
