import open3d as o3d
import numpy as np
import copy

from lerobot.camera_calibration.marker_detection import get_kinect_ir_frame, detect_aruco_markers, estimate_transformation
from lerobot.common.robot_devices.control_configs import (
    CalibrateControlConfig,
    ControlConfig,
    ControlPipelineConfig,
    RecordControlConfig,
    RemoteRobotConfig,
    ReplayControlConfig,
    TeleoperateControlConfig,
)
from lerobot.configs import parser
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.common.policies.robot_adapters import AlohaAdapter
from lerobot.scripts.yufei_policy_utils import rotation_transfer_matrix_to_6D
from lerobot.common.robot_devices.control_utils import add_eef_pose


from lerobot.camera_calibration.marker_detection import get_kinect_ir_frame, detect_aruco_markers, estimate_transformation
from lerobot.camera_calibration.solve_calibration import estimate_tag_pose
from PIL import Image
from lerobot.scripts.yufei_policy_utils import get_scene_pcd_cam_frame, transform_to_world_frame, _load_camera_extrinsics, \
    _load_camera_intrinsics, rotation_transfer_6D_to_matrix

default_intrinsics = {}
default_extrinsics = {}
cameras = {
  "cam_azure_kinect_front": {
    "intrinsics": "/data/yufei/lerobot/lerobot/scripts/aloha_calibration/intrinsics_000259921812.txt",
    # "extrinsics": "/data/yufei/lerobot/lerobot/scripts/aloha_calibration/T_world_from_camera_front_v1_1020.txt"
    "extrinsics": "/data/yufei/lerobot/lerobot/scripts/aloha_calibration/T_world_from_camera_front_1208.txt"
  },
  "cam_azure_kinect_back": {
    "intrinsics": "/data/yufei/lerobot/lerobot/scripts/aloha_calibration/intrinsics_000003493812.txt",
    "extrinsics": "/data/yufei/lerobot/lerobot/scripts/aloha_calibration/T_world_from_camera_back_v1_1020.txt"
  }
}
for cam_name, cam_cfg in cameras.items():
    # Load intrinsics
    K = _load_camera_intrinsics(cam_cfg['intrinsics'])
    orig_shape = [720, 1280]
    default_intrinsics[cam_name] = K

    # Load extrinsics
    T = _load_camera_extrinsics(cam_cfg['extrinsics'])
    default_extrinsics[cam_name] = T
    # print(f"Loaded extrinsics for cam_name: {cam_name}: {T}")

new_front_calibration = np.load("/data/yufei/lerobot/data/calibration/calibration_results/camcam_azure_kinect_front_calibration.npz")['T']
new_back_calibration = np.load("/data/yufei/lerobot/data/calibration/calibration_results/camcam_azure_kinect_back_calibration.npz")['T']
default_extrinsics["cam_azure_kinect_front"] = new_front_calibration
default_extrinsics["cam_azure_kinect_back"] = new_back_calibration

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def compute_align_to_target(target_pcd, other_pcds, threshold=0.01, visualize=False, base_cam_name="cam_azure_kinect_front"):
    """
    Compute alignments from other_pcds to base_pcd
    
    Input:
        target_pcd: Open3D point cloud
        other_pcds: dict of Open3D point clouds {cam_id: pcd}
    Return:
        dict of transforms {cam_id: transforms}
    """
    transforms = {}
    for cam_id, source in other_pcds.items():        
        print(f":: Aligning camera {cam_id} with target pcd")
        print(":: Apply point-to-point ICP")
        trans_init = np.identity(4)
        # reg_p2p = o3d.pipelines.registration.registration_icp(
        #     source, target, threshold, trans_init,
        #     o3d.pipelines.registration.TransformationEstimationPointToPoint())
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target_pcd, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        if visualize:
            draw_registration_result(source, target_pcd, reg_p2p.transformation)

        transforms[cam_id] = reg_p2p.transformation.copy()
    
    # For base camer, set to identity
    transforms[base_cam_name] = np.identity(4)
    
    return transforms

def align_pcds(pcds, transforms):
    """
    Align point clouds using transforms
    
    Input:
        pcds: dict of Open3D point clouds {cam_id: pcd}
        transforms: dict of transforms {cam_id: transforms}.
    Return:
        Open3D point cloud
    """
    transformed_pcds = o3d.geometry.PointCloud()
    for cam_name in pcds.keys():
        transformed_pcd = pcds[cam_name].transform(transforms[cam_name])
        # transformed_pcd.paint_uniform_color(colors[cam_id])
        transformed_pcds += transformed_pcd
    
    return transformed_pcds

@parser.wrap()
def main(
    cfg: ControlPipelineConfig,
):
    base_cam_name = "cam_azure_kinect_front"    # Set all other cameras to align with camera 1

    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    observation = robot.capture_observation()
    pcds = {}
    for cam_name in ['cam_azure_kinect_front', 'cam_azure_kinect_back']:
        pcd = observation[f"observation.images.{cam_name}.point_cloud"].numpy() / 1000.
        pcd = pcd.astype(np.float32).reshape(-1, 3)
        pcd_in_world = transform_to_world_frame(pcd, default_extrinsics[cam_name])
        pcd_in_world = pcd_in_world[pcd_in_world[:, 2] > 0.01]
        pcd_in_world = pcd_in_world[pcd_in_world[:, 0] < 0.3 ]
        pcd_in_world = pcd_in_world[pcd_in_world[:, 0] > -0.3 ]
        pcd_in_world = pcd_in_world[pcd_in_world[:, 1] > -0.3 ]
        pcd_in_world = pcd_in_world[pcd_in_world[:, 1] < 0.3 ]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_in_world))
        pcds[cam_name] = pcd

    # Compute alignments
    transforms = compute_align_to_target(
        pcds[base_cam_name], pcds, threshold=0.01, visualize=False, base_cam_name=base_cam_name) # Set to True to visualize pairwise alignment


    # before alignment
    ori_pcd = o3d.geometry.PointCloud()
    for cam_name in pcds.keys():
        # pcds[cam_id].paint_uniform_color(colors[cam_id])
        ori_pcd += pcds[cam_name]
    o3d.visualization.draw_geometries([ori_pcd])

    # Transform all point clouds
    camera_aligned_pcds = align_pcds(pcds, transforms)
    o3d.visualization.draw_geometries([camera_aligned_pcds])

    # Remove color
    # camera_aligned_pcds_ = o3d.geometry.PointCloud(camera_aligned_pcds.points)
    # camera_aligned_pcds_.normals = camera_aligned_pcds.normals 
    # o3d.visualization.draw_geometries([camera_aligned_pcds_])


    """Save the results"""
    import os, json
    from scipy.spatial.transform import Rotation

    save_dir = os.path.join("data/calibration")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the transform matrices as npz
    save_path = os.path.join(save_dir, "camera_alignments.npz")
    print("saving alignments to: ", save_path)
    save_content = {
        str(cam_id): transform for cam_id, transform in transforms.items()
    } # Convert cam_id to string to save as npz
    np.savez(save_path, **save_content)

    # Also save the humann readable transforms as xyz, quat into json
    save_content = {}
    for cam_id, transform in transforms.items():
        quat = Rotation.from_matrix(transform[:3, :3]).as_quat()
        save_content[cam_id] = {
            "xyz": transform[:3, 3].tolist(),
            "quaternion": quat.tolist()
        }
    save_path = os.path.join(save_dir, "camera_alignments.json")
    print("saving alignments to: ", save_path)
    with open(save_path, "w") as f:
        json.dump(save_content, f, indent=2)

if __name__ == "__main__":
    main()