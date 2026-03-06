import os
import pickle
import numpy as np
import open3d as o3d
from copy import deepcopy
from tqdm import tqdm
import itertools
import torch
from matplotlib import pyplot as plt
import time

from scipy.spatial.transform import Rotation
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

def get_pcd_in_world(depth,cam_name="cam_azure_kinect_front"):
    depth = depth / 1000.0

    pcd_in_camera, _ = get_scene_pcd_cam_frame(
        depth, default_intrinsics[cam_name], None, 1.5
    )    

    # import pdb; pdb.set_trace()
    pcd_in_world = transform_to_world_frame(pcd_in_camera, default_extrinsics[cam_name])
    pcd_in_world = pcd_in_camera
    
    return pcd_in_world

@parser.wrap()
def visualize(
        cfg: ControlPipelineConfig,
    ):
    cam_name = "cam_azure_kinect_front"

    ### TODO: change the initial joint positions
    initial_joint_positions = {
        "cam_azure_kinect_front": torch.tensor([91.3184,  193.0957,  193.4473,  157.2363,  157.4121,    5.9766,
          38.4961,   -9.4043,   46.1887,   87.5391,  134.3848,  134.2969,
         141.7676,  141.8555, -211.0254,   18.6328,  159.5215,    8.6773]).float(),
        "cam_azure_kinect_back": torch.tensor([  91.3184,  193.0957,  193.4473,  157.1484,  157.3242,    5.9766,
        38.4082,   -9.4043,   46.1887,  103.3594,   91.1426,   90.4395,
        98.5254,   98.7012, -168.9258,    4.1309,  201.8848,    8.5931]).float()
    }
    
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    robot_adapter = AlohaAdapter(action_space="right_eef")

    observation = robot.capture_observation()
    state = observation['observation.state']
    print("Initial state:", state)
    rgb_key = f"observation.images.{cam_name}.color"
    rgb = observation[rgb_key].numpy()
    plt.imshow(rgb)
    plt.show()

    for _ in range(10):
        action = initial_joint_positions[cam_name]
        print("Sending action:", action)
        for _ in range(5):
            robot.send_action(action)
            time.sleep(1)
        # new_action = initial_joint_positions + np.random.uniform(-5, 5, size=initial_joint_positions.shape)

        observation = robot.capture_observation()
        cur_eef_pose = add_eef_pose(observation['observation.state']).cpu().numpy()
        random_delta_pos = np.random.uniform(-0.06, 0.06, size=(3,))    
        random_delta_axis_angle = np.random.uniform(-0.2, 0.2, size=(3,))
        target_position = cur_eef_pose[6:9] + random_delta_pos
        cur_rotation = rotation_transfer_6D_to_matrix(cur_eef_pose[:6])
        cur_rotation = Rotation.from_matrix(cur_rotation)
        delta_rotation = Rotation.from_rotvec(random_delta_axis_angle)
        target_rotation = (delta_rotation * cur_rotation).as_matrix()
        target_rotation_6d = rotation_transfer_matrix_to_6D(target_rotation)
        target_eef_pose = np.array(target_rotation_6d.tolist() + target_position.tolist() + [action[9].item()])
        target_eef_pose = torch.from_numpy(target_eef_pose).float()
        new_action = robot_adapter.transform_action(target_eef_pose, action)
        robot.send_action(new_action.squeeze(0))

        time.sleep(2)

        # observation = robot.capture_observation()
        # depth_key = "observation.images.{}.transformed_depth".format(cam_name)
        # observation = robot.capture_observation()
        # depth = Image.fromarray(observation[depth_key].numpy()[:, :, 0])
        # depth = np.asarray(depth)
        # pcd = get_pcd_in_world(depth, cam_name=cam_name)

        pcd = observation[f"observation.images.{cam_name}.point_cloud"].numpy() / 1000.
        pcd = pcd.astype(np.float32).reshape(-1, 3)
        pcd_in_world = transform_to_world_frame(pcd, default_extrinsics[cam_name])
        pcd = pcd_in_world
        
        # gripper eef base location
        eef_pose = add_eef_pose(observation['observation.state']).cpu().numpy()
        eef_orient_6d = eef_pose[:6]
        eef_pos = eef_pose[6:9]
        eef_rotation = rotation_transfer_6D_to_matrix(eef_orient_6d)
        eef_pose_matrix = np.eye(4)
        eef_pose_matrix[:3, :3] = eef_rotation
        eef_pose_matrix[:3, 3] = eef_pos
        print("eef_pose", np.round(eef_pose, 2))
        
        eef_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        eef_coord.transform(eef_pose_matrix)

        # Estimate ArUco marker pose from forward kinematics
        _, tag_pose = estimate_tag_pose(eef_pose_matrix)
        aruco_coord_robot = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        aruco_coord_robot.transform(tag_pose)

        # Calculate ArUco marker pose from main camera
        aruco_coord_cam = None

        # aruco_coord_cam = get_aruco_coord(robot.cameras[cam_name])
        # if aruco_coord_cam is None:
        #     continue

        # Visualize
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
        o3d.visualization.draw_geometries([
            pcd, 
            eef_coord, 
            # aruco_coord_cam, 
            # aruco_coord_robot
        ])

def get_aruco_coord(camera):
    # Obtain the main camera IR image
    ir_frame = get_kinect_ir_frame(camera)
    corners, ids = detect_aruco_markers(ir_frame, debug=True)
    print(f"Detected corners: {corners}, ids: {ids}")
    if len(corners) == 0:
        print("No ArUco markers detected.")
        return None
    else:
        # Estimate the transformation matrix
        from pyk4a.calibration import CalibrationType
        camera = camera.camera
        camera_matrix = camera.calibration.get_camera_matrix(CalibrationType.DEPTH)
        dist_coeffs = camera.calibration.get_distortion_coefficients(CalibrationType.DEPTH)
        pose_in_cam = estimate_transformation(corners, ids, camera_matrix, dist_coeffs)
    
        # Convert the pose to the world frame
        aruco_coord_in_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        aruco_coord_in_cam.transform(pose_in_cam)
        # return aruco_coord_in_cam

        # points = np.asarray(aruco_coord_in_cam.vertices)
        # aruco_coord = transform_to_world_frame(points, default_extrinsics[cam_name])

        aruco_coord_in_cam = aruco_coord_in_cam.transform(default_extrinsics[cam_name]) # Apply camera extrinc
        # Convert the pose to the observation frame
        return aruco_coord_in_cam


if __name__ == "__main__":
    visualize()
    
    