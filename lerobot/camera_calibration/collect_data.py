"""
Uses Deoxys to control the robot and collect data for calibration.
"""
import numpy as np
import os, pickle
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import time

from pyk4a import PyK4A
from pyk4a.calibration import CalibrationType
import torch

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
from lerobot.scripts.yufei_policy_utils import rotation_transfer_matrix_to_6D, rotation_transfer_6D_to_matrix
from lerobot.common.robot_devices.control_utils import add_eef_pose

@parser.wrap()
def move_robot_and_record_data(
        cfg: ControlPipelineConfig,
    ):
    """
    Move the robot to random poses and record the necessary data.
    """
    
    # Initialize the robot
    ### TODO: change the robot controller to be aloha
    robot = make_robot_from_config(cfg.robot)
    robot.connect()


    cam_name = "cam_azure_kinect_front"
    cam_name = "cam_azure_kinect_back"
    # Initialize the camera
    camera = robot.cameras[cam_name].camera
    camera_matrix = camera.calibration.get_camera_matrix(CalibrationType.DEPTH)
    dist_coeffs = camera.calibration.get_distortion_coefficients(CalibrationType.DEPTH)

    data = []
    num_movements = 20
    initial_joint_positions = {
        "cam_azure_kinect_front": torch.tensor([91.3184,  193.0957,  193.4473,  157.2363,  157.4121,    5.9766,
          38.4961,   -9.4043,   46.1887,   87.5391,  134.3848,  134.2969,
         141.7676,  141.8555, -211.0254,   18.6328,  159.5215,    8.6773]).float(),
        "cam_azure_kinect_back": torch.tensor([  91.3184,  193.0957,  193.4473,  157.1484,  157.3242,    5.9766,
          38.4082,   -9.4043,   46.1887,  103.3594,   91.1426,   90.4395,
          98.5254,   98.7012, -168.9258,    4.1309,  201.8848,    8.5931]).float()
    }
    robot_adapter = AlohaAdapter(action_space="right_eef")
    
    import time
    for _ in tqdm(range(num_movements)):
        action = initial_joint_positions[cam_name]
        print("Sending action:", action)
        for _ in range(5):
            robot.send_action(action)
            time.sleep(0.5)

        observation = robot.capture_observation()
        cur_eef_pose = add_eef_pose(observation['observation.state']).cpu().numpy()
        random_delta_pos = np.random.uniform(-0.06, 0.06, size=(3,))    
        random_delta_axis_angle = np.random.uniform(-0.2, 0.2, size=(3,))
        target_position = cur_eef_pose[6:9] + random_delta_pos
        cur_rotation_matrix = rotation_transfer_6D_to_matrix(cur_eef_pose[:6])
        cur_rotation = Rotation.from_matrix(cur_rotation_matrix)
        delta_rotation = Rotation.from_rotvec(random_delta_axis_angle)
        target_rotation = (delta_rotation * cur_rotation).as_matrix()
        target_rotation_6d = rotation_transfer_matrix_to_6D(target_rotation)
        target_eef_pose = np.array(target_rotation_6d.tolist() + target_position.tolist() + [action[9].item()])
        target_eef_pose = torch.from_numpy(target_eef_pose).float()
        new_action = robot_adapter.transform_action(target_eef_pose, action)
        robot.send_action(new_action.squeeze(0))

        import time
        time.sleep(3)
        # Get current pose of the robot 
        observation = robot.capture_observation()
        cur_eef_pose = add_eef_pose(observation['observation.state']).cpu().numpy()
        cur_rotation_matrix = rotation_transfer_6D_to_matrix(cur_eef_pose[:6])
        gripper_pose = np.eye(4)
        gripper_pose[:3, :3] = cur_rotation_matrix
        gripper_pose[:3, 3] = cur_eef_pose[6:9]
        print(f"Gripper pos: {gripper_pose[:3, 3]}")

        # Capture IR frame from Kinect
        camera = robot.cameras[cam_name]
        ir_frame = get_kinect_ir_frame(camera)
        if ir_frame is not None:
            # Detect ArUco markers and get visualization
            corners, ids = detect_aruco_markers(ir_frame, debug=False)


            # Estimate transformation if marker is detected
            if ids is not None and len(ids) > 0:
                print("\033[92m" + f"Detected {len(ids)} markers." + "\033[0m")
                transform_matrix = estimate_transformation(corners, ids, camera_matrix, dist_coeffs)
                if transform_matrix is not None:
                    data.append((
                        gripper_pose,       # gripper pose in base
                        transform_matrix    # tag pose in camera
                    ))
            else:
                print("\033[91m" + "No markers detected." + "\033[0m")
        else:
            print("\033[91m" + "No IR frame captured." + "\033[0m")
    
    print(f"Recorded {len(data)} data points.")
    
    # Save data
    os.makedirs("data/calibration", exist_ok=True)
    filepath = f"data/calibration/cam{cam_name}_data.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    return filepath
    

if __name__ == "__main__":
    move_robot_and_record_data()