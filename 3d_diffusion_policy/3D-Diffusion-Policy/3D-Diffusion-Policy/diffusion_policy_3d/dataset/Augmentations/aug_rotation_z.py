import pickle
import numpy as np

class rotationZ():
    def __init__(self, mean_angle_z, std_rot_z):
        self.mean_angle_z = mean_angle_z
        self.std_rot_z = std_rot_z


    def __call__(self,data):
        angle_degrees = np.random.normal(self.mean_angle_z, self.std_rot_z)
        angle_radians = np.radians(angle_degrees)
        rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians), 0],
        [np.sin(angle_radians),  np.cos(angle_radians), 0],
        [0,                     0,                      1]
        ])
        mean_of_point_cloud = np.mean(data["point_cloud"], axis=1, keepdims=True)
        data["point_cloud"]  = data["point_cloud"] - mean_of_point_cloud
        # Apply the rotation matrix to each point in the point cloud
        data["point_cloud"][0] = np.dot(rotation_matrix, data["point_cloud"][0].T).T
        data["point_cloud"]  = data["point_cloud"] + mean_of_point_cloud
        data["gripper_pcd"] -= mean_of_point_cloud
        data["gripper_pcd"][0] = np.dot(rotation_matrix, data["gripper_pcd"][0].T).T
        data["gripper_pcd"] += mean_of_point_cloud
        data["goal_gripper_pcd"] -= mean_of_point_cloud
        data["goal_gripper_pcd"][0] = np.dot(rotation_matrix, data["goal_gripper_pcd"][0].T).T
        data["goal_gripper_pcd"] += mean_of_point_cloud
        return data