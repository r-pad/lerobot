import pickle
import numpy as np

class TranslationXY():
    def __init__(self, mean_x, mean_y, std_x, std_y, trans_x=False, trans_y=False):
        self.mean_x = mean_x
        self.mean_y = mean_y
        self.std_x = std_x
        self.std_y = std_y
        self.trans_x = trans_x
        self.trans_y = trans_y


    def __call__(self, data):
        if self.trans_x:
            random_value_x = np.random.normal(self.mean_x, self.std_x)
            data["point_cloud"][0,:,0] = data["point_cloud"][0,:,0] + random_value_x
            data["gripper_pcd"][0,:,0] = data["gripper_pcd"][0,:,0] + random_value_x
            data["goal_gripper_pcd"][0,:,0] = data["goal_gripper_pcd"][0,:,0] + random_value_x
        if self.trans_y:
            random_value_y = np.random.normal(self.mean_y, self.std_y)
            data["point_cloud"][0,:,1] = data["point_cloud"][0,:,1] + random_value_y
            data["gripper_pcd"][0,:,1] = data["gripper_pcd"][0,:,1] + random_value_y
            data["goal_gripper_pcd"][0,:,1] = data["goal_gripper_pcd"][0,:,1] + random_value_y
        
        return data

