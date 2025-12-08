import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from train import TrainDP3Workspace
from diffusion_policy_3d.common.pytorch_util import dict_apply
from manipulation.utils import build_up_env, save_numpy_as_gif, save_env
import pybullet as p
import numpy as np
from copy import deepcopy
import sys
from termcolor import cprint
from manipulation.robogen_wrapper import RobogenPointCloudWrapper
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from manipulation.gpt_reward_api import get_joint_state
import tqdm
import json
from multiprocessing import set_start_method
from multiprocessing import Pool
import time
import yaml
import pickle as pkl
from manipulation.utils import get_pc, get_pc_in_camera_frame, rotation_transfer_6D_to_matrix, rotation_transfer_matrix_to_6D, add_sphere, get_pixel_location
import cv2
import scipy

original_data = "temp"



# goal_checkpoint_name = 'epoch-24.ckpt'
# goal_exp_dir = "/project_data/held/ziyuw2/Robogen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0730-50-obj-pred-goal-gripper-pointnet-backbone-unet-diffusion-epsilon/2024.07.30/17.31.40_train_dp3_robogen_open_door"

goal_checkpoint_name = 'epoch-45.ckpt'
goal_exp_dir = "/project_data/held/ziyuw2/Robogen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0807-200-obj-pred-goal-gripper-PointNet2-backbone-UNet-diffusion-ep-75-epsilon/2024.08.07/14.03.40_train_dp3_robogen_open_door"


with hydra.initialize(config_path='../diffusion_policy_3d/config'):  # same config_path as used by @hydra.main
    recomposed_config = hydra.compose(
        config_name="dp3.yaml",  # same config_name as used by @hydra.main
        overrides=OmegaConf.load("{}/.hydra/overrides.yaml".format(goal_exp_dir)),
    )
goal_cfg = recomposed_config

goal_workspace = TrainDP3Workspace(goal_cfg)
goal_checkpoint_dir = "{}/checkpoints/{}".format(goal_exp_dir, goal_checkpoint_name)
goal_workspace.load_checkpoint(path=goal_checkpoint_dir)

goal_policy = deepcopy(goal_workspace.model)
if goal_workspace.cfg.training.use_ema:
    goal_policy = deepcopy(goal_workspace.ema_model)
goal_policy.eval()
goal_policy.reset()
goal_policy = goal_policy.to('cuda')




for i in range(3):
    data_path = f"/project_data/held/ziyuw2/Robogen-sim2real/local_exps/{original_data}/{i}.pkl"
    with open(data_path, "rb") as f:
        data = pkl.load(f)
    pcd = np.array(data["pcd"])
    gripper_pcd = np.array(data["gripper_pcd"])
    agent_pos = np.array(data["agent_pos"])

    observation_dict = {}
    observation_dict["point_cloud"] = pcd
    observation_dict['gripper_pcd'] = gripper_pcd
    distance = scipy.spatial.distance.cdist(gripper_pcd, pcd)
    min_distance_obj_idx = np.argmin(distance, axis=1)
    closest_point = pcd[min_distance_obj_idx]
    displacement = closest_point - gripper_pcd
    observation_dict['displacement_gripper_to_object'] = displacement.astype(np.float32)
    observation_dict['agent_pos'] = agent_pos.astype(np.float32)

    obs_dict_input = dict_apply(observation_dict, lambda x: torch.from_numpy(x).float().to('cuda'))
    obs_dict_input = dict_apply(obs_dict_input, lambda x: torch.stack([x, x], dim=0))
    obs_dict_input = dict_apply(obs_dict_input, lambda x: x.unsqueeze(0))

    with torch.no_grad():
        predicted_goal = goal_policy.predict_action(obs_dict_input)

    predicted_goal = predicted_goal['action'].detach().to('cpu').numpy()
    predicted_goal = predicted_goal[0, :2, :]
    predicted_goal = predicted_goal.reshape(2, 4, 3)
    predicted_goal = predicted_goal[0]
    save_data_path = f"/project_data/held/ziyuw2/Robogen-sim2real/local_exps/{original_data}/result_200_{i}.pkl"
    save_data = {
        "pcd": pcd,
        "gripper_pcd": gripper_pcd,
        "agent_pos": agent_pos,
        "predicted_goal": predicted_goal
    }
    with open(save_data_path, "wb") as f:
        pkl.dump(save_data, f)

