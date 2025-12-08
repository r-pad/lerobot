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
from test_PointNet2.model import PointNet2_small2


# goal_checkpoint_name = 'epoch-45.ckpt'
# goal_exp_dir = "/project_data/held/ziyuw2/Robogen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0807-200-obj-pred-goal-gripper-PointNet2-backbone-UNet-diffusion-ep-75-epsilon/2024.08.07/14.03.40_train_dp3_robogen_open_door"


# with hydra.initialize(config_path='../diffusion_policy_3d/config'):  # same config_path as used by @hydra.main
#     recomposed_config = hydra.compose(
#         config_name="dp3.yaml",  # same config_name as used by @hydra.main
#         overrides=OmegaConf.load("{}/.hydra/overrides.yaml".format(goal_exp_dir)),
#     )
# goal_cfg = recomposed_config

# goal_workspace = TrainDP3Workspace(goal_cfg)
# goal_checkpoint_dir = "{}/checkpoints/{}".format(goal_exp_dir, goal_checkpoint_name)
# goal_workspace.load_checkpoint(path=goal_checkpoint_dir)

# goal_policy = deepcopy(goal_workspace.model)
# if goal_workspace.cfg.training.use_ema:
#     goal_policy = deepcopy(goal_workspace.ema_model)
# goal_policy.eval()
# goal_policy.reset()
# goal_policy = goal_policy.to('cuda')

load_model_path = "/project_data/held/ziyuw2/Robogen-sim2real/test_PointNet2/results/displacement_weighted_gripper_all/model_18.pth"
pointnet2_model = PointNet2_small2(num_classes=13).to('cuda')
pointnet2_model.load_state_dict(torch.load(load_model_path))
pointnet2_model.eval()

input_path = "/project_data/held/ziyuw2/Robogen-sim2real/data/parallel_input_dict.pkl"
with open(input_path, "rb") as f:
    input_dict = pkl.load(f)

input_dict['point_cloud'] = input_dict['high_level_point_cloud']
input_dict['agent_pos'] = input_dict['high_level_agent_pos']
input_dict['gripper_pcd'] = input_dict['high_level_gripper_pcd']

for key in input_dict.keys():
    input_dict[key] = torch.tensor(input_dict[key]).to('cuda').unsqueeze(0)

# predict_goal = goal_policy.predict_action(input_dict)
# predict_goal = predict_goal['action'].reshape(-1, 4, 3)
with torch.no_grad():
    pointcloud = input_dict['point_cloud'][:, -1, :, :]
    gripper_pcd = input_dict['gripper_pcd'][:, -1, :]
    inputs = torch.cat([pointcloud, gripper_pcd], dim=1)
    inputs = inputs.to('cuda')
    inputs_ = inputs.permute(0, 2, 1)
    outputs = pointnet2_model(inputs_)
    weights = outputs[:, :, -1] # B, N
    outputs = outputs[:, :, :-1] # B, N, 12
    B, N, _ = outputs.shape
    outputs = outputs.view(B, N, 4, 3)
    outputs = outputs + inputs.unsqueeze(2)
    weights = torch.nn.functional.softmax(weights, dim=1)
    outputs = outputs * weights.unsqueeze(-1).unsqueeze(-1)
    outputs = outputs.sum(dim=1)
    outputs = outputs.unsqueeze(1)
    predict_goal = outputs.repeat(1, 2, 1, 1)
    predict_goal = predict_goal[0, 0, :].reshape(1, 4, 3)



input_dict['predicted_goal'] = predict_goal
for key in input_dict.keys():
    input_dict[key] = input_dict[key].detach().cpu().numpy().squeeze()

output_path = "/project_data/held/ziyuw2/Robogen-sim2real/local_exps/temp/parallel_pointnet_output_dict.pkl"
with open(output_path, "wb") as f:
    pkl.dump(input_dict, f)

