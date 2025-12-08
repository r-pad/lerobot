import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from diffusion_policy_3d.common.pytorch_util import dict_apply
import pybullet as p
import numpy as np
from copy import deepcopy
import sys
from termcolor import cprint
from test_PointNet2.model import PointNet2_small2
import tqdm
import json
import time
import yaml
import pickle as pkl
import cv2
import scipy

original_data = "test_pcd_microwave_0"

load_model_path = "/project_data/held/ziyuw2/Robogen-sim2real/test_PointNet2/results/displacement_weighted_gripper_all/model_18.pth"
pointnet2_model = PointNet2_small2(num_classes=13).to('cuda')
pointnet2_model.load_state_dict(torch.load(load_model_path))
pointnet2_model.eval()


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
        pointcloud = obs_dict_input['point_cloud'][:, -1, :, :]
        gripper_pcd = obs_dict_input['gripper_pcd'][:, -1, :]
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
        predicted_goal = outputs.repeat(1, 2, 1, 1)

    predicted_goal = predicted_goal[0, :2, :]
    predicted_goal = predicted_goal.reshape(2, 4, 3)
    predicted_goal = predicted_goal[0].detach().cpu().numpy()
    save_data_path = f"/project_data/held/ziyuw2/Robogen-sim2real/local_exps/{original_data}/result_pointnet_{i}.pkl"
    save_data = {
        "pcd": pcd.cpu().numpy() if isinstance(pcd, torch.Tensor) else pcd,
        "gripper_pcd": gripper_pcd.cpu().numpy()[0] if isinstance(gripper_pcd, torch.Tensor) else gripper_pcd[0],
        "agent_pos": agent_pos.cpu().numpy() if isinstance(agent_pos, torch.Tensor) else agent_pos,
        "predicted_goal": predicted_goal.cpu().numpy() if isinstance(predicted_goal, torch.Tensor) else predicted_goal
    }
    with open(save_data_path, "wb") as f:
        pkl.dump(save_data, f)

