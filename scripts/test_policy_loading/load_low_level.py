from train_ddp import TrainDP3Workspace
import hydra
from omegaconf import OmegaConf
from copy import deepcopy
import torch
import os
import numpy as np

def low_level_policy_infer(obj_pcd, agent_pos, goal_gripper_pcd, gripper_pcd, policy):
    # TODO: get the 6d orientation from the eef_quat
    # TODO: repeat this to the horizon length
    input_dict = {
        "point_cloud": obj_pcd.unsqueeze(1).repeat(1, 2, 1, 1),
        "agent_pos": agent_pos.unsqueeze(1).repeat(1, 2, 1),
        'gripper_pcd': gripper_pcd.unsqueeze(1).repeat(1, 2, 1, 1),
        'goal_gripper_pcd': goal_gripper_pcd.unsqueeze(1).repeat(1, 2, 1, 1),
    }

    cat_idx = 13
    batched_action = policy.predict_action(input_dict, torch.tensor([cat_idx]).to(policy.device))

    return batched_action['action'] # B, T, 10

exp_dir = "/data/yufei/lerobot/data/low-level-ckpt/1204_finetune_ours_sriram_plate_combine_2_step_train_longer_keep_old_normalizer" 
checkpoint_name = "epoch-300.ckpt"
with hydra.initialize(config_path='../../3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/config'):  # same config_path as used by @hydra.main
    recomposed_config = hydra.compose(
        config_name="dp3.yaml",  # same config_name as used by @hydra.main
        overrides=OmegaConf.load("{}/.hydra/overrides.yaml".format(exp_dir)),
    )
    cfg = recomposed_config
    
workspace = TrainDP3Workspace(cfg)
checkpoint_dir = "{}/checkpoints/{}".format(exp_dir, checkpoint_name)
workspace.load_checkpoint(path=checkpoint_dir)

policy = deepcopy(workspace.model)
if workspace.cfg.training.use_ema:
    policy = deepcopy(workspace.ema_model)
policy.eval()
policy.reset()
policy = policy.to('cuda')

traj_idx = 0
traj_path = f"/data/yufei/lerobot/data/sriram_plate/traj_{str(traj_idx).zfill(4)}"
t = 0
data_path = os.path.join(traj_path, f"{t}.npz")
data = np.load(data_path, allow_pickle=True)

obj_pcd_np = data['point_cloud'].reshape(-1, 3)
gripper_pcd_np = data['gripper_pcd'].reshape(-1, 3)
goal_gripper_pcd_np = data['goal_gripper_pcd'].reshape(-1, 3)
agent_pos = data['state'].reshape(1, 10)  # (T, 10)

scene_pcd = obj_pcd_np
pcd_world = scene_pcd
scene_pcd = torch.from_numpy(scene_pcd).float().to('cuda') # 4500 * 3

actions = low_level_policy_infer(
    scene_pcd,
    torch.from_numpy(agent_pos).to('cuda').float(), 
    torch.from_numpy(goal_gripper_pcd_np).float().to('cuda').reshape(1, 4, 3),
    torch.from_numpy(gripper_pcd_np).float().to('cuda').reshape(1, 4, 3),
    policy
)


