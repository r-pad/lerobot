import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import hydra
from train_ddp import TrainDP3Workspace
from omegaconf import OmegaConf
import os
from copy import deepcopy
import torch
from manipulation.utils import rotation_transfer_matrix_to_6D, rotation_transfer_6D_to_matrix

def get_4_points_from_gripper_pos_orient(gripper_pos, gripper_orn, cur_joint_angle):
    original_gripper_pcd = np.array([[ 0.5648266,   0.05482348,  0.34434554],
        [ 0.5642125,   0.02702148,  0.2877661 ],
        [ 0.53906703,  0.01263776,  0.38347825],
        [ 0.54250515, -0.00441092,  0.32957944]]
    )
    original_gripper_orn = np.array([0.21120763,  0.75430543, -0.61925177, -0.05423936])
    
    gripper_pcd_right_finger_closed = np.array([ 0.55415434,  0.02126799,  0.32605097])
    gripper_pcd_left_finger_closed = np.array([ 0.54912525,  0.01839125,  0.3451934 ])
    gripper_pcd_closed_finger_angle = 2.6652539383870777e-05
 
    original_gripper_pcd[1] = gripper_pcd_right_finger_closed + (original_gripper_pcd[1] - gripper_pcd_right_finger_closed) / (0.04 - gripper_pcd_closed_finger_angle) * (cur_joint_angle - gripper_pcd_closed_finger_angle)
    original_gripper_pcd[2] = gripper_pcd_left_finger_closed + (original_gripper_pcd[2] - gripper_pcd_left_finger_closed) / (0.04 - gripper_pcd_closed_finger_angle) * (cur_joint_angle - gripper_pcd_closed_finger_angle)
 
    # goal_R = R.from_quat(gripper_orn)
    # import pdb; pdb.set_trace()
    goal_R = R.from_matrix(gripper_orn)
    original_R = R.from_quat(original_gripper_orn)
    rotation_transfer = goal_R * original_R.inv()
    original_pcd = original_gripper_pcd - original_gripper_pcd[3]
    rotated_pcd = rotation_transfer.apply(original_pcd)
    gripper_pcd = rotated_pcd + gripper_pos
    return gripper_pcd

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



# ### load a low-level policy
# exp_dir = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1204_finetune_ours_sriram_plate/2025.12.04/22.14.09_train_dp3_robogen_open_door/"
# checkpoint_name = 'epoch-80.ckpt'
exp_dir = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data/ckpts/1020_grasp_lift_closed_goal_full"
checkpoint_name = 'epoch-92.ckpt'
# exp_dir = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data/ckpts/0930_full_closer_no_open"
# checkpoint_name = 'epoch-60.ckpt'
# exp_dir = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1204_finetune_ours_sriram_plate_using_open_lang_embedding/2025.12.05/13.14.16_train_dp3_robogen_open_door/"
# checkpoint_name = 'epoch-80.ckpt'
# exp_dir = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1204_finetune_ours_sriram_plate_using_open_lang_embedding_combine_2_step/2025.12.05/15.45.22_train_dp3_robogen_open_door"
# checkpoint_name = 'epoch-80.ckpt'
# suffix = "combined_2_step"
# exp_dir = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1204_finetune_ours_sriram_plate_combine_2_step/2025.12.05/23.32.40_train_dp3_robogen_open_door"
# checkpoint_name = 'epoch-80.ckpt'
# exp_dir = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1204_finetune_ours_sriram_plate_combine_2_step_train_longer/2025.12.06/01.22.45_train_dp3_robogen_open_door/"
# checkpoint_name = 'epoch-300.ckpt'
# exp_dir = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1204_finetune_ours_sriram_plate_combine_2_step_train_longer_keep_old_normalizer/2025.12.06/20.09.00_train_dp3_robogen_open_door/"
# checkpoint_name = 'epoch-160.ckpt'
suffix = "combine_2_step"

with hydra.initialize(config_path='diffusion_policy_3d/config'):  # same config_path as used by @hydra.main
    recomposed_config = hydra.compose(
        config_name="dp3.yaml",  # same config_name as used by @hydra.main
        overrides=OmegaConf.load("{}/.hydra/overrides.yaml".format(exp_dir)),
    )
cfg = recomposed_config

workspace = TrainDP3Workspace(cfg)
checkpoint_dir = "{}/checkpoints/{}".format(exp_dir, checkpoint_name)
workspace.load_checkpoint(path=checkpoint_dir, )

#Low level policy loading 
policy = deepcopy(workspace.model)
if workspace.cfg.training.use_ema:
    policy = deepcopy(workspace.ema_model)
policy.eval()
policy.reset()
policy = policy.to('cuda')

traj_idx = 0
traj_path = f"/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data/aloha/plate/traj_{str(traj_idx).zfill(4)}"
traj_path = f"/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data/aloha_combined_2_step_0/plate/traj_{str(traj_idx).zfill(4)}"
# traj_path = f"/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data/aloha/plate/traj_{str(traj_idx).zfill(4)}"
# traj_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data/dp3_demo/bucket_100435/2025-05-11-15-06-27"
images = []
mean_action_magnitude = []

all_np_files = [x for x in os.listdir(traj_path) if x.endswith('.npz')]
# all_np_files = [x for x in os.listdir(traj_path) if x.endswith('.pkl')]
timesteps = len(all_np_files)
for t in range(17, timesteps):

    data_path = os.path.join(traj_path, f"{t}.npz")
    data = np.load(data_path, allow_pickle=True)
    # data_path = os.path.join(traj_path, f"{t}.pkl")
    # with open(data_path, 'rb') as f:
    #     import pickle as pkl
    #     data = pkl.load(f)

    obj_pcd_np = data['point_cloud'].reshape(-1, 3)
    gripper_pcd_np = data['gripper_pcd'].reshape(-1, 3)
    goal_gripper_pcd_np = data['goal_gripper_pcd'].reshape(-1, 3)
    agent_pos = data['state'].reshape(1, 10)  # (T, 10)
    
    gt_actions = []
    for future_t in range(4):
        test_t = t + future_t
        data = np.load(os.path.join(traj_path, f"{test_t}.npz"), allow_pickle=True)
        # with open(os.path.join(traj_path, f"{test_t}.pkl"), 'rb') as f:
            # data = pkl.load(f)
            
        gt_actions.append(data['action'].reshape(1, 10))  # (T, 10)
    gt_delta_translations = [x[0,:3] for x in gt_actions]   

    scene_pcd = obj_pcd_np
    pcd_world = scene_pcd
    scene_pcd = torch.from_numpy(scene_pcd).float().to('cuda') # 4500 * 3
    
    ### get the action from the low level policy
    # predicted_goal, obj_pcd, eef_pos, eef_quat, gripper_q, policy
    actions = low_level_policy_infer(
        scene_pcd,
        torch.from_numpy(agent_pos).to('cuda').float(), 
        torch.from_numpy(goal_gripper_pcd_np).float().to('cuda').reshape(1, 4, 3),
        torch.from_numpy(gripper_pcd_np).float().to('cuda').reshape(1, 4, 3),
        policy
    )
    
    actions = actions.detach().cpu()  # B, T, 10
    # print("pred actions:", actions.numpy().reshape(4, 10))
    # print("gt actions:", np.array(gt_actions).reshape(4, 10))
    
    ### update the eef pos and orient
    actions = actions[0]
    mean_delta_translation = torch.mean(torch.norm(actions[:, :3], dim=1)).item()
    mean_action_magnitude.append(mean_delta_translation)

    cur_eef_pos = agent_pos[0, :3]
    future_pred_pos = [cur_eef_pos]
    tmp_pos = cur_eef_pos.copy()
    for act in actions:
        tmp_pos = tmp_pos + act[:3].numpy()
        future_pred_pos.append(tmp_pos)
        
    future_gt_pos = [cur_eef_pos]
    tmp_pos = cur_eef_pos.copy()
    for act in gt_delta_translations:
        tmp_pos = tmp_pos + act
        future_gt_pos.append(tmp_pos)
    
    future_pred_pos = np.array(future_pred_pos)
    future_gt_pos = np.array(future_gt_pos)
    diff = np.mean(np.linalg.norm(future_pred_pos - future_gt_pos, axis=1))
    print(f"timestep {t}, mean action magnitude: {mean_delta_translation}, pred-vs-gt position diff: {diff}")
        

    from matplotlib import pyplot as plt
    fig  = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pcd_world[:, 0], pcd_world[:, 1], pcd_world[:, 2], c='gray', s=1)
    ax.scatter(goal_gripper_pcd_np[:, 0], goal_gripper_pcd_np[:, 1], goal_gripper_pcd_np[:, 2], c='red', s=50)
    ax.scatter(gripper_pcd_np[:, 0], gripper_pcd_np[:, 1], gripper_pcd_np[:, 2], c='green', s=50)
    ax.plot(
        [x[0] for x in future_pred_pos],
        [x[1] for x in future_pred_pos],
        [x[2] for x in future_pred_pos],
        color='blue',
    )
    ax.plot(
        [x[0] for x in future_gt_pos],
        [x[1] for x in future_gt_pos],
        [x[2] for x in future_gt_pos],
        color='green',
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

        
    fig.canvas.draw()

    # Convert canvas to an RGB image
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape((height, width, 3))
    images.append(image)
    
    plt.show()
    plt.close("all")


fig, axes = plt.subplots(1, 1, figsize=(8, 8))
axes[0].plot(range(len(mean_action_magnitude)), mean_action_magnitude)
axes[0].set_title('Mean Action Magnitude Over Time')
plt.show()

from cem_policy.utils import save_numpy_as_gif
save_numpy_as_gif(np.array(images), os.path.join(os.environ['PROJECT_DIR'], 
                                f"data/debug/aloha/per-step-low-level-policy-prediction_traj_{traj_idx}_{test_t}_{suffix}.mp4"), fps=6)

    

    

    
    
    
