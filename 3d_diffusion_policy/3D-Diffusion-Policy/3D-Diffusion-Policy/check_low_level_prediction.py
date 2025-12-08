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

def low_level_policy_infer(predicted_goal, obj_pcd, eef_pos, eef_quat, gripper_q, policy):
    # TODO: get the 6d orientation from the eef_quat

    import numpy as np
    num_envs = eef_pos.shape[0]

    eef_quat_np = eef_quat.cpu().numpy()
    eef_pos_np = eef_pos.cpu().numpy()
    gripper_q_np = gripper_q.cpu().numpy()

    eef_quat_R = R.from_quat(eef_quat_np)  
    eef_orient_matrix = eef_quat_R.as_matrix()

    rotation_6ds = []
    for i in range(num_envs):
        rotation_6d = rotation_transfer_matrix_to_6D(eef_orient_matrix[i])
        rotation_6ds.append(rotation_6d)
    rotation_6ds = torch.from_numpy(np.array(rotation_6ds)).float().to(obj_pcd.device)

    gripper_points = np.zeros((num_envs, 4, 3))
    for i in range(num_envs):
        gripper_points[i] = get_4_points_from_gripper_pos_orient(eef_pos_np[i], eef_orient_matrix[i], gripper_q_np[i])
        
    gripper_points = torch.from_numpy(gripper_points).float().to(obj_pcd.device)

    agent_pos = torch.cat([eef_pos, rotation_6ds, gripper_q], dim=1) # B, 6 + 3 + 1
    
    gripper_pcd = gripper_points
    object_pcd = obj_pcd.view(num_envs, -1, 3) # B, 4500, 3
    # distance = scipy.spatial.distance.cdist(gripper_pcd, object_pcd)
    distance = torch.cdist(gripper_pcd, object_pcd) # B, 4, N
    # min_distance_obj_idx = np.argmin(distance, axis=1)
    min_distance_obj_idx = torch.argmin(distance, dim=2) # B, 4
    # closest_point = object_pcd[min_distance_obj_idx]
    closest_point = torch.gather(object_pcd, 1, min_distance_obj_idx.unsqueeze(-1).expand(-1, -1, 3)) # B, 4, 3
    displacement = closest_point - gripper_pcd


    # TODO: repeat this to the horizon length
    input_dict = {
        "point_cloud": object_pcd.unsqueeze(1).repeat(1, 2, 1, 1),
        "agent_pos": agent_pos.unsqueeze(1).repeat(1, 2, 1),
        'gripper_pcd': gripper_points.unsqueeze(1).repeat(1, 2, 1, 1),
        'goal_gripper_pcd': predicted_goal.unsqueeze(1).repeat(1, 2, 1, 1),
        "displacement_gripper_to_object": displacement.unsqueeze(1).repeat(1, 2, 1, 1),
    }

    cat_idx = 0
    batched_action = policy.predict_action(input_dict, torch.tensor([cat_idx]).to(policy.device))

    return batched_action['action'] # B, T, 10



# ### load a low-level policy
# exp_dir = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1204_finetune_ours_sriram_plate/2025.12.04/22.14.09_train_dp3_robogen_open_door/"
# checkpoint_name = 'epoch-80.ckpt'
exp_dir = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data/ckpts/1020_grasp_lift_closed_goal_full"
checkpoint_name = 'epoch-92.ckpt'
# exp_dir = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1204_finetune_ours_sriram_plate_using_open_lang_embedding/2025.12.05/13.14.16_train_dp3_robogen_open_door/"
# checkpoint_name = 'epoch-80.ckpt'
# exp_dir = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/1204_finetune_ours_sriram_plate_using_open_lang_embedding_combine_2_step/2025.12.05/15.45.22_train_dp3_robogen_open_door"
# checkpoint_name = 'epoch-80.ckpt'
suffix = "pretrained"

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

# cur_eef_pos = np.array([0.4, 0, 0.6])
# goal_eef_pos = np.array([0.7, 0.2, 0.3])
# cur_eef_orient_matrix = np.eye(3)
# goal_eef_orient_matrix = np.eye(3)
# cur_eef_quat = np.array([0, 0, 0, 1])  # Identity quaternion
test_t = 0
traj_idx = 0
data_path = f"/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data/aloha/plate/traj_{str(traj_idx).zfill(4)}/{test_t}.npz"
data = np.load(data_path, allow_pickle=True)

obj_pcd_np = data['point_cloud'].reshape(-1, 3)
gripper_pcd_np = data['gripper_pcd'].reshape(-1, 3)
goal_gripper_pcd_np = data['goal_gripper_pcd'].reshape(-1, 3)
agent_pos = data['state'].reshape(1, 10)  # (T, 10)

goal_4_points = goal_gripper_pcd_np
cur_4_points = gripper_pcd_np

cur_eef_pos = agent_pos[0, :3]

timesteps = 30
eef_positions = [cur_eef_pos]
eef_orientations = [rotation_transfer_6D_to_matrix(agent_pos[0, 3:9])]
cur_eef_quat = R.from_matrix(eef_orientations[0]).as_quat()
cur_eef_orient_matrix = eef_orientations[0]
eef_four_points = [cur_4_points]
cur_q = agent_pos[0, 9]

scene_pcd = obj_pcd_np
pcd_world = scene_pcd
scene_pcd = torch.from_numpy(scene_pcd).float().to('cuda') # 4500 * 3
goal_4_points_tensor = torch.from_numpy(goal_4_points).float().to('cuda').reshape(1, 4, 3)  # 1, 4, 3
distances = [np.linalg.norm(goal_4_points - cur_4_points)]
images = []
distance_to_goal = []
mean_action_magnitude = []
gripper_q = []
for t in range(timesteps):
    print("Step {}: distance to goal: {}".format(t, distances[-1]))
    
    ### get the action from the low level policy
    # predicted_goal, obj_pcd, eef_pos, eef_quat, gripper_q, policy
    cur_eef_pos_tensor = torch.from_numpy(cur_eef_pos).float().to('cuda').reshape(1, 3)
    cur_eef_quat_tensor = torch.from_numpy(cur_eef_quat).float().to('cuda').reshape(1, 4)
    cur_q_tensor = torch.tensor(cur_q).float().to('cuda').reshape(1, 1)
    actions = low_level_policy_infer(
        goal_4_points_tensor,
        scene_pcd,
        cur_eef_pos_tensor, 
        cur_eef_quat_tensor,
        cur_q_tensor, 
        policy
    )
    
    actions = actions.detach().cpu()  # B, T, 10
    
    ### update the eef pos and orient
    actions = actions[0]
    mean_delta_translation = torch.mean(torch.norm(actions[:, :3], dim=1)).item()
    mean_action_magnitude.append(mean_delta_translation)
    gripper_q.append(cur_q)
    distance_to_goal.append(distances[-1])
    for action in actions:
        delta_translation = action[:3]
        cur_eef_pos = cur_eef_pos + delta_translation.cpu().numpy()
        eef_positions.append(cur_eef_pos)
        
        delta_rotation = action[3:9].cpu().numpy()
        delta_rotate_matrix = rotation_transfer_6D_to_matrix(delta_rotation)
        after_rotate_matrix = cur_eef_orient_matrix @ delta_rotate_matrix
        
        eef_orientations.append(after_rotate_matrix)
        cur_eef_orient_matrix = after_rotate_matrix
        cur_eef_quat = R.from_matrix(cur_eef_orient_matrix).as_quat()
        
        cur_4_points = get_4_points_from_gripper_pos_orient(cur_eef_pos, cur_eef_orient_matrix, cur_q)
        eef_four_points.append(cur_4_points)
        
        delta_q = action[9].cpu().numpy()
        cur_q = cur_q + delta_q
        cur_q = np.clip(cur_q, 0.0, 0.04)
        # print("current gripper joint angle: ", cur_q)
        
        distances.append(np.linalg.norm(goal_4_points - cur_4_points))

        from matplotlib import pyplot as plt
        fig  = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pcd_world[:, 0], pcd_world[:, 1], pcd_world[:, 2], c='gray', s=1)
        # ax.scatter([0], [0], [0], c='blue', s=50)
        ax.scatter(goal_4_points[:, 0], goal_4_points[:, 1], goal_4_points[:, 2], c='red', s=50)
        ax.scatter(cur_4_points[:, 0], cur_4_points[:, 1], cur_4_points[:, 2], c='green', s=50)
        ax.plot(
            [x[0] for x in eef_positions],
            [x[1] for x in eef_positions],
            [x[2] for x in eef_positions],
            color='blue',
        )

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('distance to goal: {}'.format(distances[-1]))

        # fig.tight_layout()
            # plt.show()
            
        fig.canvas.draw()

        # Convert canvas to an RGB image
        width, height = fig.canvas.get_width_height()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape((height, width, 3))
        images.append(image)
        
        plt.close("all")
        # plt.show()


fig, axes = plt.subplots(3, 1, figsize=(8, 12))
axes[0].plot(range(len(distance_to_goal)), distance_to_goal)
axes[0].set_title('Distance to Goal Over Time')
axes[1].plot(range(len(mean_action_magnitude)), mean_action_magnitude)
axes[1].set_title('Mean Action Magnitude Over Time')
axes[2].plot(range(len(gripper_q)), gripper_q)
axes[2].set_title('Gripper Joint Angle Over Time')
# plt.show()
save_fig_path = os.path.join(os.environ['PROJECT_DIR'],
                                f"data/debug/aloha/low-level-policy-prediction_metrics_traj_{traj_idx}_{test_t}_{suffix}.png")
fig.tight_layout()
fig.savefig(save_fig_path)

from cem_policy.utils import save_numpy_as_gif
save_numpy_as_gif(np.array(images), os.path.join(os.environ['PROJECT_DIR'], 
                                f"data/debug/aloha/low-level-policy-prediction_traj_{traj_idx}_{test_t}_{suffix}.mp4"), fps=6)

    

    

    
    
    
