import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
# from train import TrainDP3Workspace
from train_ddp import TrainDP3Workspace
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
import argparse
from typing import Optional, List
from diffusion_policy_3d.common.gripper_orientation_from_4_points import get_gripper_pos_orient_from_4_points_torch

def construct_env(cfg, config_file, solution_path, task_name, init_state_file, obj_translation, 
                  real_world_camera=False, noise_real_world_pcd=False, randomize_camera=False):
    env, _ = build_up_env(
                    config_file,
                    solution_path,
                    task_name,
                    init_state_file,
                    # render=False, 
                    render=False, 
                    randomize=False,
                    obj_id=0,
                    horizon=600,
                    random_object_translation=obj_translation,
            )
            
    object_name = "StorageFurniture".lower()
    env.reset()
    pointcloud_env = RobogenPointCloudWrapper(env, object_name, in_gripper_frame=cfg.task.env_runner.in_gripper_frame, 
                                                gripper_num_points=cfg.task.env_runner.gripper_num_points, add_contact=cfg.task.env_runner.add_contact,
                                                num_points=cfg.task.env_runner.num_point_in_pc,
                                                use_joint_angle=cfg.task.env_runner.use_joint_angle, 
                                                use_segmask=cfg.task.env_runner.use_segmask,
                                                only_handle_points=cfg.task.env_runner.only_handle_points,
                                                observation_mode=cfg.task.env_runner.observation_mode,
                                                dense_pcd_for_goal=cfg.task.env_runner.dense_pcd_for_goal,
                                                real_world_camera=real_world_camera,
                                                noise_real_world_pcd=noise_real_world_pcd,
    )
    
    if randomize_camera:
        pointcloud_env.reset_random_cameras()
        
    env = MultiStepWrapper(pointcloud_env, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps, 
                        max_episode_steps=600, reward_agg_method='sum')
    
    return env
            
def run_eval_non_parallel(cfg, policy, goal_cfg, goal_policy, 
                          num_worker, save_path, exp_beg_idx=0, exp_end_idx=1000, pool=None, horizon=150,  exp_beg_ratio=None, exp_end_ratio=None,
                          dataset_index=None, calculate_distance_from_gt=False,
                          obj_translation: Optional[list]= None,
                          use_predicted_goal: bool = True,
                          update_goal_freq=1,
                          real_world_camera=False,
                          noise_real_world_pcd=False,
                          randomize_camera=False,
                          use_high_level=True,
                          heuristic_goal_switching=False,
                          pos_ori_imp = False
                          ):
    
    if calculate_distance_from_gt:
        all_obj_distances = []
        
    for dataset_idx, (experiment_folder, experiment_name, demo_experiment_path) in enumerate(zip(cfg.task.env_runner.experiment_folder, cfg.task.env_runner.experiment_name, cfg.task.env_runner.demo_experiment_path)):
        
        if dataset_index is not None:
            dataset_idx = dataset_index

        after_reaching_init_state_files = []
        init_state_files = []
        config_files = []
        experiment_folder = "{}/{}".format(os.environ['PROJECT_DIR'], experiment_folder)
        experiment_name = experiment_name
        experiment_path = os.path.join(experiment_folder, "experiment", experiment_name)
        all_experiments = os.listdir(experiment_path)
        all_experiments = sorted(all_experiments)
        
        if demo_experiment_path is not None:
            all_subfolder = os.listdir(demo_experiment_path)
            for string in ["action_dist", "demo_rgbs", "all_demo_path.txt", "meta_info.json", 'example_pointcloud']:
                if string in all_subfolder:
                    all_subfolder.remove(string)
            all_subfolder = sorted(all_subfolder)
            all_experiments = all_subfolder
            
        all_substeps_path = os.path.join(experiment_folder, "substeps.txt")
        with open(all_substeps_path, "r") as f:
            substeps = f.readlines()
            first_step = substeps[0].lstrip().rstrip()
        

        expert_opened_angles = []
        for experiment in all_experiments:
            if "meta" in experiment:
                continue
            
            first_step_folder = first_step.replace(" ", "_") + "_primitive"
            first_step_folder = os.path.join(experiment_path, experiment, first_step_folder)
            if os.path.exists(os.path.join(first_step_folder, "label.json")):
                with open(os.path.join(first_step_folder, "label.json"), 'r') as f:
                    label = json.load(f)
                if not label['good_traj']: continue
                
            first_stage_states_path = os.path.join(first_step_folder, "states")
            expert_states = os.listdir(first_stage_states_path)
            if len(expert_states) == 0:
                continue
                
            expert_opened_angle_file = os.path.join(experiment_path, experiment, first_step_folder, "opened_angle.txt")
            if os.path.exists(expert_opened_angle_file):
                with open(expert_opened_angle_file, "r") as f:
                    angles = f.readlines()
                    expert_opened_angle = float(angles[0].lstrip().rstrip())
                    max_angle = float(angles[-1].lstrip().rstrip())
                    ratio = expert_opened_angle / max_angle
                # if ratio < 0.65:
                #     continue
            expert_opened_angles.append(expert_opened_angle)
            
            first_stage_states_path = os.path.join(first_step_folder, "states")
            stage_lengths = os.path.join(first_step_folder, "stage_lengths.json")
            with open(stage_lengths, "r") as f:
                stage_lengths = json.load(f)
            
            if 'stage' in stage_lengths:
                reaching_phase = stage_lengths.get('open_gripper', 0) + stage_lengths['grasp_handle']
            else:
                reaching_phase = stage_lengths['reach_handle']
                
            after_init_state_file = os.path.join(first_stage_states_path, "state_{}.pkl".format(reaching_phase))
            after_reaching_init_state_files.append(after_init_state_file)
            init_state_file = os.path.join(first_stage_states_path, "state_0.pkl")
            init_state_files.append(init_state_file)
            config_file = os.path.join(experiment_path, experiment, "task_config.yaml")
            config_files.append(config_file)
                    
        after_reaching_init_state_files = after_reaching_init_state_files
        config_files = config_files

        opened_joint_angles = {}

        if exp_end_ratio is not None:
            exp_end_idx = int(exp_end_ratio * len(config_files))
        if exp_beg_ratio is not None:
            exp_beg_idx = int(exp_beg_ratio * len(config_files))

        config_files = config_files[exp_beg_idx:exp_end_idx]
        init_state_files = init_state_files[exp_beg_idx:exp_end_idx]
        expert_opened_angles = expert_opened_angles[exp_beg_idx:exp_end_idx]
        all_distances = []
        all_grasp_distances = []

        for exp_idx, (config_file, init_state_file) in enumerate(zip(config_files, init_state_files)):
                
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            solution_path = [x['solution_path'] for x in config if "solution_path" in x][0]
            all_substeps_path = os.path.join(os.environ['PROJECT_DIR'], solution_path, "substeps.txt")
            with open(all_substeps_path, "r") as f:
                substeps = f.readlines()
                first_step = substeps[0].lstrip().rstrip()
                task_name = first_step.replace(" ", "_")
            
            env = construct_env(cfg, config_file, solution_path, task_name, init_state_file, obj_translation, 
                                real_world_camera, noise_real_world_pcd, randomize_camera)
            
            obs = env.reset()
            rgb = env.env.render()
            info = env.env._env._get_info()

            initial_info = info
            all_rgbs = [rgb]
            closed = False
            last_goal = None 
            for t in range(1, horizon):
                parallel_input_dict = obs
                parallel_input_dict = dict_apply(parallel_input_dict, lambda x: torch.from_numpy(x).to('cuda'))
                
                for key in obs:
                    parallel_input_dict[key] = parallel_input_dict[key].unsqueeze(0)
                    
                if use_high_level and not heuristic_goal_switching:
                    if use_predicted_goal and (t == 1 or t % update_goal_freq == 0):
                        high_level_parallel_input_dict = deepcopy(parallel_input_dict)
                        if goal_cfg.n_obs_steps == 1:
                            for key in high_level_parallel_input_dict:
                                high_level_parallel_input_dict[key] = high_level_parallel_input_dict[key][:, -1, ...].unsqueeze(1) # take the most recent observation
                        
                        if pos_ori_imp:
                            #import pdb; pdb.set_trace();
                            open_close = np.array(high_level_parallel_input_dict['agent_pos'][:,:,9].cpu()).squeeze()
                            gripper_pcd_numpy = np.array(high_level_parallel_input_dict['gripper_pcd'].cpu())
                            gripper_pcd_numpy = gripper_pcd_numpy.reshape(-1,4,3)
                            gripper_pcd_10d_numpy = np.zeros((gripper_pcd_numpy.shape[0], 10))
                            for frame in range(gripper_pcd_numpy.shape[0]):
                                #import pdb; pdb.set_trace();
                                gripper_9d = get_gripper_pos_orient_from_4_points_torch(gripper_pcd_numpy[frame])
                                gripper_pcd_10d_numpy[frame] = np.append(gripper_9d, open_close[frame])
                            gripper_pcd_10d_numpy = np.expand_dims(np.expand_dims(gripper_pcd_10d_numpy, axis=0), axis = -1).astype(np.float32)
                            gripper_pcd_10d = torch.from_numpy(gripper_pcd_10d_numpy).cuda()
                            
                            #import pdb; pdb.set_trace();
                            displacement_gripper_to_object_numpy = np.array(high_level_parallel_input_dict['displacement_gripper_to_object'].cpu()).reshape(-1,4,3)
                            displacement_gripper_to_object_numpy_10d = np.zeros((gripper_pcd_numpy.shape[0], 3))
                            for frame in range(gripper_pcd_numpy.shape[0]):
                                #import pdb; pdb.set_trace();
                                #print("000000000000000000", sample['displacement_gripper_to_object'].shape)
                                object_point = np.array(high_level_parallel_input_dict['gripper_pcd'].cpu()).reshape(-1,4,3)[frame,0] + displacement_gripper_to_object_numpy[frame,0]
                                #print("111111111111", object_point.shape)
                                displacement_gripper_to_object_numpy_10d[frame] = (object_point - gripper_pcd_10d_numpy[0, frame, :3].flatten())
                                #print("222222222222", gripper_pcd.shape, gripper_pcd[i, :3, :].flatten().shape, (object_point - gripper_pcd[i, :3, :].flatten()).shape) 
                            #displacement_gripper_to_object = np.array(displacement_gripper_to_object).reshape(self.horizon, -1, 3)
                            displacement_gripper_to_object_10d = torch.from_numpy(np.expand_dims(np.expand_dims(displacement_gripper_to_object_numpy_10d, axis=0), axis=2).astype(np.float32)).cuda()
                            high_level_parallel_input_dict['gripper_pcd'] = gripper_pcd_10d
                            high_level_parallel_input_dict['displacement_gripper_to_object'] = displacement_gripper_to_object_10d
                            #import pdb; pdb.set_trace();



                        with torch.no_grad():
                            predicted_goal = goal_policy.predict_action(high_level_parallel_input_dict)
                        #import pdb; pdb.set_trace();
                        if goal_cfg.policy.prediction_target == 'goal_gripper_pcd':
                            if pos_ori_imp:
                                parallel_input_dict['goal_gripper_10d_repr'] = predicted_goal['action'][:, :2, :].view(1, 2, 10)
                            else:
                                parallel_input_dict['goal_gripper_pcd'] = predicted_goal['action'][:, :2, :].view(1, 2, 4, 3)
                        elif goal_cfg.policy.prediction_target == 'delta_to_goal_gripper':
                            current_gripper_pcd = parallel_input_dict['gripper_pcd'].detach()
                            delta_pcd = predicted_goal['action'][:, :2, :]
                            parallel_input_dict['goal_gripper_pcd'] = current_gripper_pcd + delta_pcd.view(1, 2, 4, 3).detach()
                        last_goal = parallel_input_dict['goal_gripper_pcd']
                    else:
                        parallel_input_dict['goal_gripper_pcd'] = last_goal
                #import pdb; pdb.set_trace();        
                if cfg.task.env_runner.dense_pcd_for_goal:
                    parallel_input_dict['point_cloud'] = parallel_input_dict['dense_point_cloud']
                    
                with torch.no_grad():
                    batched_action = policy.predict_action(parallel_input_dict)
                    
                np_batched_action = dict_apply(batched_action, lambda x: x.detach().to('cpu').numpy())
                np_batched_action = np_batched_action['action']

                obs, reward, done, info = env.step(np_batched_action.squeeze(0))
                if calculate_distance_from_gt:
                    predicted_goal = parallel_input_dict['goal_gripper_pcd'].detach().cpu().numpy().squeeze(0)[0].reshape(4, 3)
                    gt_goal = env.env.goal_gripper_pcd
                    distance = np.linalg.norm(predicted_goal - gt_goal, axis=1).mean()
                    all_distances.append(distance)
                    grasp_distance = np.linalg.norm(predicted_goal[-1] - gt_goal[-1])
                    all_grasp_distances.append(grasp_distance)
                    print("grasp distance: ", grasp_distance)
                    print("distance: ", distance)
                    rgb = env.env.render()
                    for point in predicted_goal:
                        pixel_x, pixel_y, _ = get_pixel_location(env.env._env.projection_matrix, env.env._env.view_matrix, point, env.env._env.camera_width, env.env._env.camera_height)
                        color = (255, 0, 0)
                        thickness = 2
                        radius = 3
                        image = cv2.circle(rgb, (pixel_x, pixel_y), radius, color, thickness)
                        # save image
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        # cv2.imwrite(f'data/debug/{dataset_idx}_{exp_idx}.png', image)
                    break

                if use_high_level:
                    if pos_ori_imp:
                        from diffusion_policy_3d.common.gripper_orientation_from_4_points import get_points_from_pos_rotation_matrix
                        positions_of_4_points = get_points_from_pos_rotation_matrix(parallel_input_dict['goal_gripper_10d_repr'].cpu().numpy()[0,0,:3], parallel_input_dict['goal_gripper_10d_repr'][0,0,:6])
                        env.env.goal_gripper_pcd = positions_of_4_points.reshape(4, 3)
                    else:
                        env.env.goal_gripper_pcd = parallel_input_dict['goal_gripper_pcd'].detach().cpu().numpy().squeeze(0)[0].reshape(4, 3)
            
                if not use_high_level and not heuristic_goal_switching:
                    env.env.goal_gripper_pcd = None
                    
                rgb = env.env.render()
                all_rgbs.append(rgb)
            
            env.env._env.close()

            if calculate_distance_from_gt:
                continue
            
            opened_joint_angles[config_file] = \
            {
                "final_door_joint_angle": float(info['opened_joint_angle'][-1]), 
                "expert_door_joint_angle": expert_opened_angles[exp_idx], 
                "initial_joint_angle": float(info['initial_joint_angle'][-1]),
                "ik_failure": float(info['ik_failure'][-1]),
                'grasped_handle': float(info['grasped_handle'][-1]),
                "exp_idx": exp_idx, 
            }
                    
            with open("{}/opened_joint_angles_{}.json".format(save_path, dataset_idx), "w") as f:
                json.dump(opened_joint_angles, f, indent=4)
            
            gif_save_exp_name = experiment_folder.split("/")[-2]
            gif_save_folder = "{}/{}".format(save_path, gif_save_exp_name)
            if not os.path.exists(gif_save_folder):
                os.makedirs(gif_save_folder, exist_ok=True)
            gif_save_path = "{}/{}_{}.gif".format(gif_save_folder, exp_idx, 
                    float(info["improved_joint_angle"][-1]))
            
            save_numpy_as_gif(np.array(all_rgbs), gif_save_path)

        if calculate_distance_from_gt:
            print("average distance: {}".format(np.mean(all_distances)))
            print("average grasp distance: {}".format(np.mean(all_grasp_distances)))
            all_obj_distances.append(np.mean(all_distances))

    if calculate_distance_from_gt:
        print("average distance over all objects: {}".format(np.mean(all_obj_distances)))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--low_level_exp_dir', type=str, default=None)
    parser.add_argument('--low_level_ckpt_name', type=str, default='latest.ckpt')
    parser.add_argument("--high_level_exp_dir", type=str, default=None)
    parser.add_argument("--high_level_ckpt_name", type=str, default=None)
    parser.add_argument("--eval_exp_name", type=str, default=None)
    parser.add_argument("--use_predicted_goal", type=bool, default=True)
    parser.add_argument("--test_cross_category", type=bool, default=False)
    parser.add_argument("--calculate_distance_from_gt", type=bool, default=False)
    parser.add_argument("--update_goal_freq", type=int, default=1)
    parser.add_argument("--noise_real_world_pcd", type=int, default=0)
    parser.add_argument("--real_world_camera", type=int, default=0)
    parser.add_argument("--randomize_camera", type=int, default=0)
    parser.add_argument("--use_high_level", type=int, default=1)
    parser.add_argument("--heuristic_goal_switching", type=int, default=0)
    parser.add_argument('-n', '--noise', type=float, default=None, nargs=2, help='bounds for noise. e.g. `--noise -0.1 0.1')
    parser.add_argument('--pos_ori_imp', action='store_true', help='Set the flag for 10D representation Training')

    args = parser.parse_args()
    
    num_worker = 30
    pool=None
    
    ### This is the default best low level 50 objects best policy
    if args.low_level_exp_dir is None:
        exp_dir =  "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/07201526-act3d_goal_mlp-horizon-8-num_load_episodes-1000/2024.07.20/15.26.54_train_dp3_robogen_open_door/"
        checkpoint_name = 'latest.ckpt'
    else:
        exp_dir = args.low_level_exp_dir
        checkpoint_name = args.low_level_ckpt_name

    with hydra.initialize(config_path='diffusion_policy_3d/config'):  # same config_path as used by @hydra.main
        recomposed_config = hydra.compose(
            config_name="dp3.yaml",  # same config_name as used by @hydra.main
            overrides=OmegaConf.load("{}/.hydra/overrides.yaml".format(exp_dir)),
        )
    cfg = recomposed_config
    cfg.use_pretrained_high_level_policy_as_low_level_input = False # because that is only for training

    workspace = TrainDP3Workspace(cfg)
    checkpoint_dir = "{}/checkpoints/{}".format(exp_dir, checkpoint_name)
    workspace.load_checkpoint(path=checkpoint_dir, exclude_keys=['pretrained_goal_model', 'amp_scaler'])

    policy = deepcopy(workspace.model)
    if workspace.cfg.training.use_ema:
        policy = deepcopy(workspace.ema_model)
    policy.eval()
    policy.reset()
    policy = policy.to('cuda')

    ### these are the test objects
    cfg.task.env_runner.experiment_name = ['0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first' for _ in range(10)]
    cfg.task.env_runner.experiment_folder = [
        'data/diverse_objects/open_the_door_40147/task_open_the_door_of_the_storagefurniture_by_its_handle',
        'data/diverse_objects/open_the_door_44817/task_open_the_door_of_the_storagefurniture_by_its_handle',
        'data/diverse_objects/open_the_door_44962/task_open_the_door_of_the_storagefurniture_by_its_handle',
        'data/diverse_objects/open_the_door_45132/task_open_the_door_of_the_storagefurniture_by_its_handle',
        'data/diverse_objects/open_the_door_45219/task_open_the_door_of_the_storagefurniture_by_its_handle',
        'data/diverse_objects/open_the_door_45243/task_open_the_door_of_the_storagefurniture_by_its_handle',
        'data/diverse_objects/open_the_door_45332/task_open_the_door_of_the_storagefurniture_by_its_handle',
        'data/diverse_objects/open_the_door_45378/task_open_the_door_of_the_storagefurniture_by_its_handle',
        'data/diverse_objects/open_the_door_45384/task_open_the_door_of_the_storagefurniture_by_its_handle',
        'data/diverse_objects/open_the_door_45463/task_open_the_door_of_the_storagefurniture_by_its_handle'
        
    ]
    cfg.task.env_runner.demo_experiment_path = [None for _ in range(10)]
    # cfg.task.env_runner.experiment_name += ['0822-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first' for _ in range(6)]
    # cfg.task.env_runner.experiment_folder += [
    #     "data/diverse_objects_other/open_the_door_7167/task_open_the_door_of_the_storagefurniture_by_its_handle",
    #     "data/diverse_objects_other/open_the_door_7263/task_open_the_door_of_the_storagefurniture_by_its_handle",
    #     "data/diverse_objects_other/open_the_door_7290/task_open_the_door_of_the_storagefurniture_by_its_handle",
    #     "data/diverse_objects_other/open_the_door_7310/task_open_the_door_of_the_storagefurniture_by_its_handle",
    #     "data/diverse_objects_other/open_the_door_12092/task_open_the_door_of_the_storagefurniture_by_its_handle",
    #     "data/diverse_objects_other/open_the_door_12606/task_open_the_door_of_the_storagefurniture_by_its_handle",
    # ]
    # cfg.task.env_runner.demo_experiment_path += [None for _ in range(6)]

    ### load the high-level policy
    if args.use_high_level:
        goal_exp_dir = args.high_level_exp_dir
        if args.high_level_exp_dir is None:
            goal_exp_dir = '/home/mino/Software/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0807-200-obj-pred-goal-gripper-PointNet2-backbone-UNet-diffusion-ep-75-epsilon/2024.08.07/14.03.40_train_dp3_robogen_open_door'
        goal_checkpoint_name = args.high_level_ckpt_name
        if args.high_level_ckpt_name is None:
            goal_checkpoint_name = 'epoch-30.ckpt'
        
        with hydra.initialize(config_path='diffusion_policy_3d/config'):  # same config_path as used by @hydra.main
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
    else:
        goal_policy = None
        goal_cfg = None
        goal_exp_dir = None
        goal_checkpoint_name = None
        
    
    checkpoint_dir = "{}/checkpoints/{}".format(exp_dir, checkpoint_name)
    checkpoint_name_start_idx = checkpoint_dir.find("3D-Diffusion-Policy/data/")  + len("3D-Diffusion-Policy/data/")
    
    save_path = "data/{}".format(args.eval_exp_name)
    if args.noise is not None:
        save_path = "data/{}_{}_{}".format(args.eval_exp_name, args.noise[0], args.noise[1])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    checkpoint_info = {
        "low_level_policy": checkpoint_dir,
        "low_level_policy_checkpoint": checkpoint_name,
        "high_level_policy": goal_exp_dir,
        "high_level_policy_checkpoint": goal_checkpoint_name,
    }
    checkpoint_info.update(args.__dict__)
    with open("{}/checkpoint_info.json".format(save_path), "w") as f:
        json.dump(checkpoint_info, f, indent=4)
    
    ### run evaluation
    cfg.task.env_runner.observation_mode = "act3d_goal_displacement_gripper_to_object"
    cfg.task.dataset.observation_mode = "act3d_goal_displacement_gripper_to_object"
    run_eval_non_parallel(
            cfg, policy, goal_cfg, goal_policy, 
            num_worker, save_path, 
            pool=pool, 
            horizon=35,
            exp_beg_idx=0,
            exp_end_idx=25,
            obj_translation=args.noise,
            use_predicted_goal=args.use_predicted_goal,
            calculate_distance_from_gt=args.calculate_distance_from_gt,
            update_goal_freq=args.update_goal_freq,
            real_world_camera=args.real_world_camera,
            noise_real_world_pcd=args.noise_real_world_pcd,
            randomize_camera=args.randomize_camera,
            use_high_level=args.use_high_level,
            heuristic_goal_switching=args.heuristic_goal_switching,
            pos_ori_imp = args.pos_ori_imp
    )
