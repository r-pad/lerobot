import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from train_ddp import TrainDP3Workspace
from diffusion_policy_3d.common.pytorch_util import dict_apply
from manipulation.utils import build_up_env, save_numpy_as_gif, save_env, get_pixel_location
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
import argparse
from typing import List, Optional
from collections import deque
from matplotlib import pyplot as plt
import cv2

def construct_env(cfg, config_file, solution_path, task_name, init_state_file, obj_translation, real_world_camera=False, noise_real_world_pcd=False,
                  randomize_camera=False):
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
                                                real_world_camera=real_world_camera,
                                                noise_real_world_pcd=noise_real_world_pcd,
                                                )
        
    if randomize_camera:
        pointcloud_env.reset_random_cameras()
        
    env = MultiStepWrapper(pointcloud_env, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps, 
                        max_episode_steps=600, reward_agg_method='sum')
    
    return env

def parallel_eval(args):
    config_path, init_state, action, cfg, idx = args 
    config_file = config_path
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    solution_path = [x['solution_path'] for x in config if "solution_path" in x][0]
    all_substeps_path = os.path.join(os.environ['PROJECT_DIR'], solution_path, "substeps.txt")
    with open(all_substeps_path, "r") as f:
        substeps = f.readlines()
        first_step = substeps[0].lstrip().rstrip()
        task_name = first_step.replace(" ", "_")
    
    env, _ = build_up_env(
            config_path,
            solution_path,
            task_name,
            None,
            render=False, 
            randomize=False,
            obj_id=0,
            horizon=600,
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
                                                  only_object=cfg.task.env_runner.only_object,
                                                  )
        
    env = MultiStepWrapper(pointcloud_env, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps, 
                        max_episode_steps=600, reward_agg_method='sum')
    
    env.reset(reset_state=init_state)
    obs, reward, done, info = env.step(action)
    rgb = env.env.render()
    state = save_env(env.env._env)
        
    pointcloud_env._env.close()
    return obs, rgb, info, state, idx

def parallel_reset(args):
    config_path, init_state_file, cfg, idx = args 
    config_file = config_path
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    solution_path = [x['solution_path'] for x in config if "solution_path" in x][0]
    all_substeps_path = os.path.join(os.environ['PROJECT_DIR'], solution_path, "substeps.txt")
    with open(all_substeps_path, "r") as f:
        substeps = f.readlines()
        first_step = substeps[0].lstrip().rstrip()
        task_name = first_step.replace(" ", "_")
    
    env, _ = build_up_env(
            config_path,
            solution_path,
            task_name,
            init_state_file,
            render=False, 
            # render=True, 
            randomize=False,
            obj_id=0,
            horizon=600,
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
                                                  only_object=cfg.task.env_runner.only_object,
                                                  )
        
    env = MultiStepWrapper(pointcloud_env, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps, 
                        max_episode_steps=600, reward_agg_method='sum')
    
    obs = env.reset()
    # import pdb; pdb.set_trace()
    state = save_env(env.env._env)
    rgb = env.env.render()
    info = env.env._env._get_info()
    env.env._env.close()
    return obs, state, rgb, info, idx

def parallel_save_gif(args):
    rgbs, save_path = args
    
    save_numpy_as_gif(np.array(rgbs), save_path)

def wrap_obs(list_of_obs):
    parallel_input_dict = {}
    # parallel_input_dict['point_cloud'] = np.concatenate([x['point_cloud'][None, ...] for x in list_of_obs], axis=0)
    # parallel_input_dict['agent_pos'] = np.concatenate([x['agent_pos'][None, ...] for x in list_of_obs], axis=0)
    # parallel_input_dict['feature_map'] = np.concatenate([x['feature_map'][None, ...] for x in list_of_obs], axis=0)
    # parallel_input_dict['gripper_pcd'] = np.concatenate([x['gripper_pcd'][None, ...] for x in list_of_obs], axis=0)
    # parallel_input_dict['pcd_mask'] = np.concatenate([x['pcd_mask'][None, ...] for x in list_of_obs], axis=0)
    # # TODO: add goal key
    # if 'goal_gripper_pcd' in list_of_obs[0]:
    #     parallel_input_dict['goal_gripper_pcd'] = np.concatenate([x['goal_gripper_pcd'][None, ...] for x in list_of_obs], axis=0)
    for key in list_of_obs[0]:
        parallel_input_dict[key] = np.concatenate([x[key][None, ...] for x in list_of_obs], axis=0)
    
    parallel_input_dict = dict_apply(parallel_input_dict, lambda x: torch.from_numpy(x).to('cuda'))
    return parallel_input_dict


            
def run_eval_non_parallel(cfg, policy, goal_prediction_model, num_worker, save_path, exp_beg_idx=0,
                          exp_end_idx=1000, pool=None, horizon=150,  exp_beg_ratio=None, exp_end_ratio=None,
                          dataset_index=None, calculate_distance_from_gt=False, output_obj_pcd_only=False, obj_translation: Optional[list]= None,
                          update_goal_freq=1, real_world_camera=False, noise_real_world_pcd=False,
                          randomize_camera=False):
    
    for dataset_idx, (experiment_folder, experiment_name, demo_experiment_path) in enumerate(zip(cfg.task.env_runner.experiment_folder, cfg.task.env_runner.experiment_name, cfg.task.env_runner.demo_experiment_path)):
        
        if dataset_index is not None:
            dataset_idx = dataset_index

        if calculate_distance_from_gt:
            all_obj_distances = []

        # if dataset_idx == 0:
        #     continue

        after_reaching_init_state_files = []
        init_state_files = []
        config_files = []
        experiment_folder = "{}/{}".format(os.environ['PROJECT_DIR'], experiment_folder)
        experiment_name = experiment_name
        experiment_path = os.path.join(experiment_folder, "experiment", experiment_name)
        all_experiments = os.listdir(experiment_path)
        all_experiments = sorted(all_experiments)
        
        if demo_experiment_path is not None:
            # demo_experiment_path = demo_experiment_path[demo_experiment_path.find("RoboGen_sim2real/") + len("RoboGen_sim2real/"):]
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
                reaching_phase = stage_lengths['reach_handle'] + stage_lengths['reach_to_contact'] + stage_lengths['close_gripper']
                
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
        after_reaching_init_state_files = after_reaching_init_state_files[exp_beg_idx:exp_end_idx]
        expert_opened_angles = expert_opened_angles[exp_beg_idx:exp_end_idx]
        all_distances = []
        all_grasp_distances = []

        for exp_idx, (config_file, init_state_file, opening_begining_state_file) in enumerate(zip(config_files, init_state_files, after_reaching_init_state_files)):
            
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            solution_path = [x['solution_path'] for x in config if "solution_path" in x][0]
            all_substeps_path = os.path.join(os.environ['PROJECT_DIR'], solution_path, "substeps.txt")
            with open(all_substeps_path, "r") as f:
                substeps = f.readlines()
                first_step = substeps[0].lstrip().rstrip()
                task_name = first_step.replace(" ", "_")
            
            save_names = ['init', 'open']
            for state_file, name in zip([init_state_file, opening_begining_state_file], save_names):
                env = construct_env(cfg, config_file, solution_path, task_name, state_file, obj_translation, real_world_camera, noise_real_world_pcd, 
                                    randomize_camera)
                
                obs = env.reset()
                rgb = env.env.render()
                info = env.env._env._get_info()

                initial_info = info
                all_rgbs = [rgb]
                goal_stage = 'first'
                first_step_outputs = None
                gripper_close_accumulation_buffer = deque(maxlen=5)
                last_goal = None
                
                parallel_input_dict = obs
                parallel_input_dict = dict_apply(parallel_input_dict, lambda x: torch.from_numpy(x).to('cuda'))
                
                
                for key in obs:
                    parallel_input_dict[key] = parallel_input_dict[key].unsqueeze(0)
                
                with torch.no_grad():
                    pointcloud = parallel_input_dict['point_cloud'][:, -1, :, :]
                    gripper_pcd = parallel_input_dict['gripper_pcd'][:, -1, :]
                    if not args.predict_two_goals:
                        inputs = torch.cat([pointcloud, gripper_pcd], dim=1)
                    else:
                        inputs = pointcloud
                        
                    if args.add_one_hot_encoding:
                        # for pointcloud, we add (1, 0)
                        # for gripper_pcd, we add (0, 1)
                        pointcloud_one_hot = torch.zeros(pointcloud.shape[0], pointcloud.shape[1], 2).float().to(pointcloud.device)
                        pointcloud_one_hot[:, :, 0] = 1
                        pointcloud_ = torch.cat([pointcloud, pointcloud_one_hot], dim=2)
                        gripper_pcd_one_hot = torch.zeros(gripper_pcd.shape[0], gripper_pcd.shape[1], 2).float().to(pointcloud.device)
                        gripper_pcd_one_hot[:, :, 1] = 1
                        gripper_pcd_ = torch.cat([gripper_pcd, gripper_pcd_one_hot], dim=2)
                        inputs = torch.cat([pointcloud_, gripper_pcd_], dim=1) # B, N+4, 5
                    
                    fixed_variance = args.fixed_variance
                    inputs = inputs.to('cuda')
                    inputs_ = inputs.permute(0, 2, 1)
                    outputs = goal_prediction_model(inputs_)
                    weights = outputs[:, :, -1] # B, N
                    outputs = outputs[:, :, :-1] # B, N, 12
                    if output_obj_pcd_only:
                        # cprint("using only obj pcd output!", "red")
                        weights = weights[:, :-4]
                        outputs = outputs[:, :-4, :]
                        inputs = inputs[:, :-4, :]

                    B, N, _ = outputs.shape
                    if not args.predict_two_goals:
                        outputs = outputs.view(B, N, 4, 3)
                    else:
                        outputs = outputs.view(B, N, 8, 3)
                        if first_step_outputs is None:
                            first_step_outputs = deepcopy(outputs)
                            
                        if goal_stage == 'first':
                            outputs = outputs[:, :, :4, :]
                        elif goal_stage == 'second':
                            outputs = outputs[:, :, 4:, :]
                    
                    ### sample an displacement according to the weight
                    # import pdb; pdb.set_trace()
                    probabilities = weights  # Must sum to 1
                    probabilities = torch.nn.functional.softmax(weights, dim=1)

                    # Sample one index based on the probabilities
                    sampled_outputs = []
                    for sample_idx in range(args.sample_num):
                        sampled_index = torch.multinomial(probabilities, num_samples=1)
                        sampled_index = sampled_index.item()
                        displacement_mean = outputs[:, sampled_index, :, :] # B, 4, 3
                        input_point_pos = inputs[:, sampled_index, :] # B, 3
                        prediction = input_point_pos.unsqueeze(1) + displacement_mean # B, 4, 3
                        outputs_sampled = prediction.unsqueeze(1) # B, history=1, 4, 3
                        sampled_outputs.append(outputs_sampled)
                    
                last_goal = outputs

                env.env.goal_gripper_pcd = None
                image = env.env.render()
                for outputs in sampled_outputs:
                    np_predicted_goal = outputs.detach().to('cpu').numpy()
                    goal_gripper_pcd = np_predicted_goal.squeeze(0)[0].reshape(4, 3)
                    image = np.array(image)
                    for point in goal_gripper_pcd:
                        pixel_x, pixel_y, _ = get_pixel_location(env.env._env.projection_matrix, env.env._env.view_matrix, point, 
                                                                    env.env._env.camera_width, env.env._env.camera_height)
                        color = (0, 0, 255)  # Red color in BGR
                        thickness = 2
                        radius = 5
                        image = cv2.circle(image, (pixel_x, pixel_y), radius, color, thickness)
                    
                env.env._env.close()
            
                gif_save_exp_name = experiment_folder.split("/")[-2]
                gif_save_folder = "{}/{}".format(save_path, gif_save_exp_name)
                if not os.path.exists(gif_save_folder):
                    os.makedirs(gif_save_folder, exist_ok=True)
                gif_save_path = "{}/{}_{}.png".format(gif_save_folder, exp_idx, name)
                
                plt.imsave(gif_save_path, image, dpi=300) 
            

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--low_level_exp_dir', type=str, default=None)
    parser.add_argument('--low_level_ckpt_name', type=str, default=None)
    parser.add_argument("--high_level_ckpt_name", type=str, default=None)
    parser.add_argument("--pointnet_class", type=str, default="PointNet2")
    parser.add_argument("--eval_exp_name", type=str, default=None)
    parser.add_argument("--use_predicted_goal", type=bool, default=True)
    parser.add_argument("--test_cross_category", type=bool, default=False)
    parser.add_argument("--model_invariant", type=bool, default=False)
    parser.add_argument('--predict_two_goals', action='store_true')
    parser.add_argument('--output_obj_pcd_only', action='store_true')
    parser.add_argument("--update_goal_freq", type=int, default=1)
    parser.add_argument("--noise_real_world_pcd", type=int, default=0)
    parser.add_argument("--randomize_camera", type=int, default=0)
    parser.add_argument("--real_world_camera", type=int, default=0)
    parser.add_argument('-n', '--noise', type=float, default=None, nargs=2, help='bounds for noise. e.g. `--noise -0.1 0.1')
    parser.add_argument('--keep_gripper_in_fps', type=int, default=0)
    parser.add_argument('--add_one_hot_encoding', type=int, default=0)
    parser.add_argument('--fixed_variance', type=float, default=0.05)
    parser.add_argument('--sample_num', type=int, default=10)
    args = parser.parse_args()
    
    num_worker = 30
    pool=None

    if args.low_level_exp_dir is None:
        # best 50 objects
        exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/07201526-act3d_goal_mlp-horizon-8-num_load_episodes-1000/2024.07.20/15.26.54_train_dp3_robogen_open_door"
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
    
    workspace = TrainDP3Workspace(cfg)
    checkpoint_dir = "{}/checkpoints/{}".format(exp_dir, checkpoint_name)
    workspace.load_checkpoint(path=checkpoint_dir, )

    policy = deepcopy(workspace.model)
    if workspace.cfg.training.use_ema:
        policy = deepcopy(workspace.ema_model)
    policy.eval()
    policy.reset()
    policy = policy.to('cuda')
    
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
    
    load_model_path = args.high_level_ckpt_name
        
    
    num_class = 13 if not args.predict_two_goals else 25
    input_channel = 5 if args.add_one_hot_encoding else 3

    if not args.model_invariant:
        from test_PointNet2.model import PointNet2_small2, PointNet2, PointNet2_super
        if args.pointnet_class == "PointNet2":
            pointnet2_model = PointNet2_small2(num_classes=num_class).to('cuda')
        elif args.pointnet_class == "PointNet2_large":
            pointnet2_model = PointNet2(num_classes=num_class).to('cuda')
        elif args.pointnet_class == "PointNet2_super":
            pointnet2_model = PointNet2_super(num_classes=num_class, keep_gripper_in_fps=args.keep_gripper_in_fps, input_channel=input_channel).to("cuda")
        
    else:
        from test_PointNet2.model_invariant import PointNet2, PointNet2_super, PointNet2_superplus
        if args.pointnet_class == 'PointNet2_large':
            pointnet2_model = PointNet2(num_classes=num_class).to('cuda')
        elif args.pointnet_class == 'PointNet2_super':
            pointnet2_model = PointNet2_super(num_classes=num_class, keep_gripper_in_fps=args.keep_gripper_in_fps, input_channel=input_channel).to("cuda")
        elif args.pointnet_class == "PointNet2_superplus":
            pointnet2_model = PointNet2_superplus(num_classes=13).to("cuda")
            
        
    pointnet2_model.load_state_dict(torch.load(load_model_path))
    pointnet2_model.eval()
    
    checkpoint_dir = "{}/checkpoints/{}".format(exp_dir, checkpoint_name)
    
    save_path = "data/{}".format(args.eval_exp_name)
    if args.noise is not None:
        save_path = "data/{}_{}_{}".format(args.eval_exp_name, args.noise[0], args.noise[1])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    checkpoint_info = {
        "low_level_policy": checkpoint_dir,
        "low_level_policy_checkpoint": checkpoint_name,
        "high_level_policy_checkpoint": args.high_level_ckpt_name,
    }
    checkpoint_info.update(args.__dict__)
    with open("{}/checkpoint_info.json".format(save_path), "w") as f:
        json.dump(checkpoint_info, f, indent=4)
    
    cfg.task.env_runner.observation_mode = "act3d_goal_displacement_gripper_to_object"
    cfg.task.dataset.observation_mode = "act3d_goal_displacement_gripper_to_object"
    run_eval_non_parallel(
            cfg, policy, pointnet2_model,
            num_worker, save_path, 
            pool=pool, 
            horizon=35,
            exp_beg_idx=0,
            exp_end_idx=25,
            obj_translation=args.noise,
            output_obj_pcd_only=args.output_obj_pcd_only,
            update_goal_freq=args.update_goal_freq,
            real_world_camera=args.real_world_camera,
            noise_real_world_pcd=args.noise_real_world_pcd,
            randomize_camera=args.randomize_camera
    )


# python eval_robogen_with_goal_PointNet.py --high_level_ckpt_name /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps/pointnet2_super_model_invariant_2024-09-30_use_75_episodes_200-obj/model_39.pth --eval_exp_name eval_yufei_weighted_displacement_pointnet_large_200_invariant_reproduce --pointnet_class PointNet2_super --model_invariant True