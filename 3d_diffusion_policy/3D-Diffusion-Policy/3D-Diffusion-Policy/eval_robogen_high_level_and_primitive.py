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
from diffuser_actor_3d.robogen_utils import get_gripper_pos_orient_from_4_points
import matplotlib.pyplot as plt


def construct_env(cfg, config_file, solution_path, task_name, init_state_file):
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
                                                )
        
    env = MultiStepWrapper(pointcloud_env, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps, 
                        max_episode_steps=600, reward_agg_method='sum')
        
    return env

def run_eval_non_parallel(cfg, policy, goal_cfg, goal_policy, 
                          num_worker, save_path, exp_beg_idx=0, exp_end_idx=1000, pool=None, horizon=150,  exp_beg_ratio=None, exp_end_ratio=None, dataset_index=None, calculate_distance_from_gt=False):
        
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
                #   continue
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
        # import pdb; pdb.set_trace()
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
            
            env = construct_env(cfg, config_file, solution_path, task_name, init_state_file)
            
            obs = env.reset()
            rgb = env.env.render()
            info = env.env._env._get_info()

            initial_info = info
            all_rgbs = [rgb]
            closed=False
            parallel_input_dict = obs
            # import pdb; pdb.set_trace()
            parallel_input_dict = dict_apply(parallel_input_dict, lambda x: torch.from_numpy(x).to('cuda'))
            for key in obs:
                parallel_input_dict[key] = parallel_input_dict[key].unsqueeze(0)
            with torch.no_grad():
                predicted_goal = goal_policy.predict_action(parallel_input_dict)
                np_predicted_goal = dict_apply(predicted_goal, lambda x: x.detach().to('cpu').numpy())
                np_predicted_goal = np_predicted_goal['action']
            env.env.goal_gripper_pcd = np_predicted_goal.squeeze(0)[0].reshape(4, 3)

            goal_pos, goal_orient = get_gripper_pos_orient_from_4_points(env.env.goal_gripper_pcd)
            res, rgbs = env.env.motion_planning_to_goal(goal_pos, goal_orient)
            # import pdb; pdb.set_trace()
            if res:
                all_rgbs.extend(rgbs)
                _, rgbs = env.env.close_two_fingers()
                all_rgbs.extend(rgbs)

                temp_obs = env.env._get_observation(render=True, only_object=env.env.only_object)
                for _ in range(2):
                    env.obs.append(temp_obs)
                obs = env._get_obs(env.n_obs_steps)
                parallel_input_dict = obs
                # import pdb; pdb.set_trace()
                parallel_input_dict = dict_apply(parallel_input_dict, lambda x: torch.from_numpy(x).to('cuda'))
                for key in obs:
                    parallel_input_dict[key] = parallel_input_dict[key].unsqueeze(0)
                with torch.no_grad():
                    predicted_goal = goal_policy.predict_action(parallel_input_dict)
                    np_predicted_goal = dict_apply(predicted_goal, lambda x: x.detach().to('cpu').numpy())
                    np_predicted_goal = np_predicted_goal['action']
                env.env.goal_gripper_pcd = np_predicted_goal.squeeze(0)[0].reshape(4, 3)

                goal_pos, goal_orient = get_gripper_pos_orient_from_4_points(env.env.goal_gripper_pcd)

                image = env.env.render()
                plt.imshow(image)
                plt.savefig("temp.png")
                print("Save second goal for opening door")

                _, rgbs = env.env.move_to_by_ik(goal_pos, goal_orient)
                all_rgbs.extend(rgbs)

                
            info = env.env._env._get_info()
            env.env._env.close()

            if calculate_distance_from_gt:
                continue
            
            opened_joint_angles[config_file] = \
            {
                "final_door_joint_angle": float(info['opened_joint_angle']), 
                "expert_door_joint_angle": expert_opened_angles[exp_idx], 
                "initial_joint_angle": float(info['initial_joint_angle']),
                "ik_failure": float(info['ik_failure']),
                'grasped_handle': float(info['grasped_handle']),
                "exp_idx": exp_idx, 
            }
                    
            with open("{}/opened_joint_angles_{}.json".format(save_path, dataset_idx), "w") as f:
                json.dump(opened_joint_angles, f, indent=4)
            
            gif_save_exp_name = experiment_folder.split("/")[-2]
            gif_save_folder = "{}/{}".format(save_path, gif_save_exp_name)
            if not os.path.exists(gif_save_folder):
                os.makedirs(gif_save_folder, exist_ok=True)
            gif_save_path = "{}/{}_{}.gif".format(gif_save_folder, exp_idx, 
                    float(info["improved_joint_angle"]))
            # import pdb; pdb.set_trace()
            save_numpy_as_gif(np.array(all_rgbs), gif_save_path)

        if calculate_distance_from_gt:
            print("average distance: {}".format(np.mean(all_distances)))
            print("average grasp distance: {}".format(np.mean(all_grasp_distances)))
            all_obj_distances.append(np.mean(all_distances))

    if calculate_distance_from_gt:
        print("average distance over all objects: {}".format(np.mean(all_obj_distances)))


if __name__ == "__main__":
    num_worker = 30
    pool=None

    # current best model
    # goal_checkpoint_name = 'epoch-30.ckpt'
    goal_checkpoint_name = 'epoch-45.ckpt'
    goal_exp_dir = '/project_data/held/ziyuw2/Robogen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0807-200-obj-pred-goal-gripper-PointNet2-backbone-UNet-diffusion-ep-75-epsilon/2024.08.07/14.03.40_train_dp3_robogen_open_door'
        
        
    with hydra.initialize(config_path='diffusion_policy_3d/config'):  # same config_path as used by @hydra.main
        recomposed_config = hydra.compose(
            config_name="dp3.yaml",  # same config_name as used by @hydra.main
            overrides=OmegaConf.load("{}/.hydra/overrides.yaml".format(goal_exp_dir)),
        )
    goal_cfg = recomposed_config
    cfg = deepcopy(goal_cfg)
        
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
    # import pdb; pdb.set_trace()
    cfg.task.env_runner.demo_experiment_path = [None for _ in range(10)]

    policy = None
        
    goal_workspace = TrainDP3Workspace(goal_cfg)
    goal_checkpoint_dir = "{}/checkpoints/{}".format(goal_exp_dir, goal_checkpoint_name)
    goal_workspace.load_checkpoint(path=goal_checkpoint_dir)

    goal_policy = deepcopy(goal_workspace.model)
    if goal_workspace.cfg.training.use_ema:
        goal_policy = deepcopy(goal_workspace.ema_model)
    goal_policy.eval()
    goal_policy.reset()
    goal_policy = goal_policy.to('cuda')
        
        
    checkpoint_dir = "{}/checkpoints/{}".format(goal_exp_dir, goal_checkpoint_name)
    checkpoint_name_start_idx = checkpoint_dir.find("3D-Diffusion-Policy/data/")  + len("3D-Diffusion-Policy/data/")
        
    for run_idx in range(1):
        save_path = "data/debug_replace_low_level_with_primitive/{}/{}".format(checkpoint_dir[checkpoint_name_start_idx:].replace("/", "_"), run_idx)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        cfg.task.env_runner.observation_mode = "act3d_goal_displacement_gripper_to_object"
        cfg.task.dataset.observation_mode = "act3d_goal_displacement_gripper_to_object"
        run_eval_non_parallel(
                cfg, policy, goal_cfg, goal_policy, 
                num_worker, save_path, 
                pool=pool, 
                horizon=35,
                exp_beg_idx=0,
                exp_end_idx=25,
        )
