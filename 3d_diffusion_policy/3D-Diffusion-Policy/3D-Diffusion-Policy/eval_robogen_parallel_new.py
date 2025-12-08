import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from train import TrainDP3Workspace
from diffusion_policy_3d.common.pytorch_util import dict_apply
from manipulation.utils import build_up_env, save_numpy_as_gif, save_env, load_env
import pybullet as p
import numpy as np
from copy import deepcopy
import sys
from termcolor import cprint
from manipulation.robogen_wrapper import RobogenPointCloudWrapper
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
# from manipulation.gpt_reward_api import get_joint_state
import tqdm
import json
from multiprocessing import set_start_method
from multiprocessing import Pool
import time
import yaml
import pickle as pkl

def parallel_eval(args):
    config_path, init_state, action, cfg, idx = args 
    config_file = config_path
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    # solution_path = [x['solution_path'] for x in config if "solution_path" in x][0]
    link_name = 'link_0'
    for config_dict in config:
        if 'name' in config_dict:
            object_name = config_dict['name'].lower()
        if 'link_name' in config_dict:
            link_name = config_dict['link_name']
    env, _ = build_up_env(
            config_path,
            "articulated",
            render=False, 
            randomize=False,
            obj_id=0,
            horizon=600,
    )
    
    # object_name = "StorageFurniture".lower()
    env.reset()
    pointcloud_env = RobogenPointCloudWrapper(env, object_name, link_name, in_gripper_frame=cfg.task.env_runner.in_gripper_frame, 
                                                    gripper_num_points=cfg.task.env_runner.gripper_num_points, add_contact=cfg.task.env_runner.add_contact,
                                                    num_points=cfg.task.env_runner.num_point_in_pc,
                                                    use_joint_angle=cfg.task.env_runner.use_joint_angle, 
                                                    use_segmask=cfg.task.env_runner.use_segmask,
                                                    only_handle_points=cfg.task.env_runner.only_handle_points,
                                                    observation_mode=cfg.task.env_runner.observation_mode,
                                                    only_object=cfg.task.env_runner.only_object,
                                                    use_absolute_waypoint=cfg.task.env_runner.use_absolute_waypoint,
                                                    use_chained_diffuser=cfg.task.env_runner.use_chained_diffuser
                                                  )
        
    env = MultiStepWrapper(pointcloud_env, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps, 
                        max_episode_steps=600, reward_agg_method='sum')
    
    env.reset(object_name=object_name, reset_state=init_state)
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
    link_name = 'link_0'
    for config_dict in config:
        if 'name' in config_dict:
            object_name = config_dict['name'].lower()
        if 'link_name' in config_dict:
            link_name = config_dict['link_name']
    
    env, _ = build_up_env(
            config_path,
            "articulated",
            render=False, 
            # render=True, 
            randomize=False,
            obj_id=0,
            horizon=600,
    )
    load_env(env, load_path=init_state_file)
    
    # object_name = "StorageFurniture".lower()
    env.reset(object_name=object_name)
    pointcloud_env = RobogenPointCloudWrapper(env, object_name, link_name=link_name,    
                                                  in_gripper_frame=cfg.task.env_runner.in_gripper_frame, 
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
    
    obs = env.reset(object_name=object_name)
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

def run_eval(cfg, policy, num_worker, save_path, exp_beg_idx=0, exp_end_idx=1000, pool=None, horizon=150,  exp_beg_ratio=None, exp_end_ratio=None, post_fix='', new_object=False, demo_experiment_path=''):
    # if type(cfg.task.env_runner.experiment_folder) != list:
    #     cfg.task.env_runner.experiment_folder = [cfg.task.env_runner.experiment_folder]
    # if type(cfg.task.env_runner.experiment_name) != list:
    #     cfg.task.env_runner.experiment_name = [cfg.task.env_runner.experiment_name]
    # if type(cfg.task.env_runner.demo_experiment_path) != list:
    #     cfg.task.env_runner.demo_experiment_path = [cfg.task.env_runner.demo_experiment_path]

    if new_object:
        cfg.task.env_runner.experiment_folder = ['data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_48700_2024-03-27-12-59-58/task_open_the_door_of_the_storagefurniture_by_its_handle']
        cfg.task.env_runner.experiment_name = ['0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first']
        cfg.task.env_runner.demo_experiment_path = [demo_experiment_path]

    # import pdb; pdb.set_trace()
    opened_joint_angles = {}
    for dataset_idx, (experiment_folder, experiment_name, demo_experiment_path) in enumerate(zip(cfg.task.env_runner.experiment_folder, cfg.task.env_runner.experiment_name, cfg.task.env_runner.demo_experiment_path)):
        # experiment_folder = cfg.task.env_runner.experiment_folder
        # experiment_name = cfg.task.env_runner.experiment_name
        # import pdb; pdb.set_trace()
    
        after_reaching_init_state_files = []
        init_state_files = []
        config_files = []
        experiment_folder = "{}/{}".format(os.environ['PROJECT_DIR'], experiment_folder)
        experiment_name = experiment_name
        experiment_path = os.path.join(experiment_folder, "experiment", experiment_name)
        all_experiments = os.listdir(experiment_path)
        all_experiments = sorted(all_experiments)
        
        if  demo_experiment_path is not None:
            # all_demo_path = os.path.join(os.environ['PROJECT_DIR'], cfg.task.env_runner.demo_experiment_path, "all_demo_path.txt")
            # with open(all_demo_path, "r") as f:
            #     all_demo_path = f.readlines()
            #     all_demo_path = [x.lstrip().rstrip().split("/")[-1] for x in all_demo_path]
            # all_experiments = all_demo_path
            # demo_experiment_path = demo_experiment_path[demo_experiment_path.find("RoboGen_sim2real/") + len("RoboGen_sim2real/"):]
            # if '/scratch' in demo_experiment_path:
            #     demo_experiment_path = demo_experiment_path.replace("/scratch/yufei", "/project_data/held/yufeiw2/RoboGen_sim2real/data")
            all_subfolder = os.listdir(demo_experiment_path)
            for string in ["action_dist", "demo_rgbs", "all_demo_path.txt", "meta_info.json", 'example_pointcloud']:
                if string in all_subfolder:
                    all_subfolder.remove(string)
            all_subfolder = sorted(all_subfolder)
            all_experiments = all_subfolder
            
        # all_substeps_path = os.path.join(experiment_folder, "substeps.txt")
        # with open(all_substeps_path, "r") as f:
        #     substeps = f.readlines()
        #     first_step = substeps[0].lstrip().rstrip()
        

        expert_opened_angles = []
        for experiment in all_experiments:
            if "meta" in experiment:
                continue
            
            # first_step_folder = first_step.replace(" ", "_") + "_primitive"
            # first_step_folder = os.path.join(experiment_path, experiment, first_step_folder)
            # if os.path.exists(os.path.join(first_step_folder, "label.json")):
            #     with open(os.path.join(first_step_folder, "label.json"), 'r') as f:
            #         label = json.load(f)
            #     if not label['good_traj']: continue
            exp_path = os.path.join(experiment_path, experiment)
            states_path = os.path.join(exp_path, "states")
            expert_states = os.listdir(states_path)
            if len(expert_states) == 0:
                continue
                
            expert_opened_angle_file = os.path.join(experiment_path, experiment, "opened_angle.txt")
            if os.path.exists(expert_opened_angle_file):
                with open(expert_opened_angle_file, "r") as f:
                    angles = f.readlines()
                    expert_opened_angle = float(angles[0].lstrip().rstrip())
                    max_angle = float(angles[-1].lstrip().rstrip())
                    ratio = expert_opened_angle / max_angle
                # if ratio < 0.65:
                #     continue
            expert_opened_angles.append(expert_opened_angle)
            
            stage_lengths = os.path.join(exp_path, "stage_lengths.json")
            with open(stage_lengths, "r") as f:
                stage_lengths = json.load(f)
            
            if 'stage' in stage_lengths:
                reaching_phase = stage_lengths.get('open_gripper', 0) + stage_lengths['grasp_handle']
            else:
                reaching_phase = stage_lengths['reach_handle']
                
            after_init_state_file = os.path.join(states_path, "state_{}.pkl".format(reaching_phase))
            after_reaching_init_state_files.append(after_init_state_file)
            init_state_file = os.path.join(states_path, "state_0.pkl")
            init_state_files.append(init_state_file)
            # config_file = os.path.join(experiment_path, experiment, "task_config_added_distractors.yaml")
            config_file = os.path.join(experiment_path, experiment, "task_config.yaml")
            config_files.append(config_file)
                    
        after_reaching_init_state_files = after_reaching_init_state_files
        config_files = config_files

        horizon = horizon

        if exp_end_ratio is not None:
            exp_end_idx = int(exp_end_ratio * len(config_files))
        if exp_beg_ratio is not None:
            exp_beg_idx = int(exp_beg_ratio * len(config_files))

        config_files = config_files[exp_beg_idx:exp_end_idx]
        init_state_files = init_state_files[exp_beg_idx:exp_end_idx]
        expert_opened_angles = expert_opened_angles[exp_beg_idx:exp_end_idx]
        num_iters = (len(config_files) - 1) // num_worker + 1
        for iter in range(num_iters):
            
            beg_idx = iter * num_worker
            end_idx = min((iter + 1) * num_worker, len(config_files))

            # first do reset of all envs
            args_to_run = [
                [config_files[idx], init_state_files[idx], cfg, idx] for idx in range(beg_idx, end_idx)
            ]
            results = pool.map(parallel_reset, args_to_run)
            # parallel_reset(args_to_run[0])
            results = sorted(results, key=lambda x: x[-1])
            res_obs = [res[0] for res in results]
            batched_states = [res[1] for res in results]
            batched_rgbs = [res[2] for res in results]
            batched_infos = [res[3] for res in results]
            batched_obs = wrap_obs(res_obs)
            with torch.no_grad():
                batched_action = policy.predict_action(batched_obs)
            np_batched_action = dict_apply(batched_action, lambda x: x.detach().to('cpu').numpy())
            np_batched_action = np_batched_action['action']
            
            initial_info = batched_infos
            all_rgbs = [batched_rgbs]
            max_door_joint_angles = np.ones(len(args_to_run)) * -1
            ik_failures = np.zeros(len(args_to_run))
            grasped_handles = np.zeros(len(args_to_run))
            
            for t_idx in tqdm.tqdm(range(1, horizon)):
                args_to_run = [
                    [config_files[idx], batched_states[idx - beg_idx], np_batched_action[idx - beg_idx], cfg, idx] for idx in range(beg_idx, end_idx)
                ]    
                beg = time.time()
                results = pool.map(parallel_eval, args_to_run)
                results = sorted(results, key=lambda x: x[-1])
                res_obs = [res[0] for res in results]
                res_rgb = [res[1] for res in results]
                res_info = [res[2] for res in results]
                res_states = [res[3] for res in results]
                end = time.time()
                door_joint_angles_step = np.array([float(info['initial_joint_angle'][-1]) for info in res_info])
                max_door_joint_angles = np.maximum(max_door_joint_angles, door_joint_angles_step)
                grasped_handle_step = np.array([float(info['grasped_handle'][-1]) for info in res_info])
                grasped_handles = np.logical_or(grasped_handles, grasped_handle_step)
                ik_failure_step = np.array([float(info['ik_failure'][-1]) for info in res_info])
                ik_failures = np.logical_or(ik_failures, ik_failure_step)
                
                # cprint("step time: {}".format(end - beg), "red")
                
                beg = time.time()
                batched_states = res_states
                batched_obs = wrap_obs(res_obs)
                with torch.no_grad():
                    batched_action = policy.predict_action(batched_obs)
                np_batched_action = dict_apply(batched_action, lambda x: x.detach().to('cpu').numpy())
                np_batched_action = np_batched_action['action']
                end = time.time()
                # cprint("predict time: {}".format(end - beg), "red")
                
                all_rgbs.append(res_rgb)

            for idx in range(beg_idx, end_idx):
                opened_joint_angles[config_files[idx]] = \
                    {
                        "max_door_joint_angle": max_door_joint_angles[idx - beg_idx],
                        "final_door_joint_angle": float(res_info[idx - beg_idx]['initial_joint_angle'][-1]), 
                        "expert_door_joint_angle": expert_opened_angles[idx], 
                        "initial_joint_angle": float(initial_info[idx - beg_idx]['initial_joint_angle']),
                        "ik_failure": float(ik_failures[idx - beg_idx]),
                        'grasped_handle': float(grasped_handles[idx - beg_idx]),
                    }
                
                # # [Chialiang]   
                # with open("{}/opened_joint_angles-{}{}.json".format(save_path, dataset_idx, post_fix), "w") as f:
                #     json.dump(opened_joint_angles, f, indent=4)
                # [Chialiang]   
                with open("{}/opened_joint_angles{}.json".format(save_path, post_fix), "w") as f:
                    json.dump(opened_joint_angles, f, indent=4)
            
            gif_save_exp_name = experiment_folder.split("/")[-1]
            gif_save_folder = "{}/{}".format(save_path, gif_save_exp_name)
            if not os.path.exists(gif_save_folder):
                os.makedirs(gif_save_folder, exist_ok=True)

            args_to_run = [
                [
                    [per_step_rgbs[idx] for per_step_rgbs in all_rgbs], 
                    "{}/{}_{}.gif".format(gif_save_folder, idx + beg_idx, float(res_info[idx]['initial_joint_angle'][-1]) - float(initial_info[idx]['initial_joint_angle']))
                ] for idx in range(end_idx - beg_idx)
            ]
            pool.map(parallel_save_gif, args_to_run)
            
        # # [Chialiang]   
        # with open("{}/opened_joint_angles{}.json".format(save_path, post_fix), "w") as f:
        #     json.dump(opened_joint_angles, f, indent=4)
            
def run_eval_non_parallel(cfg, policy, num_worker, save_path, exp_beg_idx=0, exp_end_idx=1000, pool=None, horizon=150,  exp_beg_ratio=None, exp_end_ratio=None, post_fix='', new_object=True, demo_experiment_path='', mobile=False):

    # cfg.task.env_runner.experiment_folder = [cfg.task.env_runner.experiment_folder]
    # cfg.task.env_runner.experiment_name = [cfg.task.env_runner.experiment_name]
    # cfg.task.env_runner.demo_experiment_path = [cfg.task.env_runner.demo_experiment_path]
    # cprint(cfg.task.env_runner.experiment_folder, 'green')
    # cprint(cfg.task.env_runner.experiment_name, 'green')
    # cprint(cfg.task.env_runner.demo_experiment_path, 'green')
    # cfg.task.env_runner.experiment_folder = ['data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle']
    # cfg.task.env_runner.experiment_name = ['0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first']

    if new_object:
        # cfg.task.env_runner.experiment_folder = ['data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_48700_2024-03-27-12-59-58/task_open_the_door_of_the_storagefurniture_by_its_handle']
        # cfg.task.env_runner.experiment_name = ['0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first']
        # cfg.task.env_runner.demo_experiment_path = [demo_experiment_path]
        cfg.task.env_runner.experiment_folder = [
            'data/diverse_objects/open_the_door_40147/task_open_the_door_of_the_storagefurniture_by_its_handle',
            'data/diverse_objects/open_the_door_44817/task_open_the_door_of_the_storagefurniture_by_its_handle',
            'data/diverse_objects/open_the_door_44962/task_open_the_door_of_the_storagefurniture_by_its_handle',
            'data/diverse_objects/open_the_door_45132/task_open_the_door_of_the_storagefurniture_by_its_handle',
            'data/diverse_objects/open_the_door_45219/task_open_the_door_of_the_storagefurniture_by_its_handle',
            'data/diverse_objects/open_the_door_45243/task_open_the_door_of_the_storagefurniture_by_its_handle',
            # 'data/diverse_objects/open_the_door_45297/task_open_the_door_of_the_storagefurniture_by_its_handle',
            'data/diverse_objects/open_the_door_45332/task_open_the_door_of_the_storagefurniture_by_its_handle',
            'data/diverse_objects/open_the_door_45378/task_open_the_door_of_the_storagefurniture_by_its_handle',
            'data/diverse_objects/open_the_door_45384/task_open_the_door_of_the_storagefurniture_by_its_handle',
            'data/diverse_objects/open_the_door_45463/task_open_the_door_of_the_storagefurniture_by_its_handle',
        ]
        cfg.task.env_runner.experiment_name = [
            '0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
            '0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
            '0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
            '0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
            '0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
            '0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
            # '0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
            '0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
            '0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
            '0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
            '0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first',
        ]
        cfg.task.env_runner.demo_experiment_path = [
            '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-40147',
            '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-44817',
            '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-44962',
            '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45132',
            '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45219',
            '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45243',
            # '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45297',
            '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45332',
            '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45378',
            '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45384',
            '/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0705-obj-45463',
        ]

    opened_joint_angles = {}

    for dataset_idx, (experiment_folder, experiment_name, demo_experiment_path) in enumerate(zip(cfg.task.env_runner.experiment_folder, cfg.task.env_runner.experiment_name, cfg.task.env_runner.demo_experiment_path)):
    
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
            
            if not mobile:
                first_stage_states_path = os.path.join(first_step_folder, "states")
            else:
                first_stage_states_path = os.path.join(first_step_folder, "mobile_states")
                
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
            if not mobile:
                config_file = os.path.join(experiment_path, experiment, "task_config.yaml")
            else:
                config_file = os.path.join(experiment_path, experiment, "mobile_config.yaml")
            config_files.append(config_file)
                    
        after_reaching_init_state_files = after_reaching_init_state_files
        config_files = config_files


        if exp_end_ratio is not None:
            exp_end_idx = int(exp_end_ratio * len(config_files))
        if exp_beg_ratio is not None:
            exp_beg_idx = int(exp_beg_ratio * len(config_files))

        config_files = config_files[exp_beg_idx:exp_end_idx]
        init_state_files = init_state_files[exp_beg_idx:exp_end_idx]
        expert_opened_angles = expert_opened_angles[exp_beg_idx:exp_end_idx]
        # import pdb; pdb.set_trace()
        for exp_idx, (config_file, init_state_file) in enumerate(zip(config_files, init_state_files)):
                
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            solution_path = [x['solution_path'] for x in config if "solution_path" in x][0]
            all_substeps_path = os.path.join(os.environ['PROJECT_DIR'], solution_path, "substeps.txt")
            with open(all_substeps_path, "r") as f:
                substeps = f.readlines()
                first_step = substeps[0].lstrip().rstrip()
                task_name = first_step.replace(" ", "_")
            
            env, _ = build_up_env(
                    config_file,
                    solution_path,
                    task_name,
                    init_state_file,
                    render=False, 
                    randomize=False,
                    obj_id=0,
                    horizon=600,
                    mobile=mobile,
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
                                                        use_absolute_waypoint=cfg.task.env_runner.use_absolute_waypoint,
                                                        use_chained_diffuser=cfg.task.env_runner.use_chained_diffuser,
                                                        dense_pcd_for_goal=cfg.task.env_runner.dense_pcd_for_goal,
                                                        )
                
            env = MultiStepWrapper(pointcloud_env, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps, 
                                max_episode_steps=600, reward_agg_method='sum')
            
            obs = env.reset()
            rgb = env.env.render()
            info = env.env._env._get_info()

            initial_info = info
            all_rgbs = [rgb]
            for t in range(1, horizon):
                parallel_input_dict = obs
                parallel_input_dict = dict_apply(parallel_input_dict, lambda x: torch.from_numpy(x).to('cuda'))
                for key in obs:
                    parallel_input_dict[key] = parallel_input_dict[key].unsqueeze(0)
                

                with torch.no_grad():
                    batched_action = policy.predict_action(parallel_input_dict)
                    
                np_batched_action = dict_apply(batched_action, lambda x: x.detach().to('cpu').numpy())
                np_batched_action = np_batched_action['action']
                
                obs, reward, done, info = env.step(np_batched_action.squeeze(0))
                rgb = env.env.render()
                all_rgbs.append(rgb)
            
            env.env._env.close()
            
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
            print(f'{gif_save_path} has been saved')
            
            save_numpy_as_gif(np.array(all_rgbs), gif_save_path)
        
if __name__ == "__main__":
    
    num_worker = 30
    pool = Pool(processes=num_worker)
    
    checkpoint_name = "epoch-200.ckpt"

    ### first generalization experiments 
    # exp_dir = "/project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0617-per-step-load-ddp-obj-45448-horizon-8-train-episodes-260/2024.06.18/11.34.47_train_dp3_robogen_open_door"
    # exp_dir = "/project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0617-per-step-load-ddp-obj-45448-horizon-8-train-episodes-260-gripper-goal/2024.06.17/22.05.56_train_dp3_robogen_open_door"

    ### add features as distance to closest object point
    # exp_dir = "/project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0622-per-step-load-ddp-obj-45448-horizon-8-train-episodes-260-with-gripper-displacement-to-closest-obj-point/2024.06.22/01.48.13_train_dp3_robogen_open_door"
    # exp_dir = "/project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0622-per-step-load-ddp-obj-45448-horizon-8-train-episodes-260-gripper-goal-with-gripper-displacement-to-closest-obj-point/2024.06.22/01.51.29_train_dp3_robogen_open_door"
    
    ### add features as distance to closest object point, with smoothed dataset
    # exp_dir = "/project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0623-smoothed-obj-45448-horizon-8-train-episodes-260-with-gripper-displacement-to-closest-obj-point/2024.06.23/14.32.11_train_dp3_robogen_open_door"
    # exp_dir = "/project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0623-smoothed-obj-45448-horizon-8-train-episodes-260-gripper-goal-with-gripper-displacement-to-closest-obj-point/2024.06.23/14.29.09_train_dp3_robogen_open_door"
    
    ### goal conditioning, alternating attention + self attention
    # exp_dir = "/project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0624-ddp-obj-45448-hor-8-train-ep-260-gripper-goal-w-gripper-displacement-to-closest-objpoint-self-attention/2024.06.25/01.16.16_train_dp3_robogen_open_door"
    
    # -------------------- #
    # -       0719       - #
    # -------------------- #

    # # act3d_goal_mlp displacement
    # exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/07191257-act3d_goal_mlp_displacement_gripper_to_object-horizon-8-num_load_episodes-1000/2024.07.19/12.57.05_train_dp3_robogen_open_door"
    # new_object = True
    # checkpoint_name = "latest.ckpt"

    # # act3d_goal_mlp displacement
    # exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/07201526-act3d_goal_mlp-horizon-8-num_load_episodes-1000/2024.07.20/15.26.54_train_dp3_robogen_open_door"
    # new_object = True
    # checkpoint_name = "latest.ckpt"
    ### no goal conditioning + self attention
    # exp_dir = "/project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0624-per-step-load-ddp-obj-45448-horizon-8-train-episodes-260-with-gripper-displacement-to-closest-obj-point-self-attention/2024.06.25/00.47.11_train_dp3_robogen_open_door"
    
    ### goal conditioning trained on 2 objects
    checkpoint_name = 'epoch-150.ckpt'
    exp_dir = "/project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0625-ddp-obj-45448-46462-hor-8-train-ep-260-gripper-goal-w-gripper-displacement-to-closest-objpoint/2024.06.25/13.53.54_train_dp3_robogen_open_door"
    
    ### goal conditioning trained on 3 objects
    checkpoint_name = 'epoch-175.ckpt'
    exp_dir = "/project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0627-ddp-obj-45448-46462-41510-hor-8-train-ep-260-gripper-goal-w-gripper-displacement-to-closest-objpoint/2024.06.27/00.42.24_train_dp3_robogen_open_door"
    
    ### no goal conditioning trained on 3 objects
    ### goal conditioning trained on 3 objects
    checkpoint_name = 'epoch-175.ckpt'
    exp_dir = "/project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0629-ddp-obj-45448-46462-41510-hor-8-train-ep-260-w-gripper-displacement-to-closest-objpoint/2024.06.29/01.14.30_train_dp3_robogen_open_door"
    
    

    # checkpoint_name = 'epoch-175.ckpt'
    # exp_dir = "/project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0629-ddp-obj-45448-46462-41510-hor-8-train-ep-260-w-gripper-displacement-to-closest-objpoint/2024.06.29/01.14.30_train_dp3_robogen_open_door"
    
    # ### with goal gripper, with self attention, fixed order bug in attention
    # checkpoint_name = 'epoch-300.ckpt'
    # exp_dir = "/project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0701-ddp-obj-45448-hor-8-train-ep-260-gripper-goal-w-gripper-displacement-to-closest-objpoint-self-attention-correct-order/2024.07.01/18.35.59_train_dp3_robogen_open_door"
    
    # ### w/o goal gripper, with self attention, fixed order bug in attention
    # checkpoint_name = 'epoch-150.ckpt'
    # exp_dir = "/project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0701-ddp-obj-45448-hor-8-train-ep-260-w-gripper-displacement-to-closest-objpoint-self-attention-correct-order/2024.07.02/15.18.18_train_dp3_robogen_open_door"
    
    ### Act3d + UNet + goal, trained on 10 objects
    # checkpoint_name = 'epoch-100.ckpt'
    # exp_dir = "/project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0710-10-obj-goal-act3d_goal_displacement_gripper_to_object-horizon-8-num_load_episodes-1000/2024.07.12/05.50.32_train_dp3_robogen_open_door/"
    
    ### Act3d + UNet no goal, trained on 10 objects
    # checkpoint_name = 'epoch-100.ckpt'
    # exp_dir = "/project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0710-10-obj-no-goal-act3d_displacement_gripper_to_object-horizon-8-num_load_episodes-1000/2024.07.12/05.50.32_train_dp3_robogen_open_door"
    
    ### chialiang's best low-level model
    # checkpoint_name = 'latest.ckpt'
    # exp_dir = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/07031908-act3d_goal_mlp-horizon-8-num_load_episodes-1000/2024.07.03/19.08.43_train_dp3_robogen_open_door"
    
    ### Act3d + UNet no goal, trained on 10 objects
    # checkpoint_name = 'epoch-100.ckpt'
    # exp_dir = "/project_data/held/yufeiw2/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0710-10-obj-no-goal-act3d_displacement_gripper_to_object-horizon-8-num_load_episodes-1000/2024.07.12/05.50.32_train_dp3_robogen_open_door"    # -------------------- #
   
    # -------------------- #
    # -       0802       - #
    # -------------------- #

    # exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/07282338-act3d_goal_mlp-n_obs_steps-4-horizon-8-num_load_episodes-1000-aug_pcd_rot/2024.07.28/23.39.05_train_dp3_robogen_open_door"
    # new_object = False
    # checkpoint_name = "latest.ckpt"

    # exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/07282339-act3d_goal_mlp-n_obs_steps-4-horizon-8-num_load_episodes-1000/2024.07.28/23.39.36_train_dp3_robogen_open_door"
    # new_object = False
    # checkpoint_name = "latest.ckpt"

    # exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/08030037-act3d_goal_mlp-n_obs_steps-4-horizon-8-num_load_episodes-1000-normalize_action/2024.08.03/00.37.23_train_dp3_robogen_open_door"
    # new_object = False
    # checkpoint_name = "latest.ckpt"
    
    # -------------------- #
    # -       0803       - #
    # -------------------- #

    # exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/08031154-act3d_goal_mlp-n_obs_steps-4-horizon-8-num_load_episodes-1000-aug_pcd_rot/2024.08.03/11.54.43_train_dp3_robogen_open_door"
    # new_object = False
    # checkpoint_name = "latest.ckpt"
    
    # -------------------- #
    # -       0809       - #
    # -------------------- #

    # # delta wpt, aug rot
    # exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/08081744-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-aug_rot/2024.08.08/17.44.49_train_dp3_robogen_open_door"
    # new_object = True
    # checkpoint_name = "latest.ckpt"

    # # delta wpt, aug noise
    # exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/08062012-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-aug_pcd/2024.08.06/20.12.30_train_dp3_robogen_open_door"
    # new_object = True
    # checkpoint_name = "latest.ckpt"

    # # delta wpt, aug scale
    # exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/08090103-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-aug_scale/2024.08.09/01.03.17_train_dp3_robogen_open_door"
    # new_object = True
    # checkpoint_name = "latest.ckpt"

    # # delta wpt, aug all
    # exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/08091652-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-aug_all/2024.08.09/16.53.26_train_dp3_robogen_open_door"
    # new_object = True
    # checkpoint_name = "latest.ckpt"

    # # delta wpt, scale scene by pcd
    # exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/08110058-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-scale_scene_by_pcd/2024.08.11/00.58.27_train_dp3_robogen_open_door"
    # new_object = True
    # checkpoint_name = "latest.ckpt"

    #############################################################

    # # delta wpt, epsilon, normalize action
    # exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/08101108-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-normalize_action-epsilon/2024.08.10/11.08.09_train_dp3_robogen_open_door"
    # new_object = True
    # checkpoint_name = "latest.ckpt"

    # # delta wpt, epsilon, aug rot
    # exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/08101115-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-aug_rot-epsilon/2024.08.10/11.15.56_train_dp3_robogen_open_door"
    # new_object = True
    # checkpoint_name = "latest.ckpt"

    #############################################################
    
    # # abs wpt, aug noise
    # exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/08081732-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-abs_wpt/2024.08.08/17.46.44_train_dp3_robogen_open_door"
    # new_object = True
    # checkpoint_name = "latest.ckpt"

    # # abs wpt, aug noise
    # exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/08090103-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-abs_wpt-aug_scale/2024.08.09/01.03.16_train_dp3_robogen_open_door"
    # new_object = True
    # checkpoint_name = "latest.ckpt"
    
    # #############################################################
    
    # # abs wpt, epsilon, normalize action
    # exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/08110104-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-abs_wpt-epsilon/2024.08.11/01.04.17_train_dp3_robogen_open_door"
    # new_object = True
    # checkpoint_name = "latest.ckpt"
    
    # # abs wpt, epsilon, aug rot
    # exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/08110104-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-abs_wpt-aug_rot-epsilon/2024.08.11/01.04.19_train_dp3_robogen_open_door"
    # new_object = True
    # checkpoint_name = "latest.ckpt"

    # #############################################################

    # # delta wpt, v_prediction, normalize action
    # exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/08120322-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-normalize_action-v_prediction/2024.08.12/03.22.34_train_dp3_robogen_open_door"
    # new_object = True
    # checkpoint_name = "latest.ckpt"

    # # delta wpt, v_prediction, aug rot
    # exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/08120006-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-aug_rot-v_prediction/2024.08.12/00.06.46_train_dp3_robogen_open_door"
    # new_object = True
    # checkpoint_name = "latest.ckpt"

    # #############################################################

    # # delta wpt, sample, normalize action, all objects
    # exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/08121539-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-all_object-normalize_action/2024.08.12/15.40.03_train_dp3_robogen_open_door"
    # new_object = True
    # checkpoint_name = "latest.ckpt"

    # # delta wpt, sample, normalize action, all objects, 30 epochs
    # exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/08130042-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-all_object-normalize_action-30/2024.08.13/00.42.27_train_dp3_robogen_open_door"
    # new_object = True
    # checkpoint_name = "latest.ckpt"

    # # # delta wpt, v_prediction, normalize action, all objects, 30 epochs
    # exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/08131154-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-abs_wpt-v_prediction/2024.08.13/11.54.56_train_dp3_robogen_open_door"
    # new_object = True
    # checkpoint_name = "latest.ckpt"

    # #############################################################

    # # delta wpt, sample, normalize action, all obj
    # exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/08141110-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-all_object-normalize_action-30/2024.08.14/11.10.47_train_dp3_robogen_open_door"
    # new_object = True
    # checkpoint_name = "latest.ckpt"

    # # delta wpt, sample, normalize action, pointnet++
    # exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/08140150-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-pn2-normalize_action/2024.08.14/01.50.59_train_dp3_robogen_open_door"
    # new_object = True
    # checkpoint_name = "latest.ckpt"

    # # delta wpt, epsilon, normalize action, pcd noise, pointnet++
    # exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/08150052-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-all_object-pn2-aug_pcd-epsilon/2024.08.15/00.52.14_train_dp3_robogen_open_door"
    # new_object = True
    # checkpoint_name = "latest.ckpt"

    # # delta wpt, epsilon, normalize action, pointnet++
    # exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/08161917-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-all_object-pn2-aug_pcd-epsilon-new/2024.08.16/19.17.46_train_dp3_robogen_open_door"
    # new_object = True
    # checkpoint_name = "latest.ckpt"

    # # #############################################################

    # # delta wpt, sample, normalize action, dense wpt
    # exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/08142025-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-dense_gripper/2024.08.14/20.25.35_train_dp3_robogen_open_door"
    # new_object = True
    # checkpoint_name = "latest.ckpt"

    # # #############################################################

    # # delta wpt, sample, normalize action, combine-2 (new), observation horizon=2
    # exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/08191542-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-combine-pcd_noise-new/2024.08.19/15.42.15_train_dp3_robogen_open_door"
    # new_object = True
    # checkpoint_name = "latest.ckpt"

    # # # delta wpt, sample, normalize action, combine-2 (new), dense gripper, observation horizon=2
    # exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/08191250-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-dense_gripper-combine-pcd_noise-new/2024.08.19/12.50.15_train_dp3_robogen_open_door"
    # new_object = True
    # checkpoint_name = "latest.ckpt"
    
    # # #############################################################
    # # debug, no use
    # exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/08191826-act3d_goal_mlp-n_obs_steps-2-horizon-8-num_load_episodes-1000-nouse/2024.08.19/18.26.15_train_dp3_robogen_open_door"
    # new_object = True
    # checkpoint_name = "latest.ckpt"
    
    # #############################################################
    # debug, no use
    exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/08240109-act3d_goal_mlp_displacement_gripper_to_object-ns-2-h-8-demonum-75-all-pt_goal/2024.08.24/01.09.48_train_dp3_robogen_open_door"
    new_object = True
    checkpoint_name = "latest.ckpt"

    ### Chialiang's best low-level model trained on 50 objects
    checkpoint_name = 'latest.ckpt'
    exp_dir = exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/07201526-act3d_goal_mlp-horizon-8-num_load_episodes-1000/2024.07.20/15.26.54_train_dp3_robogen_open_door"

    with hydra.initialize(config_path='diffusion_policy_3d/config'):  # same config_path as used by @hydra.main
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
    
    checkpoint_dir = "{}/checkpoints/{}".format(exp_dir, checkpoint_name)
    checkpoint_name_start_idx = checkpoint_dir.find("3D-Diffusion-Policy/data/")  + len("3D-Diffusion-Policy/data/")
    
    for run_idx in range(3):
        save_path = "data/eval_generalization_mulitple_object_multiple_runs_non_parallel/{}/{}".format(checkpoint_dir[checkpoint_name_start_idx:].replace("/", "_"), run_idx)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        exp_beg_ratio = 0.9
        exp_end_ratio = 1
            
        run_eval_non_parallel(cfg, policy, num_worker, save_path, 
                pool=pool, 
                horizon=35,
                # exp_beg_ratio=exp_beg_ratio,
                # exp_end_ratio=exp_end_ratio,
                exp_beg_idx=0, exp_end_idx=25,
                post_fix=f'-unseen-{run_idx}',
                new_object=new_object,
                mobile=False,
        )
    
        # run_eval(cfg, policy, num_worker, save_path, 
        #          pool=pool, 
        #          horizon=35,
        #          exp_beg_ratio=exp_beg_ratio,
        #          exp_end_ratio=exp_end_ratio,
        # )
    