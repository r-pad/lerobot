import numpy as np
from manipulation.utils import build_up_env
from manipulation.utils import load_env, rotation_transfer_6D_to_matrix, rotation_transfer_matrix_to_6D
import os
import pickle
import yaml
import zarr
import tqdm
import time
from manipulation.robogen_wrapper import RobogenPointCloudWrapper
from termcolor import cprint
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from multiprocessing import set_start_method
import json
from scipy.spatial.transform import Rotation as R
import pickle
from manipulation.utils import save_numpy_as_gif
from multiprocessing import Pool
from add_distractors_around_target import add_distractors_around_target
from collections import defaultdict
# from filter_obs_pcd_only_door_utils import get_filter_obs_pcd_plane


def sort_states_file_by_file_number(state_path):
    # all the file are named as state_0.pkl, state_1.pkl, ...
    ret_files = []
    for file in os.listdir(state_path):
        if file.startswith("state_") and file.endswith(".pkl"):
            ret_files.append(file)

    ret_files = sorted(ret_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
    return ret_files

def get_all_expert_angles(all_experiments, invert=False):
    ratios = []
    opened_angles = []
    for experiment_path in all_experiments:
        opened_angle_file = os.path.join(experiment_path, "opened_angle.txt")
        if os.path.exists(opened_angle_file): # for some perturbed trajectories, we did not really continue openeing the handle. 
            with open(opened_angle_file, "r") as f:
                angles = f.readlines()
                opened_angle = float(angles[0].lstrip().rstrip())
                max_angle = float(angles[-1].lstrip().rstrip())
                if invert:
                    opened_angle = max_angle - opened_angle
                ratio = opened_angle / max_angle
                ratios.append(ratio)
                opened_angles.append(opened_angle)
            
    return np.array(ratios), np.array(opened_angles)

def extract_pc_states_for_all_trajectories(pool_args):
    task_config_path, solution_path, env_name, exp_name, gripper_num_points, add_contact, \
        experiments, save_path, add_distractors, angle_threshold, invert, args = pool_args
       
    if exp_name is None:
        experiment_folder = os.path.join(solution_path, "experiment")
    else:
        experiment_folder = os.path.join(solution_path, "experiment", exp_name)
    if not os.path.exists(experiment_folder):
        print("Experiment folder does not exist: ", experiment_folder)
        if exp_name is not None:
            experiment_folder = solution_path
        else:
            experiment_folder = os.path.join(solution_path, exp_name)
        
    all_traj_stage_lengths = []
    all_traj_store_label_paths = []
    obs_keys = ['point_cloud', 'agent_pos', 'gripper_pcd', 'goal_gripper_pcd', 'displacement_gripper_to_object']
    all_traj_obs_dict_of_list = defaultdict(list)
    
    for experiment in experiments:
        if "meta" in experiment:
            continue
        
        expert_states = []
        experiment_path = os.path.join(experiment_folder, experiment)
        task_config_path = os.path.join(experiment_path, "task_config.yaml")
        states_path = os.path.join(experiment_path, "states")
        
        states = sort_states_file_by_file_number(states_path)      
        expert_states.extend([os.path.join(states_path, x) for x in states])
        if len(expert_states) == 0:
            print("No states found for experiment continue")
            continue
        
        stage_lengths = os.path.join(experiment_path, "stage_lengths.json")
        with open(stage_lengths, "r") as f:
            stage_lengths = json.load(f)

        reach_till_contact_idx = stage_lengths['reach_handle'] + stage_lengths["reach_to_contact"]
        open_time_idx = stage_lengths['reach_handle'] + stage_lengths["reach_to_contact"] + stage_lengths["close_gripper"]

        if len(expert_states) - open_time_idx < 5: # if the opening time is too short, skip this trajectory
            print("Opening time is too short, continue")
            continue
    
        opened_angle_file = os.path.join(experiment_path, "opened_angle.txt")
        if os.path.exists(opened_angle_file): 
            with open(opened_angle_file, "r") as f:
                angles = f.readlines()
                opened_angle = float(angles[0].lstrip().rstrip())
                max_angle = float(angles[-1].lstrip().rstrip())
                if invert:
                    opened_angle = max_angle - opened_angle
                ratio = opened_angle / max_angle
            if opened_angle < angle_threshold or ratio < args.min_opened_ratio:
                print("not open enough, continue")
                continue

        # already saved the data
        if os.path.exists(os.path.join(save_path, experiment)):
            print("Already saved the datsa, continue")
            continue

        if add_distractors:
            raise NotImplementedError("add_distractors is not implemented yet")
            new_yaml_path = os.path.join(experiment_path, "task_config_added_distractors.yaml")
            if not os.path.exists(new_yaml_path):
                add_distractors_around_target(task_config_path, solution_path, first_step.replace(" ", "_"), save_path=new_yaml_path)
            task_config_path = new_yaml_path
        
        bad_experiment = False
        camera_detected = True
        beg = time.time()
            
        if args.filter_pcd_only_door:
            raise NotImplementedError("filter pcd only door is not implemented yet")
            handle_length_offset = np.random.uniform(0.05, 0.1)
            filter_obs_pcd_plane = get_filter_obs_pcd_plane(task_config_path, solution_path, first_step, object_name, handle_length_offset=handle_length_offset)

        config = yaml.safe_load(open(task_config_path, "r"))
        for config_dict in config:
            if 'name' in config_dict:
                object_name = config_dict['name'].lower()

        simulator, _ = build_up_env(
            task_config=task_config_path,
            env_name=env_name,
            restore_state_file=None,
            render=False,
            randomize=False,
            obj_id=0,
        )
        simulator = RobogenPointCloudWrapper(simulator, 
            object_name, seed=0,
            gripper_num_points=gripper_num_points, add_contact=add_contact, num_points=args.pointcloud_num,
            use_segmask=args.use_segmask, only_handle_points=args.only_handle_points,
            observation_mode=args.observation_mode, record_all_observation=False, 
            noise_real_world_pcd=args.noise_real_world_pcd,
            real_world_camera=args.real_world_camera)

        if args.filter_pcd_only_door:
            raise NotImplementedError("filter pcd only door is not implemented yet")
            simulator.set_point_cloud_filter(filter_obs_pcd_plane)

        door_joint_angles = []
        traj_list = defaultdict(list)
        
        reach_till_contact_idx = reach_till_contact_idx // args.combine_action_steps
        open_time_idx = open_time_idx // args.combine_action_steps

        expert_states = expert_states[::args.combine_action_steps]
        load_env(simulator._env, load_path=expert_states[0])
        # random set the camera of the environment
        if args.randomize_camera:
            simulator.reset_random_cameras()
        
        # import pdb; pdb.set_trace()
        if simulator.check_handle_observed_in_pc() < 5:
            print("Handle not observed in the point cloud, continue")
            camera_detected = False
            simulator._env.close()

        if camera_detected:
            for t_idx, state in enumerate(tqdm.tqdm(expert_states)):
                # if t_idx % 10 == 0:
                #     print("extracting step {}".format(t_idx), flush=True)
                load_env(simulator._env, load_path=state)
                
                info = simulator._env._get_info()
                joint_angle = info['opened_joint_angle']
                door_joint_angles.append(joint_angle)
                
                # only object is the opposite to add_distractors
                only_object = not add_distractors
                observation = simulator._get_observation(only_object=only_object)  
                rgb = simulator._env.render()

                traj_list['rgb'].append(rgb)
                for key in obs_keys:
                    traj_list[key].append(observation[key].tolist())
                    
            goal_gripper_pcd_at_grasping = traj_list['gripper_pcd'][open_time_idx]
            goal_gripper_pcd_at_end = traj_list['gripper_pcd'][-1]
            for t in range(open_time_idx):
                traj_list['goal_gripper_pcd'][t] = goal_gripper_pcd_at_grasping
            for t in range(open_time_idx, len(traj_list['gripper_pcd'])):
                traj_list['goal_gripper_pcd'][t] = goal_gripper_pcd_at_end
                                
            door_joint_angles = np.array(door_joint_angles)
            door_joint_angle_diffs = np.diff(door_joint_angles)
            door_joint_angle_diffs = door_joint_angle_diffs[open_time_idx:]
            
            simulator._env.close()
    
        end = time.time()
        cprint(f"Finished extracting data from trajectory with length: {len(expert_states) // args.combine_action_steps} time cost {end - beg}", "green")

        if not bad_experiment and camera_detected:
            all_traj_stage_lengths.append(stage_lengths)
            all_traj_store_label_paths.append(experiment_path)
            for key in obs_keys + ['rgb']:
                all_traj_obs_dict_of_list[key].append(traj_list[key])
        else:
            label_path = os.path.join(experiment_path, "label.json")
            print("handle not in camera!!")
            try:
                with open(label_path, "w") as f:
                    json.dump({"good_traj": False, "failure reason": "simulation error during door opening"}, f)        
            except:
                pass
    
    return all_traj_obs_dict_of_list, all_traj_stage_lengths, all_traj_store_label_paths
    
def extract_demos_from_a_directory(dirtory_path, invert=False, exp_name=None, env_name=None, extract_name=None, parallel=True, 
                                    gripper_num_points=0, add_contact=False, save_path=None, add_distractors=False):
    
    action_dist_save_path = os.path.join(save_path, "action_dist")
    if not os.path.exists(action_dist_save_path):
        os.makedirs(action_dist_save_path)
    demo_rgb_save_path = os.path.join(save_path, "demo_rgbs")
    if not os.path.exists(demo_rgb_save_path):
        os.makedirs(demo_rgb_save_path)

    all_demo_paths = []
    task_path = extract_name
    solution_path = os.path.join(dirtory_path, task_path)
    files_and_folders = os.listdir(solution_path)
    task_config_path = None
    for file_or_folder in files_and_folders:
        if file_or_folder.endswith(".yaml"):
            task_config_path = os.path.join(dirtory_path, task_path, file_or_folder)
    if task_config_path is None:
        task_config_path = os.path.join('/mnt/RoboGen_sim2real/data/', solution_path.split("/")[-2], solution_path.split("/")[-1], "base_config.yaml")
        print("task_config_path is not found, using default path: ", task_config_path)
        if not os.path.exists(task_config_path):
            print("No task config found for task: ", task_path)
            return
    if exp_name is None:
        experiment_folder = os.path.join(solution_path, "experiment")
    else:
        experiment_folder = os.path.join(solution_path, "experiment", exp_name)
    if not os.path.exists(experiment_folder):
        print("Experiment folder does not exist: ", experiment_folder)
        if exp_name is not None:
            experiment_folder = solution_path
        else:
            experiment_folder = os.path.join(solution_path, exp_name)
    # all_experiments = os.listdir(experiment_folder)
    all_experiments = [x for x in os.listdir(experiment_folder) if os.path.isdir(os.path.join(experiment_folder, x)) and not x.startswith(".")]
    all_experiments = sorted(all_experiments)
    print("all experiments: ", all_experiments)
    # filter out successful experiments
    success_experiments = []
    for exp in all_experiments:
        state_path = os.path.join(experiment_folder, exp, "states")
        if os.path.exists(state_path):
            if len(os.listdir(state_path)) > 1 and os.path.exists(os.path.join(experiment_folder, exp, "all.gif")):
                success_experiments.append(exp)

    all_experiments = success_experiments
    num_experiment = len(all_experiments)
    
    _, all_expert_opened_angles = get_all_expert_angles([os.path.join(experiment_folder, exp) for exp in all_experiments], invert=invert)
    angle_threshold = np.quantile(all_expert_opened_angles, 0.1)
    
    num_experiment = min(num_experiment, args.num_experiment)        
    batch_size = 1
    num_batch = (num_experiment - 1) // batch_size + 1
    
    for batch_idx in range(num_batch):
        beg_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_experiment)
        
        # import pdb; pdb.set_trace()
        if not parallel:   
            all_traj_obs_dict_of_list, all_traj_stage_lengths, all_traj_store_label_paths = extract_pc_states_for_all_trajectories(
                [
                    task_config_path, solution_path, env_name, exp_name, 
                    gripper_num_points, add_contact, 
                    all_experiments[beg_idx:end_idx], 
                    save_path, add_distractors,
                    angle_threshold, invert, args
                ])
            # import pdb; pdb.set_trace()
        else:
            pool_args = [
                [task_config_path, solution_path, env_name, exp_name, gripper_num_points, add_contact, \
                all_experiments[i:i+1], save_path, add_distractors, angle_threshold, invert, args] for i in range(beg_idx, end_idx)
            ]
            beg = time.time()
            res = pool.map(extract_pc_states_for_all_trajectories, pool_args)
            end = time.time()
            cprint("Finished extracting data from all trajectories using time {}".format(end - beg), "green")
            all_traj_pc = [x[0][0] for x in res if len(x[0]) > 0]
            all_traj_pos_ori = [x[1][0] for x in res if len(x[1]) > 0]
            all_traj_rgbs = [x[2][0] for x in res if len(x[2]) > 0]
            all_traj_gripper_pcds = [x[3][0] for x in res if len(x[3]) > 0]
            all_traj_stage_lengths = [x[4][0] for x in res if len(x[4]) > 0]
            all_traj_store_label_paths = [x[5][0] for x in res if len(x[5]) > 0]

        all_traj_pc = all_traj_obs_dict_of_list['point_cloud']
        all_traj_pos_ori = all_traj_obs_dict_of_list['agent_pos']
        all_traj_rgbs = all_traj_obs_dict_of_list['rgb']
        all_traj_gripper_pcds = all_traj_obs_dict_of_list['gripper_pcd']
        all_traj_goal_gripper_pcd = all_traj_obs_dict_of_list['goal_gripper_pcd']
        all_traj_displacement_gripper_to_object = all_traj_obs_dict_of_list['displacement_gripper_to_object']
        
        # import pdb; pdb.set_trace()
        print("all traj_pc length: ", len(all_traj_pc))
        
        for traj_idx in tqdm.tqdm(range(len(all_traj_pc)), total=len(all_traj_pc)):

            traj_pc, traj_pos_ori, traj_gripper_pcd = all_traj_pc[traj_idx], all_traj_pos_ori[traj_idx], all_traj_gripper_pcds[traj_idx]
            traj_stage_length, traj_store_label_path  = all_traj_stage_lengths[traj_idx], all_traj_store_label_paths[traj_idx]
            # traj_dp3_pc = all_traj_dp3_pc[traj_idx]
            traj_goal_gripper_pcd, traj_displacement_gripper_to_object = all_traj_goal_gripper_pcd[traj_idx], all_traj_displacement_gripper_to_object[traj_idx]
            
            print(f"starting to save traj {traj_idx} with length {len(traj_pc)}")
            
            good_traj = True
            failure_reason = "null"

            traj_actions = []
            quaternion_diffs = []
            
            opening_start_idx = traj_stage_length['reach_handle'] + traj_stage_length['reach_to_contact'] + traj_stage_length['close_gripper']
            after_contact_idx = traj_stage_length['reach_handle'] + traj_stage_length['reach_to_contact']
            opening_start_idx = opening_start_idx // args.combine_action_steps
            after_contact_idx = after_contact_idx // args.combine_action_steps
        
            filtered_pcs = []
            filtered_pos_oris = []
            filtered_gripper_pcds = []
            filtered_rgbs = []
            # filtered_dp3_pcs = []
            filtered_goal_gripper_pcds = []
            filtered_displacement_gripper_to_objects = []
            
            base_pos = traj_pos_ori[0][:3]
            base_ori_6d = traj_pos_ori[0][3:9]
            base_finger_angle = traj_pos_ori[0][9]
            base_rgb = all_traj_rgbs[traj_idx][0]
            base_gripper_pcd = traj_gripper_pcd[0]
            base_pc = traj_pc[0]
            base_pos_ori = traj_pos_ori[0]
            # base_dp3_pc = traj_dp3_pc[0]
            base_goal_gripper_pcd = traj_goal_gripper_pcd[0]
            base_displacement_gripper_to_object = traj_displacement_gripper_to_object[0]
            
            for i in range(len(traj_pos_ori) - 1):
                # target_pos_ori = traj_pos_ori[i+1]
                cur_pos = traj_pos_ori[i][:3]
                target_pos = traj_pos_ori[i+1][:3]

                single_step_delta_pos = np.array(target_pos) - np.array(cur_pos)
                
                # if single step translation is too large, ignore this trajectory
                if np.linalg.norm(single_step_delta_pos) > 0.02 * args.combine_action_steps:
                    good_traj = False
                    failure_reason = "delta movement too large"
                    print("not good traj due to delta movement too large")
                    break
                
                delta_pos = np.array(target_pos) - np.array(base_pos)

                cur_ori_6d = traj_pos_ori[i][3:9]
                
                target_ori_6d = traj_pos_ori[i+1][3:9]
                cur_ori_matrix = rotation_transfer_6D_to_matrix(cur_ori_6d)
                base_ori_matrix = rotation_transfer_6D_to_matrix(base_ori_6d)
                target_ori_matrix = rotation_transfer_6D_to_matrix(target_ori_6d)

                delta_ori_matrix = base_ori_matrix.T @ target_ori_matrix
                delta_ori_6d = rotation_transfer_matrix_to_6D(delta_ori_matrix)
                
                cur_ori_quat =  R.from_matrix(cur_ori_matrix).as_quat()
                base_ori_quat =  R.from_matrix(base_ori_matrix).as_quat()
                target_ori_quat = R.from_matrix(target_ori_matrix).as_quat()
                quat_diff = np.arccos(2 * np.dot(base_ori_quat, target_ori_quat)**2 - 1)
                one_step_quaternion_diff = np.arccos(2 * np.dot(cur_ori_quat, target_ori_quat)**2 - 1)
                quaternion_diffs.append(quat_diff)
                
                # if single step rotation is too large, ignore this trajectory
                if np.abs(one_step_quaternion_diff) > 0.085 * args.combine_action_steps:
                    good_traj = False
                    failure_reason = "delta quaternion too large"
                    print("not good due to delta quaternion too large")
                    break
                                        
                target_finger_angle = traj_pos_ori[i+1][9]
                delta_finger_angle = target_finger_angle - base_finger_angle
                        
                filter_action = False
                if i >= after_contact_idx and i < opening_start_idx:
                    if np.abs(delta_finger_angle) < args.min_finger_angle_diff and args.filter_close_zero_action:
                        # print("filter close zero action")
                        filter_action = True
                        
                if filter_action:
                    continue
                else:
                    action = delta_pos.tolist() + delta_ori_6d.tolist() + [delta_finger_angle]
                    traj_actions.append(action)
                    filtered_pcs.append(base_pc)
                    filtered_gripper_pcds.append(base_gripper_pcd)
                    filtered_pos_oris.append(base_pos_ori)
                    filtered_rgbs.append(base_rgb)
                    # filtered_dp3_pcs.append(base_dp3_pc)
                    filtered_goal_gripper_pcds.append(base_goal_gripper_pcd)
                    filtered_displacement_gripper_to_objects.append(base_displacement_gripper_to_object)
                    
                    base_pc = traj_pc[i+1]
                    base_gripper_pcd = traj_gripper_pcd[i+1]
                    base_pos_ori = traj_pos_ori[i+1]
                    base_rgb = all_traj_rgbs[traj_idx][i+1]
                    base_pos = target_pos
                    base_ori_6d = target_ori_6d
                    base_finger_angle = target_finger_angle
                    # base_dp3_pc = traj_dp3_pc[i+1]
                    base_displacement_gripper_to_object = traj_displacement_gripper_to_object[i+1]
                    base_goal_gripper_pcd = traj_goal_gripper_pcd[i+1]
                    
        
            # plot the delta translation action distribution
            if traj_idx % 5 == 0:        
                try:
                    save_numpy_as_gif(np.array(filtered_rgbs), os.path.join(demo_rgb_save_path, "demo_" + str(traj_idx) + ".gif"))
                    plt.close("all")

                    delta_translations = np.array(traj_actions)[:, :3]
                    delta_translations_lengths = np.linalg.norm(delta_translations, axis=1)
                    delta_joint_angles = np.array(traj_actions)[:, -1]
                    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                    axes = axes.reshape(-1)
                    vals = [delta_translations_lengths, quaternion_diffs, delta_joint_angles]
                    titles = ["delta_translation_lengths", "quaternion_diffs", "delta_joint_angles"]
                    for idx, val in enumerate(vals):
                        axes[idx].plot(range(len(val)), val, "-*")
                        keys = ["reach_handle", "reach_to_contact", "close_gripper", "open_door"]
                        
                        base = 0
                        for key in keys:
                            base += traj_stage_length[key]
                            axes[idx].axvline(x=base, color='r', linestyle='--')
                            axes[idx].text(base, 0, key, rotation=90)
                        axes[idx].set_title(titles[idx])
                            
                    suffix = "good" if good_traj else "bad"
                    save_fig_path = os.path.join(action_dist_save_path, "delta_distribution_{}_{}.png".format(traj_idx, suffix))
                    plt.savefig(save_fig_path)
                    plt.close("all")
                except:
                    pass

            path = os.path.join(traj_store_label_path, "label.json")
            if not os.path.exists(path):
                try:
                    with open(path, 'w') as f:
                        json.dump({"good_traj": good_traj, "failure reason": failure_reason}, f)
                except:
                    pass

            print(f"finished parsing traj, right before saving data, good_traj is {good_traj}")
            if good_traj:
                for i in range(len(traj_actions)):
                    if i >= after_contact_idx:
                        # print("set gripper to be always close after contact")
                        traj_actions[i][-1] = args.close_gripper_action
                        
                experiment_path = traj_store_label_path
                all_demo_paths.append(experiment_path)
                data_save_path = os.path.join(save_path, experiment_path.split("/")[-1])
                beg = time.time()
                save_data_pickle(filtered_pcs, filtered_pos_oris, filtered_gripper_pcds, 
                        traj_actions, None, 
                        filtered_goal_gripper_pcds, filtered_displacement_gripper_to_objects, 
                        data_save_path)
                end = time.time()
                cprint("Finished saving data to {} using time {}".format(data_save_path, end-beg), "green")
                del filtered_pcs, filtered_pos_oris, filtered_gripper_pcds, traj_actions, filtered_displacement_gripper_to_objects, filtered_goal_gripper_pcds
        
        save_example_pointcloud([traj_pc[0] for traj_pc in all_traj_pc], save_path)
        # save_example_pointcloud([traj_dp3_pc[0] for traj_dp3_pc in all_traj_dp3_pc], save_path, name='example_dp3_pointcloud')
        del all_traj_pc, all_traj_pos_ori, all_traj_rgbs,  all_traj_gripper_pcds,  all_traj_stage_lengths, all_traj_store_label_paths
            
    with open(os.path.join(save_path, "all_demo_path.txt"), "w") as f:
        f.write("\n".join(all_demo_paths))


def save_data_pickle(pc_list, state_list, gripper_pcd_list, action_list, 
              dp3_pc_list,
              goal_gripper_pcd, 
              displacement_gripper_to_object,
              save_dir):

    state_arrays = np.array(state_list)
    point_cloud_arrays = np.array(pc_list)
    action_arrays = np.array(action_list)
    gripper_pcd_arrays = np.array(gripper_pcd_list)
    
    chunk_size = 1
    # state_chunk_size = (chunk_size, state_arrays.shape[1])
    # point_cloud_chunk_size = (chunk_size, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
    # action_chunk_size = (chunk_size, action_arrays.shape[1])
    # gripper_pcd_chunk_size = (chunk_size, gripper_pcd_arrays.shape[1], gripper_pcd_arrays.shape[2])
    if goal_gripper_pcd is not None:
        goal_gripper_pcd = np.array(goal_gripper_pcd)
        # goal_gripper_pcd_chunk_size = (chunk_size, goal_gripper_pcd.shape[1], goal_gripper_pcd.shape[2])
    if displacement_gripper_to_object is not None:
        displacement_gripper_to_object = np.array(displacement_gripper_to_object)
        # displacement_gripper_to_object_chunk_size = (chunk_size, displacement_gripper_to_object.shape[1], displacement_gripper_to_object.shape[2])
    if dp3_pc_list is not None:
        dp3_pc_list = np.array(dp3_pc_list)
        # dp3_point_cloud_chunk_size = (chunk_size, dp3_pc_list.shape[1], dp3_pc_list.shape[2])
        
    traj_len = len(state_list)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for t_idx in range(traj_len):
        step_save_dir = os.path.join(save_dir, str(t_idx) + ".pkl")
            
        pickle_data = {}
        pickle_data['state'] = state_arrays[t_idx][None, :]
        pickle_data['point_cloud'] = point_cloud_arrays[t_idx][None, :]
        pickle_data['action'] = action_arrays[t_idx][None, :]
        pickle_data['gripper_pcd'] = gripper_pcd_arrays[t_idx][None, :]
        if goal_gripper_pcd is not None:
            pickle_data['goal_gripper_pcd'] = goal_gripper_pcd[t_idx][None, :]
        if displacement_gripper_to_object is not None:
            pickle_data['displacement_gripper_to_object'] = displacement_gripper_to_object[t_idx][None, :]
        if dp3_pc_list is not None:
            pickle_data['dp3_point_cloud'] = dp3_pc_list[t_idx][None, :]
        
        with open(step_save_dir, 'wb') as f:
            pickle.dump(pickle_data, f)


    del state_arrays, point_cloud_arrays, gripper_pcd_arrays, action_arrays
    if goal_gripper_pcd is not None:
        del goal_gripper_pcd
    if displacement_gripper_to_object is not None:
        del displacement_gripper_to_object
    if dp3_pc_list is not None:
        del dp3_pc_list

def save_example_pointcloud(pc_list, save_dir, name='example_pointcloud'):
    if len(pc_list) > 10:
        idxes = np.random.choice(len(pc_list), 10)
    else:
        idxes = range(len(pc_list))
    save_dir = os.path.join(save_dir, "example_pointcloud")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, idx in enumerate(idxes):
        point_cloud = np.array(pc_list[idx])
        ax = plt.axes(projection='3d')
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
        ax.view_init(azim=-90, elev=10)
        plt.savefig(os.path.join(save_dir, "example_pc_" + str(i) + ".png"))
        # plt.show()
        plt.close()


def main(folder_name, save_path, invert=False, exp_name=None, env_name=None, extract_name=None, parallel=True,
         gripper_num_points=0, add_contact=False, add_distractors=False):
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    meta_info = {
        "folder_name": folder_name,
        "exp_name": exp_name,
    }
    meta_info.update(args.__dict__)
    with open(os.path.join(save_path, "meta_info.json"), "w") as f:
        json.dump(meta_info, f, indent=4)
    
    extract_demos_from_a_directory(folder_name, invert=invert, exp_name=exp_name, env_name=env_name, extract_name=extract_name, parallel=parallel, 
        gripper_num_points=gripper_num_points, add_contact=add_contact, save_path=save_path, add_distractors=add_distractors)
    

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--gripper_num_points", type=int, default=0)
    args.add_argument("--add_contact", type=int, default=0)
    args.add_argument("--after_opening", type=int, default=0)
    args.add_argument("--fixed_finger_movement", type=int, default=1)
    args.add_argument("--pointcloud_num", type=int, default=4500)
    args.add_argument("--use_segmask", type=int, default=0)
    args.add_argument("--only_handle_points", type=int, default=0)
    args.add_argument("--observation_mode", type=str, default='segmask')
    args.add_argument("--filter_close_zero_action", type=int, default=1)
    args.add_argument("--min_finger_angle_diff", type=float, default=0.001)
    args.add_argument("--close_gripper_action", type=float, default=-0.006)
    args.add_argument("--combine_action_steps", type=int, default=2)
    args.add_argument("--min_opened_ratio", type=float, default=0.2)
    args.add_argument("--randomize_camera", type=int, default=0)
    args.add_argument("--noise_real_world_pcd", type=int, default=0)
    args.add_argument("--real_world_camera", type=int, default=0)
    
    # args.add_argument("--object_name", type=str, required=True)
    args.add_argument("--save_path", type=str, required=True)
    args.add_argument("--folder_name", type=str, required=True)
    args.add_argument("--generate", type=int, default=1)
    args.add_argument("--parallel", type=int, default=1)
    args.add_argument("--exp_name", type=str, default=None)
    args.add_argument("--env_name", type=str, default="articulated")
    args.add_argument("--extract_name", type=str, default=None)
    args.add_argument("--num_experiment", type=int, default=10000)
    args.add_argument("--num_worker", type=int, default=20)
    args.add_argument("--add_distractors", type=int, default=0)
    args.add_argument("--filter_pcd_only_door", type=int, default=0, help="whether to filter the point cloud only containing the door, filter out the pointcloud of the 4 walls")
    args.add_argument("--invert", action='store_true')
    args = args.parse_args()
    
    if args.generate:
        if args.parallel:
            set_start_method('spawn', force=True)
            num_worker = args.num_worker
            pool = Pool(processes=num_worker)
        main(args.folder_name, args.save_path, invert=args.invert, exp_name=args.exp_name, env_name=args.env_name, extract_name=args.extract_name,
             parallel=args.parallel, 
             gripper_num_points=args.gripper_num_points, add_contact=args.add_contact, add_distractors=args.add_distractors)
    else:
        raise NotImplementedError("Not implemented yet")
        # # load the data
        zarr_root = zarr.open("data/dp3_demo/0512-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first-joint-angle-action")
        zarr_data = zarr_root['data']
        zarr_meta = zarr_root['meta']
        action_arrays = zarr_data['action'][:]
        last_state_indices = zarr_meta['episode_ends'][:]

        action_list = action_arrays.tolist()

        accumulated_angle_diff_list = []

        for j in range(len(last_state_indices)):

            # target_pos_ori = target_pos_ori[0]
            env, _ = build_up_env(
                "data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first/2024-05-11-00-24-52/task_config.yaml",
                "data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle",
                "grasp_the_door_handle",
                "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first/2024-05-11-00-24-52/grasp_the_door_handle_primitive/states/state_0.pkl", 
                render=True, 
                randomize=False,
                obj_id=0,
            )
            object_name = "StorageFurniture"
            env.reset()
            
            env = RobogenPointCloudWrapper(env, 
                object_name, seed=0,
                gripper_num_points=args.gripper_num_points, add_contact=args.add_contact, 
                num_points=args.pointcloud_num)
            
            rgbs = []

            np.random.seed(time.time_ns() % 2**32)
            robot = env._env.robot

            current_joint_angle = robot.get_joint_angles(robot.all_joint_indices)
            accumulated_angle_diff = 0
            if j == 0:
                offset = 0
            else:
                offset = last_state_indices[j]
                
            for i in range(min(400, len(action_list))):
                env.step(action_list[i+offset])
                rgbs.append(env.render())
                # control_rgbs = env._env.get_control_rgbs()
                # rgbs.extend(control_rgbs)

                # pos, ori = env._env.robot.get_pos_orient(env._env.robot.right_end_effector)
                
                # new_current_joint_angle = robot.get_joint_angles(robot.all_joint_indices)
                # diff = np.array(new_current_joint_angle) - np.array(current_joint_angle)
                # accumulated_angle_diff += np.linalg.norm(diff)
                # current_joint_angle = new_current_joint_angle

            # cprint("accumulated_angle_diff: " + str(accumulated_angle_diff), "green")
            # accumulated_angle_diff_list.append(accumulated_angle_diff)

            env._env.close()

            save_numpy_as_gif(np.array(rgbs), "data/debug.gif")
