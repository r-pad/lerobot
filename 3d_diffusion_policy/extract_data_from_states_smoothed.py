# TODO: after the reach till contact stage, the gripper action should always be close. 
import numpy as np
from manipulation.utils import build_up_env
from manipulation.utils import load_env, rotation_transfer_6D_to_matrix, rotation_transfer_matrix_to_6D, save_env
import os
from copy import deepcopy
import zarr
import tqdm
import time
from manipulation.robogen_wrapper import RobogenPointCloudWrapper
from termcolor import cprint
import shutil
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from multiprocessing import set_start_method
import json
from scipy.spatial.transform import Rotation as R
import pickle
from manipulation.utils import save_numpy_as_gif
from multiprocessing import Pool
import pybullet as p

def slerp(q1, q2, t):
    dot = np.dot(q1, q2)
    dot = np.clip(dot, -1.0, 1.0)
    
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    
    q3 = q2 - q1 * dot
    q3 = q3 / np.linalg.norm(q3)
    
    return q1 * np.cos(theta) + q3 * np.sin(theta)

def linear_interpolation(q1, q2, t):
    return q1 + t * (q2 - q1)

def parallel_render(args):
    task_config_path, solution_path, first_step, rpy, in_gripper_frame, gripper_num_points, add_contact, \
        state, object_name, num_point_in_pc, use_joint_angle, use_segmask, only_handle_points, observation_mode, idx = args
    
    # cprint("Extracting data from state idx " + str(idx), "blue")
    simulator, _ = build_up_env(
                task_config=task_config_path,
                solution_path=solution_path,
                task_name=first_step.replace(" ", "_"),
                restore_state_file=None,
                render=False,
                randomize=False,
                obj_id=0,
    )
    
    simulator = RobogenPointCloudWrapper(simulator, 
        object_name, rpy_mean_list=rpy, seed=0, in_gripper_frame=in_gripper_frame, 
        gripper_num_points=gripper_num_points, add_contact=add_contact, num_points=num_point_in_pc,
        use_joint_angle=use_joint_angle, use_segmask=use_segmask, only_handle_points=only_handle_points, 
        observation_mode=observation_mode)
    
    load_env(simulator._env, load_path=state)
    observation = simulator._get_observation()
    rgb = simulator._env.render()
    
    point_cloud = observation['point_cloud'].tolist()
    traj_pos_ori = observation['agent_pos'].tolist()
    feature_map = observation['feature_map'].tolist()
    gripper_pcd = observation['gripper_pcd'].tolist()
    pcd_mask = observation['pcd_mask'].tolist()
    
    simulator._env.close()
        
    return point_cloud, traj_pos_ori, rgb, feature_map, gripper_pcd, pcd_mask, idx

def sort_states_file_by_file_number(state_path):
    # all the file are named as state_0.pkl, state_1.pkl, ...
    ret_files = []
    for file in os.listdir(state_path):
        if file.startswith("state_") and file.endswith(".pkl"):
            ret_files.append(file)

    ret_files = sorted(ret_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
    return ret_files

def smooth_reach_to_contact_stage(states, task_config_path=None, solution_path=None, task_name=None, exp_folder=None):
    ret_states = []
    init_state = states[0]
    end_state = states[-1]
    smoothed_length = len(states) - 1
    simulator, _ = build_up_env(
                        task_config=task_config_path,
                        solution_path=solution_path,
                        task_name=task_name,
                        restore_state_file=None,
                        render=False,
                        randomize=False,
                        obj_id=0,
                    )
    # for s in states:
    #     load_env(simulator, load_path=s)
    #     robot_pos, _ = simulator.robot.get_pos_orient(simulator.robot.right_end_effector)
    #     p.addUserDebugPoints([robot_pos], [[1,0,0]], pointSize=10)
    #     import pdb; pdb.set_trace()


    load_env(simulator, load_path=end_state)
    end_robot_pos, end_robot_ori = simulator.robot.get_pos_orient(simulator.robot.right_end_effector)
    load_env(simulator, load_path=init_state)
    init_robot_pos, init_robot_ori = simulator.robot.get_pos_orient(simulator.robot.right_end_effector)

    ret_states.append(init_state)
    # interpolate the robot position and orientation into smoothed_length parts
    # the first part has length smoothed_length 
    # the second part has length smoothed_length - 1
    # ...
    # the last part has length 1
    total_length = smoothed_length * (smoothed_length + 1) / 2
    quantiles = 1 / total_length
    agent = simulator.robot
    joint = agent.right_end_effector if 'right' in agent.controllable_joints else agent.left_end_effector
    ik_indices = [_ for _ in range(len(agent.right_arm_ik_indices))]
    for i in range(smoothed_length):
        ratio = (2 * smoothed_length - i) * (i + 1) * quantiles / 2
        new_pos = linear_interpolation(init_robot_pos, end_robot_pos, ratio)
        new_ori = slerp(init_robot_ori, end_robot_ori, ratio)
        agent_joint_angles = agent.ik(joint, new_pos, new_ori, ik_indices, max_iterations=5000)
        for _ in range(10):
            agent.control(agent.controllable_joint_indices, agent_joint_angles)
            p.stepSimulation(physicsClientId=simulator.id)
        save_env(simulator, save_path=os.path.join(exp_folder, "smoothed_state_" + str(i+1) + ".pkl"))
        ret_states.append(os.path.join(exp_folder, "smoothed_state_" + str(i+1) + ".pkl"))
        # robot_pos, _ = simulator.robot.get_pos_orient(simulator.robot.right_end_effector)
        # p.addUserDebugPoints([robot_pos], [[0,1,0]], pointSize=10)
        # import pdb; pdb.set_trace()

    simulator.close()
    
    return ret_states


def extract_pc_states_for_all_trajectories(task_config_path, solution_path, object_name, exp_name=None, 
                                           in_gripper_frame=False, parallel=True,
                                           gripper_num_points=0, add_contact=False, smooth_reach_to_contact_stage=False):
    # make the stage "reach_to_contact" smoother (no sharp change in movement)
    
    if exp_name is None:
        experiment_folder = os.path.join(solution_path, "experiment")
    else:
        experiment_folder = os.path.join(solution_path, "experiment", exp_name)
    all_experiments = os.listdir(experiment_folder)
    all_experiments = sorted(all_experiments)
    all_experiments = all_experiments
    all_experiments = all_experiments[:args.num_experiment]
    
    all_traj_pc = []
    all_traj_pos_ori = []
    all_traj_stage_lengths = []
    all_traj_feature_maps = []
    all_traj_gripper_pcds = []
    all_traj_store_label_paths = []
    all_traj_rgbs = []
    all_traj_pcd_masks = []
    for experiment in tqdm.tqdm(all_experiments):
        if "meta" in experiment:
            continue
        
        expert_states = []
        experiment_path = os.path.join(experiment_folder, experiment)
        cprint("Extracting data from experiment: " + experiment, "blue")
        task_config_path = os.path.join(experiment_path, "task_config.yaml")

        all_substeps_path = os.path.join(solution_path, "substeps.txt")
        with open(all_substeps_path, "r") as f:
            substeps = f.readlines()
            first_step = substeps[0].lstrip().rstrip()

        all_substeps_type = os.path.join(solution_path, "substep_types.txt")
        with open(all_substeps_type, "r") as f:
            all_substeps_type = f.readlines()
            first_step_type = all_substeps_type[0].lstrip().rstrip()
        first_step_folder = first_step.replace(" ", "_") + "_"  + first_step_type
        first_stage_states_path = os.path.join(experiment_path, first_step_folder, "states")
        
            
        first_stage_states = sort_states_file_by_file_number(first_stage_states_path)      
        expert_states.extend([os.path.join(first_stage_states_path, x) for x in first_stage_states])
        if len(expert_states) == 0:
            print("No states found for experiment continue")
            continue
        
        label_path = os.path.join(experiment_path, first_step_folder, "label.json")
        if os.path.exists(label_path):
            label = json.load(open(label_path, "r"))
            if not label["good_traj"]:
                print("Not good traj continue")
                continue
        
        stage_lengths = os.path.join(experiment_path, first_step_folder, "stage_lengths.json")
        with open(stage_lengths, "r") as f:
            stage_lengths = json.load(f)
        if 'reach_handle' in stage_lengths.keys():
            reaching_phase = stage_lengths['reach_handle']
        else:
            reaching_phase = stage_lengths.get('open_gripper', 0) + stage_lengths['grasp_handle']
    
        opened_angle_file = os.path.join(experiment_path, first_step_folder, "opened_angle.txt")
        if os.path.exists(opened_angle_file): # for some perturbed trajectories, we did not really continue openeing the handle. 
            with open(opened_angle_file, "r") as f:
                angles = f.readlines()
                opened_angle = float(angles[0].lstrip().rstrip())
                max_angle = float(angles[-1].lstrip().rstrip())
                ratio = opened_angle / max_angle
            if not (ratio > args.opened_threshold):
                print("not open enough, continue")
                continue

        all_traj_stage_lengths.append(stage_lengths)
        all_traj_store_label_paths.append(os.path.join(experiment_path, first_step_folder))

        # stored_pkl_path = os.path.join(experiment_path, first_step_folder, "extracted_ja_{}_sm_{}_hd_{}.pkl".format(args.use_joint_angle, args.use_segmask, args.only_handle_points))   
        pickle_loaded = False
        stored_pkl_path = os.path.join(experiment_path, first_step_folder, "extracted_{}.pkl".format(args.observation_mode))   
        if args.use_extracted and os.path.exists(stored_pkl_path):
            print("Using extracted data")
            beg = time.time()
            with open(stored_pkl_path, "rb") as f:
               
                try:
                    data = pickle.load(f)
                    if len(data) == 3:
                        pc_list, pos_ori_list, rgb_list = data
                    else:
                        pc_list, pos_ori_list, rgb_list, feature_map_list, gripper_pcd_list, pcd_mask_list = data
                        
                    assert len(pc_list[0]) == args.pointcloud_num, f"point cloud length is {len(pc_list[0])}, should be {args.pointcloud_num}"
                    pickle_loaded = True
                    if args.after_reaching:
                        pc_list = pc_list[reaching_phase:]
                        pos_ori_list = pos_ori_list[reaching_phase:]
                        if rgb_list is not None:
                            rgb_list = rgb_list[reaching_phase:]
                    if args.after_opening:
                        pc_list = pc_list[stage_lengths['open_gripper']:]
                        pos_ori_list = pos_ori_list[stage_lengths['open_gripper']:]
                        if rgb_list is not None:
                            rgb_list = rgb_list[stage_lengths['open_gripper']:]
                    
                    end = time.time()
                    cprint(f"Finished extracting data from trajectory index: {str(len(all_traj_pc))} time cost {end - beg}" , "green")

                except:
                    pickle_loaded = False
                
        if not pickle_loaded:
            rpy_list = [[[0, 0, -45], [0, 0, -135]]]
            beg = time.time()
            for rpy in rpy_list:
                
                if smooth_reach_to_contact_stage:
                    expert_states[stage_lengths['reach_handle'] : stage_lengths['reach_handle'] + stage_lengths['reach_to_contact']] = \
                        smooth_reach_to_contact_stage(expert_states[stage_lengths['reach_handle'] : stage_lengths['reach_handle'] + stage_lengths['reach_to_contact']], task_config_path, solution_path, first_step, first_stage_states_path)

                if not parallel:
                    simulator, _ = build_up_env(
                        task_config=task_config_path,
                        solution_path=solution_path,
                        task_name=first_step.replace(" ", "_"),
                        restore_state_file=None,
                        render=False,
                        randomize=False,
                        obj_id=0,
                    )
                    simulator = RobogenPointCloudWrapper(simulator, 
                        object_name, rpy_mean_list=rpy, seed=0, in_gripper_frame=in_gripper_frame, 
                        gripper_num_points=gripper_num_points, add_contact=add_contact, num_points=args.pointcloud_num,
                        use_joint_angle=args.use_joint_angle, use_segmask=args.use_segmask, only_handle_points=args.only_handle_points,
                        observation_mode=args.observation_mode)
                        
                    pc_list = []
                    pos_ori_list = []
                    rgb_list = []
                    feature_map_list = []
                    gripper_pcd_list = []
                    pcd_mask_list = []
                    for state in tqdm.tqdm(expert_states):
                        load_env(simulator._env, load_path=state)
                        observation = simulator._get_observation()            
                        rgb = simulator._env.render()
                        point_cloud = observation['point_cloud'].tolist()
                        traj_pos_ori = observation['agent_pos'].tolist()
                        feature_map = observation['feature_map'].tolist()
                        gripper_pcd = observation['gripper_pcd'].tolist()
                        pcd_mask = observation['pcd_mask'].tolist()
                        
                        pc_list.append(point_cloud)
                        pos_ori_list.append(traj_pos_ori)
                        rgb_list.append(rgb)
                        feature_map_list.append(feature_map)
                        gripper_pcd_list.append(gripper_pcd)
                        pcd_mask_list.append(pcd_mask)
                        
                    simulator._env.close()
                else:
                    # parallel version
                    if args.after_reaching:
                        expert_states = expert_states[reaching_phase:]
                    if args.after_opening:
                        expert_states = expert_states[stage_lengths['open_gripper']:]
                    results = pool.map(parallel_render, 
                        [(task_config_path, solution_path, first_step, rpy, in_gripper_frame, gripper_num_points, add_contact,
                        expert_states[i], object_name, args.pointcloud_num, args.use_joint_angle, args.use_segmask, args.only_handle_points, \
                            args.observation_mode,
                            i) for i in range(len(expert_states))])
                    results = sorted(results, key=lambda x: x[-1])
                    # print([result[2] for result in results])
                    pc_list = [x[0] for x in results]        
                    pos_ori_list = [x[1] for x in results]
                    rgb_list = [x[2] for x in results]
                    feature_map_list = [x[3] for x in results]
                    gripper_pcd_list = [x[4] for x in results]
                    pcd_mask_list = [x[5] for x in results]
    
            end = time.time()
            cprint(f"Finished extracting data from trajectory index: {str(len(all_traj_pc))} time cost {end - beg}" , "green")

        all_traj_pc.append(pc_list)
        all_traj_pos_ori.append(pos_ori_list)
        all_traj_rgbs.append(rgb_list)
        all_traj_feature_maps.append(feature_map_list)
        all_traj_gripper_pcds.append(gripper_pcd_list)
        all_traj_pcd_masks.append(pcd_mask_list)
            
        if not args.after_reaching and not args.after_opening:
            # store_pickle_path = os.path.join(experiment_path, first_step_folder, "extracted_ja_{}_sm_{}_hd_{}.pkl".format(args.use_joint_angle, args.use_segmask, args.only_handle_points))
            store_pickle_path = os.path.join(experiment_path, first_step_folder, "extracted_{}.pkl".format(args.observation_mode))
            with open(store_pickle_path, "wb") as f:
                pickle.dump((pc_list, pos_ori_list, rgb_list, feature_map_list, gripper_pcd_list, pcd_mask_list), f, protocol=pickle.HIGHEST_PROTOCOL)
        
    return all_traj_pc, all_traj_pos_ori, all_traj_rgbs, all_traj_feature_maps, all_traj_gripper_pcds, all_traj_pcd_masks, \
        all_traj_stage_lengths, all_traj_store_label_paths
    
def extract_demos_from_a_directory(dirtory_path, object_category, exp_name=None, in_gripper_frame=False, parallel=True, 
                                    gripper_num_points=0, add_contact=False, save_path=None, smooth_reach_to_contact_stage=False):
    task_paths = os.listdir(dirtory_path)
    task_paths = sorted(task_paths)
    
    action_dist_save_path = os.path.join(save_path, "action_dist")
    if not os.path.exists(action_dist_save_path):
        os.makedirs(action_dist_save_path)
    demo_rgb_save_path = os.path.join(save_path, "demo_rgbs")
    if not os.path.exists(demo_rgb_save_path):
        os.makedirs(demo_rgb_save_path)

    all_pc_list = []
    all_state_list = []
    all_action_list = []
    all_feature_map_list = []
    all_gripper_pcd_list = []
    all_pcd_mask_list = []
    last_state_indices = []
    total_count = 0
    for task_path in task_paths[args.task_beg_idx:args.task_end_idx]:
        files_and_folders = os.listdir(os.path.join(dirtory_path, task_path))
        solution_path, task_config_path = None, None
        for file_or_folder in files_and_folders:
            if file_or_folder.startswith("task"):
                solution_path = os.path.join(dirtory_path, task_path, file_or_folder)
            if file_or_folder.endswith(".yaml"):
                task_config_path = os.path.join(dirtory_path, task_path, file_or_folder)
        if solution_path is None or task_config_path is None:
            print("No solution path or task config path found for task: ", task_path)
            continue

        # all_traj_pc, all_traj_pos_ori = extract_pc_states_for_one_trajectory(task_config_path, solution_path, object_category, in_gripper_frame=in_gripper_frame)
        all_traj_pc, all_traj_pos_ori, all_traj_rgbs, all_traj_feature_maps, all_traj_gripper_pcds, all_traj_pcd_masks, \
                all_traj_stage_lengths, all_traj_store_label_paths = extract_pc_states_for_all_trajectories(
            task_config_path, solution_path, object_category, exp_name=exp_name, 
            in_gripper_frame=in_gripper_frame, parallel=parallel,
            gripper_num_points=gripper_num_points, add_contact=add_contact, smooth_reach_to_contact_stage=smooth_reach_to_contact_stage)

        # TODO: update this to actually used exp path        
        
        all_demo_paths = []
        for traj_idx in tqdm.tqdm(range(len(all_traj_pc)), total=len(all_traj_pc)):

            traj_pc, traj_pos_ori, traj_feature_maps, traj_gripper_pcd, traj_pcd_masks = all_traj_pc[traj_idx], all_traj_pos_ori[traj_idx], all_traj_feature_maps[traj_idx], all_traj_gripper_pcds[traj_idx], all_traj_pcd_masks[traj_idx]
            traj_stage_length, traj_store_label_path  = all_traj_stage_lengths[traj_idx], all_traj_store_label_paths[traj_idx]
            
            good_traj = True

            traj_actions = []
            quaternion_diffs = []
            if not args.use_joint_angle:
                base_pos = traj_pos_ori[0][:3]
                base_ori_6d = traj_pos_ori[0][3:9]
                base_finger_angle = traj_pos_ori[0][9]
            
            open_door_start_idx = 0
            # NOTE: for open_door_per_angle_new.py the keys order are different
            if 'stage' not in traj_stage_length.keys():
                keys = ["reach_handle", "open_gripper", "reach_to_contact", "close_gripper"]
            else:
                keys = ['open_gripper', "grasp_handle", 'close_gripper'] if "open_gripper" in traj_stage_length['stage'] else ['grasp_handle', 'close_gripper']

            for key in keys:
                open_door_start_idx += traj_stage_length.get(key, 0)
            
            after_contact_step_idx = traj_stage_length['reach_handle'] + traj_stage_length['reach_to_contact']
        
            filtered_pcs = []
            filtered_pos_oris = []
            filtered_feature_maps = []
            filtered_gripper_pcds = []
            filtered_pcd_masks = []
            filtered_rgbs = []
            base_rgb = all_traj_rgbs[traj_idx][0]
            base_feature_map = traj_feature_maps[0]
            base_gripper_pcd = traj_gripper_pcd[0]
            base_pc = traj_pc[0]
            base_pcd_mask = traj_pcd_masks[0]
            base_pos_ori = traj_pos_ori[0]
            for i in range(len(traj_pos_ori) - 1):
                target_pos_ori = traj_pos_ori[i+1]

                if not args.use_joint_angle:                
                    cur_pos = traj_pos_ori[i][:3]
                    target_pos = traj_pos_ori[i+1][:3]

                    single_step_delta_pos = np.array(target_pos) - np.array(cur_pos)
                    
                    # if single step translation is too large, ignore this trajectory
                    if np.linalg.norm(single_step_delta_pos) > 0.02:
                        good_traj = False
                        print("not good traj due to delta movement too large")
                        break
                    
                    delta_pos = np.array(target_pos) - np.array(base_pos)

                    cur_ori_6d = traj_pos_ori[i][3:9]
                    
                    # change the delta_pos into gripper frame
                    if in_gripper_frame:
                        cur_mat = rotation_transfer_6D_to_matrix(cur_ori_6d)
                        delta_pos = cur_mat.T @ delta_pos                    

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
                    if np.abs(one_step_quaternion_diff) > 0.085:
                        good_traj = False
                        print("not good due to delta quaternion too large")
                        break
                    
                    if i > open_door_start_idx and np.abs(one_step_quaternion_diff) > 0.02: # open door has strange behavior
                        good_traj = False
                        print("not good due to delta quaternion too large during opening door")
                        break
                    
                    # cur_finger_angle = traj_pos_ori[i][9]
                    target_finger_angle = traj_pos_ori[i+1][9]

                    # delta_finger_angle = target_finger_angle - cur_finger_angle
                    delta_finger_angle = target_finger_angle - base_finger_angle
                
                filter_action = False
                if args.filter_small_action: 
                    if args.after_reaching or args.after_opening:
                        if np.linalg.norm(delta_pos) < args.min_translation and np.linalg.norm(quat_diff) < args.min_rotation and np.abs(delta_finger_angle) < args.min_finger_angle_diff:
                            filter_action = True
                    else:
                        if np.linalg.norm(delta_pos) < args.min_translation and np.linalg.norm(quat_diff) < args.min_rotation and np.abs(delta_finger_angle) < args.min_finger_angle_diff:
                            if args.filter_after_reaching and i > traj_stage_length["reach_handle"]:
                                filter_action = True
                            if not args.filter_after_reaching:
                                filter_action = True

                if filter_action:
                    continue
                else:
                    if not args.use_joint_angle:
                        action = delta_pos.tolist() + delta_ori_6d.tolist() + [delta_finger_angle]
                    else:
                        # traj_pos_ori is the normalized joint angle. Action is the delta change in normalized joint angle
                        action = np.array(target_pos_ori) - np.array(base_pos_ori) 
                        action = action[:-1] # action only controls one finger; the other one is symmetric
                        if i > after_contact_step_idx and args.fixed_finger_movement:
                            action[-1] = args.close_finger_angle # TODO: double check this value
                            
                        action = action.tolist()
                            
                    traj_actions.append(action)
                    filtered_pcs.append(base_pc)
                    filtered_pcd_masks.append(base_pcd_mask)
                    filtered_gripper_pcds.append(base_gripper_pcd)
                    filtered_feature_maps.append(base_feature_map)
                    filtered_pos_oris.append(base_pos_ori)
                    filtered_rgbs.append(base_rgb)
                    base_pc = traj_pc[i+1]
                    base_pcd_mask = traj_pcd_masks[i+1]
                    base_gripper_pcd = traj_gripper_pcd[i+1]
                    base_feature_map = traj_feature_maps[i+1]
                    base_pos_ori = traj_pos_ori[i+1]
                    base_rgb = all_traj_rgbs[traj_idx][i+1]
                    if not args.use_joint_angle:
                        base_pos = target_pos
                        base_ori_6d = target_ori_6d
                        base_finger_angle = target_finger_angle
                    
           
            # plot the delta translation action distribution
            if traj_idx % 5 == 0:        
                try:
                    save_numpy_as_gif(np.array(filtered_rgbs), os.path.join(demo_rgb_save_path, "demo_" + str(traj_idx) + ".gif"))
                    plt.close("all")

                    if not args.use_joint_angle:
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
                    else:
                        delta_joint_angles = np.linalg.norm(np.array(traj_actions), axis=1)
                        fig, axes = plt.subplots(1, 1, figsize=(6, 5))
                        axes.plot(range(len(delta_joint_angles)), delta_joint_angles, "-*")
                        keys = ["reach_handle", "reach_to_contact", "close_gripper", "open_door"]
                        base = 0
                        for key in keys:
                            base += traj_stage_length[key]
                            axes.axvline(x=base, color='r', linestyle='--')
                            axes.text(base, 0, key, rotation=90)
                        axes.set_title("delta_joint_angles")
                            
                    suffix = "good" if good_traj else "bad"
                    save_fig_path = os.path.join(action_dist_save_path, "delta_distribution_{}_{}.png".format(traj_idx, suffix))
                    plt.savefig(save_fig_path)
                    plt.close("all")
                except:
                    pass

            path = os.path.join(traj_store_label_path, "label.json")
            if not os.path.exists(path):
                with open(path, 'w') as f:
                    json.dump({"good_traj": good_traj}, f)

            if good_traj:
                all_pc_list = all_pc_list + filtered_pcs
                all_state_list = all_state_list + filtered_pos_oris
                all_feature_map_list = all_feature_map_list + filtered_feature_maps
                all_gripper_pcd_list = all_gripper_pcd_list + filtered_gripper_pcds
                all_pcd_mask_list = all_pcd_mask_list + filtered_pcd_masks
                all_action_list = all_action_list + traj_actions
                total_count += len(filtered_pcs)
                last_state_indices.append(deepcopy(total_count))
                all_demo_paths.append(os.path.dirname(traj_store_label_path))
                
    with open(os.path.join(save_path, "all_demo_path.txt"), "w") as f:
        f.write("\n".join(all_demo_paths))
    
    return all_pc_list, all_state_list, all_feature_map_list, all_gripper_pcd_list, all_pcd_mask_list, all_action_list, last_state_indices
        
def save_data(pc_list, state_list, feature_map_list, gripper_pcd_list, pcd_mask_list, action_list, last_state_indices, save_dir):
    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')

    state_arrays = np.stack(state_list, axis=0)
    point_cloud_arrays = np.stack(pc_list, axis=0)
    action_arrays = np.stack(action_list, axis=0)
    if 'act3d' in args.observation_mode:
        # import pdb; pdb.set_trace()
        feature_map_arrays = np.stack(feature_map_list, axis=0)
        gripper_pcd_arrays = np.stack(gripper_pcd_list, axis=0)
        pcd_mask_list = np.stack(pcd_mask_list, axis=0)
    episode_ends_arrays = np.array(last_state_indices)


    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    state_chunk_size = (100, state_arrays.shape[1])
    point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
    action_chunk_size = (100, action_arrays.shape[1])
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    if 'act3d' in args.observation_mode:
        # feature_map_chunk_size = (100, feature_map_arrays.shape[1], feature_map_arrays.shape[2], feature_map_arrays.shape[3])
        feature_map_chunk_size = (100, feature_map_arrays.shape[1], feature_map_arrays.shape[2], feature_map_arrays.shape[3], feature_map_arrays.shape[4]) # there can be mutiple cameras
        gripper_pcd_chunk_size = (100, gripper_pcd_arrays.shape[1], gripper_pcd_arrays.shape[2])
        pcd_mask_chunk_size = (100, pcd_mask_list.shape[1])
        zarr_data.create_dataset('feature_map', data=feature_map_arrays, chunks=feature_map_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('gripper_pcd', data=gripper_pcd_arrays, chunks=gripper_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('pcd_mask', data=pcd_mask_list, chunks=pcd_mask_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)

    del state_arrays, point_cloud_arrays, feature_map_arrays, gripper_pcd_arrays, action_arrays, episode_ends_arrays
    del zarr_root, zarr_data, zarr_meta

def save_example_pointcloud(pc_list, save_dir):
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
        plt.show()
        plt.close()


def main(folder_name, object_name, save_path, exp_name=None, in_gripper_frame=True, parallel=True,
         gripper_num_points=0, add_contact=False, smooth_reach_to_contact_stage=False):
    
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    meta_info = {
        "folder_name": folder_name,
        "in_gripper_frame": in_gripper_frame,
        "exp_name": exp_name,
    }
    meta_info.update(args.__dict__)
    with open(os.path.join(save_path, "meta_info.json"), "w") as f:
        json.dump(meta_info, f, indent=4)
    
    all_pc_list, all_state_list, all_feature_map_list, all_gripper_pcd_list, all_pcd_mask_list, \
        all_action_list, last_state_indices = extract_demos_from_a_directory(
        folder_name, object_name,exp_name=exp_name, in_gripper_frame=in_gripper_frame, parallel=parallel, 
        gripper_num_points=gripper_num_points, add_contact=add_contact, save_path=save_path, smooth_reach_to_contact=smooth_reach_to_contact_stage)
    
        
    # import pickle
    # with open(os.path.join(save_path, "raw_data.pkl"), "wb") as f:
    #     pickle.dump((pc_list, state_list, action_list, last_state_indices), f, protocol=pickle.HIGHEST_PROTOCOL)

    save_data(all_pc_list, all_state_list, all_feature_map_list, all_gripper_pcd_list, all_pcd_mask_list,
              all_action_list, last_state_indices, save_path)
    save_example_pointcloud(all_pc_list, save_path)


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--in_gripper_frame", type=int, default=0)
    args.add_argument("--gripper_num_points", type=int, default=0)
    args.add_argument("--add_contact", type=int, default=0)
    args.add_argument("--after_reaching", type=int, default=0)
    args.add_argument("--after_opening", type=int, default=0)
    args.add_argument("--filter_small_action", type=float, default=0)
    args.add_argument("--filter_after_reaching", type=float, default=0)
    args.add_argument("--min_translation", type=float, default=0.0045)
    args.add_argument("--min_rotation", type=float, default=0.008)
    args.add_argument("--min_finger_angle_diff", type=float, default=0.001)
    args.add_argument("--include_reaching_perturbation", type=int, default=0)
    args.add_argument("--include_open_perturbation", type=int, default=0)
    args.add_argument("--fixed_finger_movement", type=int, default=1)
    args.add_argument("--pointcloud_num", type=int, default=4500)
    args.add_argument("--use_joint_angle", type=int, default=0)
    args.add_argument("--use_segmask", type=int, default=0)
    args.add_argument("--only_handle_points", type=int, default=0)
    args.add_argument("--opened_threshold", type=float, default=0.65)
    args.add_argument("--close_finger_angle", type=float, default=-0.08)
    args.add_argument("--observation_mode", type=str, default='segmask')

    
    args.add_argument("--object_name", type=str, required=True)
    args.add_argument("--save_path", type=str, required=True)
    args.add_argument("--exp_name", type=str, default=None)
    args.add_argument("--folder_name", type=str, required=True)
    args.add_argument("--generate", type=int, default=1)
    args.add_argument("--parallel", type=int, default=1)
    args.add_argument("--task_beg_idx", type=int, default=0)
    args.add_argument("--task_end_idx", type=int, default=1)
    args.add_argument("--num_experiment", type=int, default=10000)
    args.add_argument("--num_worker", type=int, default=80)
    args.add_argument("--use_extracted", type=int, default=1)
    args.add_argument("--smooth_reach_to_contact", type=int, default=0)
    args = args.parse_args()
    
   

    if args.generate:
        if args.parallel:
            set_start_method('spawn', force=True)
            num_worker = args.num_worker
            pool = Pool(processes=num_worker)
        main(args.folder_name, args.object_name, args.save_path, exp_name=args.exp_name, 
             in_gripper_frame=args.in_gripper_frame, parallel=args.parallel, 
             gripper_num_points=args.gripper_num_points, add_contact=args.add_contact, smooth_reach_to_contact_stage=args.smooth_reach_to_contact)
    else:
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
                object_name, seed=0, in_gripper_frame=args.in_gripper_frame, 
                gripper_num_points=args.gripper_num_points, add_contact=args.add_contact, 
                num_points=args.pointcloud_num,
                use_joint_angle=args.use_joint_angle)
            
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

    # debug
    # exp_folder = "data/temp/nihao/2024-05-11-00-58-38/grasp_the_door_handle_primitive/states"
    # task_config_path = "data/temp/nihao/2024-05-11-00-58-38/task_config.yaml"
    # solution_path = "data/temp/nihao"
    # task_name = "grasp_the_door_handle"
    # states = sort_states_file_by_file_number(exp_folder)
    # states = [os.path.join(exp_folder, x) for x in states]
    # states = states[91:100]

    # ret_states = smooth_reach_to_contact_stage(states, task_config_path, solution_path, task_name, exp_folder)
    # print("ret_states: ", ret_states)
    # print("=========================")
    # print("states: ", states)

