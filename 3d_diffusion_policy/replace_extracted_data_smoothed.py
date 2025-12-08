from manipulation.utils import load_env, build_up_env, rotation_transfer_6D_to_matrix, rotation_transfer_matrix_to_6D
from manipulation.robogen_wrapper import RobogenPointCloudWrapper
import numpy as np
import os
import zarr
import json
from extract_data_from_states_smoothed import sort_states_file_by_file_number, smooth_reach_to_contact_stage
import tqdm

# sort zarr_file folders
def sort_zarr_files(zarr_files):
    sorted_zarr_files = []
    for zarr_file in zarr_files:
        if zarr_file.startswith("2024-05"):
            sorted_zarr_files.append(zarr_file)
    # zarr_files are named as 2024-05-11-hour-minute-second
    # sort by date and time
    sorted_zarr_files.sort()
    return sorted_zarr_files

def render_small_part_of_data(states, task_config_path, solution_path, task_name):
    simulator, _ = build_up_env(
                task_config=task_config_path,
                solution_path=solution_path,
                task_name=task_name,
                restore_state_file=None,
                render=False,
                randomize=False,
                obj_id=0,
    )
    
    env = RobogenPointCloudWrapper(simulator, 
        'storagefurniture', rpy_mean_list=[[0, 0, -45], [0, 0, -135]], seed=0, in_gripper_frame=0, 
        gripper_num_points=0, add_contact=0, num_points=4500,
        use_joint_angle=0, use_segmask=0, only_handle_points=0, 
        observation_mode='act3d')
    
    observations = []
    for state in states:
        load_env(simulator, load_path=state)
        observation = env._get_observation()
        observations.append(observation)
    
    simulator.close()
    return observations


zarr_file_root_path = "data/dp3_demo/0531-act3d-obj-45448"
save_zarr_file_root_path = "data/dp3_demo/0622-act3d-obj-45448-reach-to-contact-smoothed"
zarr_files = os.listdir(zarr_file_root_path)
zarr_files = sort_zarr_files(zarr_files)

solution_path = "data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_45448_2024-03-27-22-40-39/task_open_the_door_of_the_storagefurniture_by_its_handle"
task_name = "grasp_the_door_handle"
whole_experiment_folder = "data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_45448_2024-03-27-22-40-39/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-2-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"

for zarr_file in tqdm.tqdm(zarr_files):
    zarr_file_path = os.path.join(zarr_file_root_path, zarr_file)
    group = zarr.open(zarr_file_path, 'r')
    src_store = group.store
    src_root = zarr.group(src_store)
    data = dict()
    for key in src_root['data'].keys():
        arr = src_root['data'][key]
        data[key] = arr[:]
    
    # get to the experiment folder
    experiment_folder = os.path.join(whole_experiment_folder, zarr_file)
    task_config_file = os.path.join(experiment_folder, "task_config.yaml")
    stage_lengths = json.load(open(os.path.join(experiment_folder, task_name+"_primitive" ,"stage_lengths.json"), 'r'))
    all_state_files = sort_states_file_by_file_number(os.path.join(experiment_folder, task_name+"_primitive", "states"))
    all_state_files = [os.path.join(experiment_folder, task_name+"_primitive", "states", state_file) for state_file in all_state_files]
    beg_index = stage_lengths['reach_handle']
    end_index = stage_lengths['reach_handle'] + stage_lengths['reach_to_contact']
    ret_states = smooth_reach_to_contact_stage(all_state_files[beg_index:end_index], task_config_path=task_config_file, solution_path=solution_path, task_name=task_name, exp_folder=os.path.join(experiment_folder, task_name+"_primitive", "states"))
    
    observations = render_small_part_of_data(ret_states, task_config_path=task_config_file, solution_path=solution_path, task_name=task_name)
    pc_list = [x['point_cloud'].tolist() for x in observations]
    pos_ori_list = [x['agent_pos'].tolist() for x in observations]
    feature_map_list = [x['feature_map'].tolist() for x in observations]
    gripper_pcd_list = [x['gripper_pcd'].tolist() for x in observations]
    pcd_mask_list = [x['pcd_mask'].tolist() for x in observations]

    actions = []
    for i in range(len(ret_states) - 1):
        cur_pos = pos_ori_list[i][:3]
        target_pos = pos_ori_list[i+1][:3]

        delta_pos = np.array(target_pos) - np.array(cur_pos)

        cur_ori_6d = pos_ori_list[i][3:9]
        target_ori_6d = pos_ori_list[i+1][3:9]
        cur_ori_matrix = rotation_transfer_6D_to_matrix(cur_ori_6d)
        target_ori_matrix = rotation_transfer_6D_to_matrix(target_ori_6d)

        delta_ori_matrix = cur_ori_matrix.T @ target_ori_matrix
        delta_ori_6d = rotation_transfer_matrix_to_6D(delta_ori_matrix)

        target_finger_angle = pos_ori_list[i+1][9]
        base_finger_angle = pos_ori_list[i][9]
        delta_finger_angle = target_finger_angle - base_finger_angle

        action = delta_pos.tolist() + delta_ori_6d.tolist() + [delta_finger_angle]
        actions.append(action)

    smoothed_actions = np.array(actions)
    smoothed_pc = np.array(pc_list)
    smoothed_pos_ori = np.array(pos_ori_list)
    smoothed_feature_map = np.array(feature_map_list)
    smoothed_gripper_pcd = np.array(gripper_pcd_list)
    smoothed_pcd_mask = np.array(pcd_mask_list)

    data['action'][beg_index:end_index-1] = smoothed_actions
    data['point_cloud'][beg_index:end_index-1] = smoothed_pc[:-1]
    data['state'][beg_index:end_index-1] = smoothed_pos_ori[:-1]
    data['feature_map'][beg_index:end_index-1] = smoothed_feature_map[:-1]
    data['gripper_pcd'][beg_index:end_index-1] = smoothed_gripper_pcd[:-1]
    data['pcd_mask'][beg_index:end_index-1] = smoothed_pcd_mask[:-1]

    zarr_root = zarr.group(os.path.join(save_zarr_file_root_path, zarr_file))
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')
    # save the smoothed data
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    state_chunk_size = (100, data['state'].shape[1])
    point_cloud_chunk_size = (100, data['point_cloud'].shape[1], data['point_cloud'].shape[2])
    action_chunk_size = (100, data['action'].shape[1])
    feature_map_chunk_size = (100, data['feature_map'].shape[1], data['feature_map'].shape[2], data['feature_map'].shape[3], data['feature_map'].shape[4]) # there can be mutiple cameras
    gripper_pcd_chunk_size = (100, data['gripper_pcd'].shape[1], data['gripper_pcd'].shape[2])
    pcd_mask_chunk_size = (100, data['pcd_mask'].shape[1])
    zarr_data.create_dataset('state', data=data['state'], chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('point_cloud', data=data['point_cloud'], chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=data['action'], chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('feature_map', data=data['feature_map'], chunks=feature_map_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('gripper_pcd', data=data['gripper_pcd'], chunks=gripper_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('pcd_mask', data=data['pcd_mask'], chunks=pcd_mask_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
