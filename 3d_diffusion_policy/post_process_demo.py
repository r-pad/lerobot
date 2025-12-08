import zarr
import os
import numpy as np
import json
from matplotlib import pyplot as plt
from tqdm import tqdm
from manipulation.utils import rotation_transfer_6D_to_matrix, rotation_transfer_matrix_to_6D
from scipy.spatial.transform import Rotation as R
from termcolor import cprint

def save_data(pc_list, state_list, feature_map_list, gripper_pcd_list, pcd_mask_list, action_list, save_dir):
    
    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')

    state_arrays = np.array(state_list)
    point_cloud_arrays = np.array(pc_list)
    action_arrays = np.array(action_list)
    feature_map_arrays = np.array(feature_map_list)
    gripper_pcd_arrays = np.array(gripper_pcd_list)
    pcd_mask_list = np.array(pcd_mask_list)

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    state_chunk_size = (100, state_arrays.shape[1])
    point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
    action_chunk_size = (100, action_arrays.shape[1])
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    feature_map_chunk_size = (100, feature_map_arrays.shape[1], feature_map_arrays.shape[2], feature_map_arrays.shape[3], feature_map_arrays.shape[4]) # there can be mutiple cameras
    goal_gripper_pcd_chunk_size = (100, goal_gripper_pcd.shape[1], goal_gripper_pcd.shape[2])
    pcd_mask_chunk_size = (100, pcd_mask_list.shape[1])
    zarr_data.create_dataset('feature_map', data=feature_map_arrays, chunks=feature_map_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('gripper_pcd', data=gripper_pcd_arrays, chunks=gripper_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('pcd_mask', data=pcd_mask_list, chunks=pcd_mask_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
    if goal_gripper_pcd is not None:
        zarr_data.create_dataset('goal_gripper_pcd', data=goal_gripper_pcd, chunks=goal_gripper_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    if displacement_gripper_to_object is not None:
        zarr_data.create_dataset('displacement_gripper_to_object', data=displacement_gripper_to_object, chunks=goal_gripper_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)

    del state_arrays, point_cloud_arrays, feature_map_arrays, gripper_pcd_arrays, action_arrays
    del zarr_root, zarr_data, zarr_meta
    
def filter_traj(traj_feature_maps, traj_gripper_pcd, traj_pc, traj_pcd_masks, traj_pos_ori, traj_stage_length=None, 
                min_translation=0.002, min_rotation=0.005, min_finger_angle_diff=0.0008):
    
    traj_actions = []
    
    base_pos = traj_pos_ori[0][:3]
    base_ori_6d = traj_pos_ori[0][3:9]
    base_finger_angle = traj_pos_ori[0][9]
    base_feature_map = traj_feature_maps[0]
    base_gripper_pcd = traj_gripper_pcd[0]
    base_pc = traj_pc[0]
    base_pcd_mask = traj_pcd_masks[0]
    base_pos_ori = traj_pos_ori[0]
    
    filtered_pcs = []
    filtered_pos_oris = []
    filtered_feature_maps = []
    filtered_gripper_pcds = []
    filtered_pcd_masks = []
    traj_actions = []
    
    
    for i in range(len(traj_pos_ori) - 1):
        target_pos = traj_pos_ori[i+1][:3]
        delta_pos = np.array(target_pos) - np.array(base_pos)

        target_ori_6d = traj_pos_ori[i+1][3:9]
        base_ori_matrix = rotation_transfer_6D_to_matrix(base_ori_6d)
        target_ori_matrix = rotation_transfer_6D_to_matrix(target_ori_6d)

        delta_ori_matrix = base_ori_matrix.T @ target_ori_matrix
        delta_ori_6d = rotation_transfer_matrix_to_6D(delta_ori_matrix)
        
        # base_ori_quat =  R.from_matrix(base_ori_matrix).as_quat()
        # target_ori_quat = R.from_matrix(target_ori_matrix).as_quat()
        # quat_diff = np.arccos(2 * np.dot(base_ori_quat, target_ori_quat)**2 - 1)
        
        # if single step rotation is too large, ignore this trajectory
        target_finger_angle = traj_pos_ori[i+1][9]
        delta_finger_angle = target_finger_angle - base_finger_angle
        # change the finger action to be not changing after closing
        opening_start_idx = traj_stage_length['reach_handle'] + traj_stage_length["reach_to_contact"] + traj_stage_length["close_gripper"]
        closing_start_idx = traj_stage_length['reach_handle'] + traj_stage_length["reach_to_contact"]
        
        filter_action = False
        
        # if np.linalg.norm(delta_pos) < min_translation and np.linalg.norm(quat_diff) < min_rotation and i >= opening_start_idx:
        #     print("Filtered out action at step: ", i)
        #     filter_action = True
        # if i > closing_start_idx and i < opening_start_idx and i - closing_start_idx >= 14:
        #     print("Filtered out action at step: ", i)
        #     filter_action = True

        if filter_action:
            continue
        else:
            action = delta_pos.tolist() + delta_ori_6d.tolist() + [delta_finger_angle]
                    
            traj_actions.append(action)
            filtered_pcs.append(base_pc)
            filtered_pcd_masks.append(base_pcd_mask)
            filtered_gripper_pcds.append(base_gripper_pcd)
            filtered_feature_maps.append(base_feature_map)
            filtered_pos_oris.append(base_pos_ori)
            base_pc = traj_pc[i+1]
            base_pcd_mask = traj_pcd_masks[i+1]
            base_gripper_pcd = traj_gripper_pcd[i+1]
            base_feature_map = traj_feature_maps[i+1]
            base_pos_ori = traj_pos_ori[i+1]
            base_pos = target_pos
            base_ori_6d = target_ori_6d
            base_finger_angle = target_finger_angle
            
    # return filtered_pcs, filtered_pos_oris, filtered_feature_maps, filtered_gripper_pcds, filtered_pcd_masks, traj_actions
    return np.array(filtered_pcs), np.array(filtered_pos_oris), np.array(filtered_feature_maps), np.array(filtered_gripper_pcds), np.array(filtered_pcd_masks), np.array(traj_actions)

# demo_path = "data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46462_2024-03-27-23-35-10/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-4-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
# demo_path = "data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"

new_zarr_path = "/project_data/held/chialiak/RoboGen-sim2realdata/dp3_demo/0630-dp3-goal-whole"
zarr_path = "/project_data/held/chialiak/RoboGen-sim2realdata/dp3_demo/0622-act3d-obj-45448-reach-to-contact-smoothed"
demo_path = "/project_data/held/chialiak/RoboGen-sim2realdata/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_45448_2024-03-27-22-40-39/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-2-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"

# 1
new_zarr_path = "/project_data/held/chialiak/RoboGen-sim2realdata/dp3_demo/0630-dp3-goal-whole"
zarr_path = "/project_data/held/chialiak/RoboGen-sim2realdata/dp3_demo/0622-act3d-obj-45448-reach-to-contact-smoothed"
demo_path = "/project_data/held/chialiak/RoboGen-sim2realdata/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_45448_2024-03-27-22-40-39/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-2-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"

zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0701-act3d-obj-45448-remove-reaching-collision-resize-2-full-dp3_goal_gripper_whole"
demo_path = "/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_45448_2024-03-27-22-40-39/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-2-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
new_zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0703-dp3-goal-whole"

# zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0701-act3d-obj-45448-remove-reaching-collision-resize-2-full-dp3_goal_gripper_part"
# demo_path = "/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_45448_2024-03-27-22-40-39/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-2-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
# new_zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0703-dp3-goal-part"


all_subfolder = os.listdir(zarr_path)
for string in ["action_dist", "demo_rgbs", "all_demo_path.txt", "meta_info.json", 'example_pointcloud']:
    if string in all_subfolder:
        all_subfolder.remove(string)
all_subfolder = sorted(all_subfolder)
zarr_paths = [os.path.join(zarr_path, subfolder) for subfolder in all_subfolder]
path_list = zarr_paths

per_episode_root = []
keys = ['state', 'action', 'point_cloud']
# keys += ['feature_map', 'gripper_pcd', 'pcd_mask']
            
for zarr_path in tqdm(path_list, desc='Processing'):
    exp_name = zarr_path.split('/')[-1]
    stage_lengths_json_file = os.path.join(demo_path, exp_name, "grasp_the_door_handle_primitive", 'stage_lengths.json')
    # stage_lengths_json_file = os.path.join(demo_path, exp_name, "grasp_the_handle_of_the_drawer_primitive", 'stage_lengths.json')
    with open(stage_lengths_json_file, 'r') as f:
        stage_lengths = json.load(f)
    
    
    group = zarr.open(zarr_path, 'r')
    src_store = group.store

    # numpy backend
    src_root = zarr.group(src_store)
    meta = dict()
    
    for key, value in src_root['meta'].items():
        if len(value.shape) == 0:
            meta[key] = np.array(value)
        else:
            meta[key] = value[:]

    if keys is None:
        keys = src_root['data'].keys()
    data = dict()
    for key in keys:
        arr = src_root['data'][key]
        data[key] = arr[:]
        

    # filtered_pcs, filtered_pos_oris, filtered_feature_maps, filtered_gripper_pcds, filtered_pcd_masks, traj_actions = \
    #     filter_traj(data['feature_map'], data['gripper_pcd'], data['point_cloud'], data['pcd_mask'], data['state'], stage_lengths)

    # cprint("{} Filtered out {} actions".format(exp_name, len(data['action']) - len(traj_actions)), "red")
    
    # opening_start_idx = stage_lengths['reach_handle'] + stage_lengths["reach_to_contact"] + stage_lengths["close_gripper"]
    # for i in range(len(data['action'])):
    #     if i >= opening_start_idx and data['action'][i][-1] > 0:
    #         data['action'][i][-1] = 0.0
    
    contact_idx = stage_lengths['reach_handle'] + stage_lengths["reach_to_contact"]
    for key in keys:
        data[key] = np.concatenate([data[key][:contact_idx-2], data[key][contact_idx:]])
    
    filtered_pcs, filtered_pos_oris, filtered_feature_maps, filtered_gripper_pcds, filtered_pcd_masks, traj_actions = \
        filter_traj(data['feature_map'][::2], data['gripper_pcd'][::2], data['point_cloud'][::2], data['pcd_mask'][::2], data['state'][::2], stage_lengths)
    new_after_contact_idx = stage_lengths['reach_handle'] // 2 + (stage_lengths["reach_to_contact"] - 2) // 2
    for i in range(len(traj_actions)):
        if i >= new_after_contact_idx:
            traj_actions[i][-1] = -0.006
    
    # save new data
    new_data_save_dir = os.path.join(new_zarr_path, exp_name)
    print("Saving new data to: ", new_data_save_dir)
    save_data(filtered_pcs, filtered_pos_oris, filtered_feature_maps, filtered_gripper_pcds, filtered_pcd_masks, traj_actions, new_data_save_dir)
    # save_data(data['point_cloud'], data['state'], data['feature_map'], data['gripper_pcd'], data['pcd_mask'], data['action'], new_data_save_dir)