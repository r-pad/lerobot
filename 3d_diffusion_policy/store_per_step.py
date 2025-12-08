import zarr
import os
import numpy as np
import json
from matplotlib import pyplot as plt
from tqdm import tqdm
from manipulation.utils import rotation_transfer_6D_to_matrix, rotation_transfer_matrix_to_6D
from scipy.spatial.transform import Rotation as R
from termcolor import cprint
import scipy

def save_data(pc_list, state_list, feature_map_list, gripper_pcd_list, pcd_mask_list, action_list, 
              goal_gripper_pcd, 
              displacement_gripper_to_object,
              save_dir):

    state_arrays = np.array(state_list)
    point_cloud_arrays = np.array(pc_list)
    action_arrays = np.array(action_list)
    feature_map_arrays = np.array(feature_map_list)
    gripper_pcd_arrays = np.array(gripper_pcd_list)
    pcd_mask_list = np.array(pcd_mask_list)
    
    chunk_size = 1
    state_chunk_size = (chunk_size, state_arrays.shape[1])
    point_cloud_chunk_size = (chunk_size, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
    action_chunk_size = (chunk_size, action_arrays.shape[1])
    feature_map_chunk_size = (chunk_size, feature_map_arrays.shape[1], feature_map_arrays.shape[2], feature_map_arrays.shape[3], feature_map_arrays.shape[4]) # there can be mutiple cameras
    gripper_pcd_chunk_size = (chunk_size, gripper_pcd_arrays.shape[1], gripper_pcd_arrays.shape[2])
    pcd_mask_chunk_size = (chunk_size, pcd_mask_list.shape[1])
    if goal_gripper_pcd is not None:
        goal_gripper_pcd_chunk_size = (chunk_size, goal_gripper_pcd.shape[1], goal_gripper_pcd.shape[2])
    if displacement_gripper_to_object is not None:
        displacement_gripper_to_object_chunk_size = (chunk_size, displacement_gripper_to_object.shape[1], displacement_gripper_to_object.shape[2])
    
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    
    traj_len = len(state_list)
    for t_idx in range(traj_len):
        step_save_dir = os.path.join(save_dir, str(t_idx))
        if not os.path.exists(step_save_dir):
            os.makedirs(step_save_dir)
            
        zarr_root = zarr.group(step_save_dir)
        zarr_data = zarr_root.create_group('data')
        zarr_meta = zarr_root.create_group('meta')
        zarr_data.create_dataset('state', data=state_arrays[t_idx][None, :], chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('point_cloud', data=point_cloud_arrays[t_idx][None, :], chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('action', data=action_arrays[t_idx][None, :], chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('feature_map', data=feature_map_arrays[t_idx][None, :], chunks=feature_map_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('gripper_pcd', data=gripper_pcd_arrays[t_idx][None, :], chunks=gripper_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('pcd_mask', data=pcd_mask_list[t_idx][None, :], chunks=pcd_mask_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
        if goal_gripper_pcd is not None:
            zarr_data.create_dataset('goal_gripper_pcd', data=goal_gripper_pcd[t_idx][None, :], chunks=goal_gripper_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        if displacement_gripper_to_object is not None:
            zarr_data.create_dataset('displacement_gripper_to_object', data=displacement_gripper_to_object[t_idx][None, :], chunks=displacement_gripper_to_object_chunk_size, dtype='float32', overwrite=True, compressor=compressor)

    del state_arrays, point_cloud_arrays, feature_map_arrays, gripper_pcd_arrays, action_arrays
    del zarr_root, zarr_data, zarr_meta
    if goal_gripper_pcd is not None:
        del goal_gripper_pcd
    if displacement_gripper_to_object is not None:
        del displacement_gripper_to_object
    
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

new_zarr_path = "data/dp3_demo/0616-act3d-obj-41510-remove-reaching-collision-resize-2-per-step"
zarr_path = "data/dp3_demo/0607-act3d-obj-41510-remove-reaching-collision-resize-2"

# store goal gripper pcd
new_zarr_path = "/scratch/yufei/dp3_demo/0616-act3d-obj-45448-remove-reaching-collision-resize-2-full-per-step-gripper-goal"
zarr_path = "data/dp3_demo/0607-act3d-obj-45448-remove-reaching-collision-resize-2-full"
demo_path = "data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_45448_2024-03-27-22-40-39/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-2-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"

# store goal grippre pcd and gripper distance to closest object point
# new_zarr_path = "/scratch/yufei/dp3_demo/0622-act3d-obj-45448-remove-reaching-collision-resize-2-full-per-step-gripper-goal-displacement-to-closest-obj-point"
new_zarr_path = "/scratch/yufei/dp3_demo/0622-act3d-obj-45448-remove-reaching-collision-resize-2-full-per-step-gripper-goal-displacement-to-closest-obj-point"
zarr_path = "data/dp3_demo/0607-act3d-obj-45448-remove-reaching-collision-resize-2-full"
demo_path = "data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_45448_2024-03-27-22-40-39/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-2-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"

all_subfolder = os.listdir(zarr_path)
for string in ["action_dist", "demo_rgbs", "all_demo_path.txt", "meta_info.json", 'example_pointcloud']:
    if string in all_subfolder:
        all_subfolder.remove(string)
        
all_subfolder = sorted(all_subfolder)
zarr_paths = [os.path.join(zarr_path, subfolder) for subfolder in all_subfolder]
path_list = zarr_paths

per_episode_root = []
keys = ['state', 'action', 'point_cloud']
keys += ['feature_map', 'gripper_pcd', 'pcd_mask']

add_gripper_goal_obs = True
add_gripper_distance_to_closest_point = True

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
        
    if add_gripper_goal_obs:
        open_begin_t_idx = stage_lengths['reach_handle'] // 2 + (stage_lengths["reach_to_contact"] - 2) // 2 + stage_lengths["close_gripper"] // 2
        goal_gripper_pcd_1 = data['gripper_pcd'][open_begin_t_idx]
        goal_gripper_pcd_2 = data['gripper_pcd'][-1] # NOTE: ideally this should be the last gripper pcd that is in contact with the handle
        goal_gripper_pcd = np.zeros((len(data['gripper_pcd']), 4, 3)).astype(data['gripper_pcd'].dtype)
        goal_gripper_pcd[:open_begin_t_idx] = goal_gripper_pcd_1
        goal_gripper_pcd[open_begin_t_idx:] = goal_gripper_pcd_2
    else:
        goal_gripper_pcd = None
        
    if add_gripper_distance_to_closest_point:
        # compute the distance between the gripper and the closest point on the object
        # object point cloud
        displacement_gripper_to_object = np.zeros((len(data['gripper_pcd']), 4, 3)).astype(data['gripper_pcd'].dtype)
        for t in range(len(data['gripper_pcd'])):
            gripper_pcd = data['gripper_pcd'][t]
            object_pcd = data['point_cloud'][t]
            distance = scipy.spatial.distance.cdist(gripper_pcd, object_pcd)
            min_distance_obj_idx = np.argmin(distance, axis=1)
            closest_point = object_pcd[min_distance_obj_idx]
            displacement = closest_point - gripper_pcd
            assert displacement.shape == (4, 3)
            displacement_gripper_to_object[t] = displacement
    else:
        displacement_gripper_to_object = None

    # save new data
    new_data_save_dir = os.path.join(new_zarr_path, exp_name)
    print("Saving new data to: ", new_data_save_dir)
    save_data(data['point_cloud'], data['state'], data['feature_map'], data['gripper_pcd'], data['pcd_mask'], data['action'], 
              goal_gripper_pcd, 
              displacement_gripper_to_object,
              new_data_save_dir)