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

def save_data(pc_list, state_list, gripper_pcd_list, action_list, 
              goal_gripper_pcd, 
              displacement_gripper_to_object,
              save_dir):

    state_arrays = np.array(state_list)
    point_cloud_arrays = np.array(pc_list)
    action_arrays = np.array(action_list)
    gripper_pcd_arrays = np.array(gripper_pcd_list)
    
    chunk_size = 1
    state_chunk_size = (chunk_size, state_arrays.shape[1])
    point_cloud_chunk_size = (chunk_size, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
    action_chunk_size = (chunk_size, action_arrays.shape[1])
    gripper_pcd_chunk_size = (chunk_size, gripper_pcd_arrays.shape[1], gripper_pcd_arrays.shape[2])
    goal_gripper_pcd_chunk_size = (chunk_size, goal_gripper_pcd.shape[1], goal_gripper_pcd.shape[2])
    displacement_gripper_to_object_chunk_size = (chunk_size, displacement_gripper_to_object.shape[1], displacement_gripper_to_object.shape[2])
    
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
            
    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('gripper_pcd', data=gripper_pcd_arrays, chunks=gripper_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('goal_gripper_pcd', data=goal_gripper_pcd, chunks=goal_gripper_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('displacement_gripper_to_object', data=displacement_gripper_to_object, chunks=displacement_gripper_to_object_chunk_size, dtype='float32', overwrite=True, compressor=compressor)

    del state_arrays, point_cloud_arrays, gripper_pcd_arrays, action_arrays
    del pc_list, state_list, gripper_pcd_list, action_list
    del zarr_root, zarr_data, zarr_meta
    del goal_gripper_pcd
    del displacement_gripper_to_object
    

zarr_path = "/scratch/yufeiw2/0705-obj-41510"
demo_path = "data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"

all_subfolder = os.listdir(zarr_path)
for string in ["action_dist", "demo_rgbs", "all_demo_path.txt", "meta_info.json", 'example_pointcloud']:
    if string in all_subfolder:
        all_subfolder.remove(string)
        
all_subfolder = sorted(all_subfolder)

keys = ['state', 'action', 'point_cloud', ]
keys += ['gripper_pcd', 'displacement_gripper_to_object', 'goal_gripper_pcd']
combine_action_steps = 2

# read each step of each episode
all_trajs = os.listdir(zarr_path)
all_trajs = sorted(all_trajs)

for traj in tqdm(all_trajs, desc='processing'):

    zarr_path_traj = os.path.join(zarr_path, traj)
    exp_name = traj
    all_subfolder = os.listdir(os.path.join(demo_path, exp_name))
    all_subfolder = [d for d in all_subfolder if os.path.isdir(os.path.join(demo_path, exp_name, d))]
    d = all_subfolder[0]
    stage_lengths_json_file = os.path.join(demo_path, exp_name, d, 'stage_lengths.json')
    with open(stage_lengths_json_file, 'r') as f:
        stage_lengths = json.load(f)
    new_opening_start_idx = (stage_lengths['reach_handle'] + stage_lengths["reach_to_contact"] + stage_lengths["close_gripper"]) // combine_action_steps    
    
    all_steps = os.listdir(zarr_path_traj)
    all_steps = sorted(all_steps, key=lambda x: int(x))
    
    last_step = all_steps[-1]
    zarr_path_step = os.path.join(zarr_path_traj, last_step)
    group = zarr.open(zarr_path_step, 'r')
    src_store = group.store
    src_root = zarr.group(src_store)
    goal_gripper_pcd_arr = src_root['data']["gripper_pcd"][:]

    # for every step after opening, change the goal gripper pcd
    for step in all_steps[new_opening_start_idx:]:
        zarr_path_step = os.path.join(zarr_path_traj, step)
        group = zarr.open(zarr_path_step, 'r')
        src_store = group.store

        # numpy backend
        src_root = zarr.group(src_store)
        if keys is None:
            keys = src_root['data'].keys()
        data = dict()
        for key in keys:
            arr = src_root['data'][key]
            data[key] = arr[:]
            
        data['goal_gripper_pcd'] = goal_gripper_pcd_arr
    
        # remove the old data
        cmd = "rm -r " + zarr_path_step
        os.system(cmd)
    
        # save new data
        new_data_save_dir = zarr_path_step
        # print("Saving new data to: ", new_data_save_dir)
        save_data(
                data['point_cloud'], 
                data['state'], 
                data['gripper_pcd'], 
                data['action'], 
                data['goal_gripper_pcd'], 
                data['displacement_gripper_to_object'],
                new_data_save_dir
        )