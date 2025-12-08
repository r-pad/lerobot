import zarr
import os
import numpy as np
import json
from matplotlib import pyplot as plt
from tqdm import tqdm

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
    gripper_pcd_chunk_size = (100, gripper_pcd_arrays.shape[1], gripper_pcd_arrays.shape[2])
    pcd_mask_chunk_size = (100, pcd_mask_list.shape[1])
    zarr_data.create_dataset('feature_map', data=feature_map_arrays, chunks=feature_map_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('gripper_pcd', data=gripper_pcd_arrays, chunks=gripper_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('pcd_mask', data=pcd_mask_list, chunks=pcd_mask_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)

    del state_arrays, point_cloud_arrays, feature_map_arrays, gripper_pcd_arrays, action_arrays
    del zarr_root, zarr_data, zarr_meta

new_zarr_path = "data/dp3_demo/0527-act3d-always-close/"
zarr_path = "data/dp3_demo/0527-act3d/"
demo_path = "data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
all_subfolder = os.listdir(zarr_path)
for string in ["action_dist", "demo_rgbs", "all_demo_path.txt", "meta_info.json"]:
    all_subfolder.remove(string)
all_subfolder = sorted(all_subfolder)
zarr_paths = [os.path.join(zarr_path, subfolder) for subfolder in all_subfolder]
path_list = zarr_paths

per_episode_root = []
keys = ['state', 'action', 'point_cloud']
keys += ['feature_map', 'gripper_pcd', 'pcd_mask']
            
for zarr_path in tqdm(path_list, desc='Processing'):
    exp_name = zarr_path.split('/')[-1]
    stage_lengths_json_file = os.path.join(demo_path, exp_name, "grasp_the_door_handle_primitive", 'stage_lengths.json')
    with open(stage_lengths_json_file, 'r') as f:
        stage_lengths = json.load(f)
    open_time_idx = stage_lengths['reach_handle'] + stage_lengths["reach_to_contact"]
    
    
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
        
    # change the finger action to be always close after contact
    data['action'][open_time_idx:, -1] = -0.003
    
    # save new data
    new_data_save_dir = os.path.join(new_zarr_path, exp_name)
    print("Saving new data to: ", new_data_save_dir)
    save_data(data['point_cloud'], data['state'], data['feature_map'], data['gripper_pcd'], data['pcd_mask'], data['action'], new_data_save_dir)
    # import pdb; pdb.set_trace()