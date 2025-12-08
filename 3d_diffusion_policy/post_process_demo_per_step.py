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
import fpsample

def save_data(pc_list, state_list, feature_map_list, gripper_pcd_list, pcd_mask_list, action_list, 
              goal_gripper_pcd, 
              displacement_gripper_to_object,
              save_dir):

    state_arrays = np.array(state_list)
    point_cloud_arrays = np.array(pc_list)
    action_arrays = np.array(action_list)

    # print(state_arrays.shape)
    # print(point_cloud_arrays.shape)
    # print(action_arrays.shape)

    # out_path = '/project_data/held/chialiak/RoboGen-sim2real/pcd_from_feat_3.npy'
    # np.save(out_path, point_cloud_arrays[10])
    # print(f'{out_path} has been written')
    # exit(0)

    # feature_map_arrays = np.array(feature_map_list)
    # gripper_pcd_arrays = np.array(gripper_pcd_list)
    # pcd_mask_list = np.array(pcd_mask_list)

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    state_chunk_size = (100, state_arrays.shape[1])
    point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
    action_chunk_size = (100, action_arrays.shape[1])
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    # feature_map_chunk_size = (100, feature_map_arrays.shape[1], feature_map_arrays.shape[2], feature_map_arrays.shape[3], feature_map_arrays.shape[4]) # there can be mutiple cameras
    # gripper_pcd_chunk_size = (100, gripper_pcd_arrays.shape[1], gripper_pcd_arrays.shape[2])
    # goal_gripper_pcd_chunk_size = (100, goal_gripper_pcd.shape[1], goal_gripper_pcd.shape[2])
    # pcd_mask_chunk_size = (100, pcd_mask_list.shape[1])
    # zarr_data.create_dataset('feature_map', data=feature_map_arrays, chunks=feature_map_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    # zarr_data.create_dataset('gripper_pcd', data=gripper_pcd_arrays, chunks=gripper_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    # zarr_data.create_dataset('pcd_mask', data=pcd_mask_list, chunks=pcd_mask_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
    # if goal_gripper_pcd is not None:
    #     zarr_data.create_dataset('goal_gripper_pcd', data=goal_gripper_pcd, chunks=goal_gripper_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    # if displacement_gripper_to_object is not None:
    #     zarr_data.create_dataset('displacement_gripper_to_object', data=displacement_gripper_to_object, chunks=goal_gripper_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)

    cprint(f'{save_dir} saved', 'green')
    del state_arrays, point_cloud_arrays, action_arrays #, feature_map_arrays, gripper_pcd_arrays, action_arrays
    del zarr_root, zarr_data, zarr_meta

def save_data_per_step(pc_list, state_list, feature_map_list, gripper_pcd_list, pcd_mask_list, action_list, 
              goal_gripper_pcd, 
              displacement_gripper_to_object,
              save_dir):

    state_arrays = np.array(state_list)
    point_cloud_arrays = np.array(pc_list)
    action_arrays = np.array(action_list)
    # feature_map_arrays = np.array(feature_map_list)
    # gripper_pcd_arrays = np.array(gripper_pcd_list)
    # pcd_mask_list = np.array(pcd_mask_list)
    
    chunk_size = 1
    state_chunk_size = (chunk_size, state_arrays.shape[1])
    point_cloud_chunk_size = (chunk_size, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
    action_chunk_size = (chunk_size, action_arrays.shape[1])
    # feature_map_chunk_size = (chunk_size, feature_map_arrays.shape[1], feature_map_arrays.shape[2], feature_map_arrays.shape[3], feature_map_arrays.shape[4]) # there can be mutiple cameras
    # feature_map_chunk_size = (chunk_size, feature_map_arrays.shape[1], feature_map_arrays.shape[2]) # there can be mutiple cameras
    # gripper_pcd_chunk_size = (chunk_size, gripper_pcd_arrays.shape[1], gripper_pcd_arrays.shape[2])
    # pcd_mask_chunk_size = (chunk_size, pcd_mask_list.shape[1])
    # if goal_gripper_pcd is not None:
    #     goal_gripper_pcd_chunk_size = (chunk_size, goal_gripper_pcd.shape[1], goal_gripper_pcd.shape[2])
    # if displacement_gripper_to_object is not None:
    #     displacement_gripper_to_object_chunk_size = (chunk_size, displacement_gripper_to_object.shape[1], displacement_gripper_to_object.shape[2])
    
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
        # zarr_data.create_dataset('feature_map', data=feature_map_arrays[t_idx][None, :], chunks=feature_map_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        # zarr_data.create_dataset('gripper_pcd', data=gripper_pcd_arrays[t_idx][None, :], chunks=gripper_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        # zarr_data.create_dataset('pcd_mask', data=pcd_mask_list[t_idx][None, :], chunks=pcd_mask_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
        # if goal_gripper_pcd is not None:
        #     zarr_data.create_dataset('goal_gripper_pcd', data=goal_gripper_pcd[t_idx][None, :], chunks=goal_gripper_pcd_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        # if displacement_gripper_to_object is not None:
        #     zarr_data.create_dataset('displacement_gripper_to_object', data=displacement_gripper_to_object[t_idx][None, :], chunks=displacement_gripper_to_object_chunk_size, dtype='float32', overwrite=True, compressor=compressor)

    # del state_arrays, point_cloud_arrays, feature_map_arrays, gripper_pcd_arrays, action_arrays
    # del zarr_root, zarr_data, zarr_meta
    # del state_arrays, point_cloud_arrays, gripper_pcd_arrays, action_arrays
    del state_arrays, point_cloud_arrays, action_arrays
    del zarr_root, zarr_data, zarr_meta
    if goal_gripper_pcd is not None:
        del goal_gripper_pcd
    if displacement_gripper_to_object is not None:
        del displacement_gripper_to_object
    
def filter_traj(traj_feature_maps, traj_gripper_pcd, traj_pc, traj_pcd_masks, traj_pos_ori, 
                goal_gripper_pcd, displacement_gripper_to_object,
                after_contact_idx,
                opening_start_idx, 
                filter_close_zero_action,
                min_translation=0.002, min_rotation=0.005, min_finger_angle_diff=0.001):
    
    traj_actions = []
    
    base_pos = traj_pos_ori[0][:3]
    base_ori_6d = traj_pos_ori[0][3:9]
    base_finger_angle = traj_pos_ori[0][9]
    base_pc = traj_pc[0]
    base_pos_ori = traj_pos_ori[0]
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
    
def filter_traj(traj_feature_maps, traj_gripper_pcd, traj_pc, traj_pcd_masks, traj_pos_ori, 
                goal_gripper_pcd, displacement_gripper_to_object,
                after_contact_idx,
                opening_start_idx, 
                filter_close_zero_action,
                min_translation=0.002, min_rotation=0.005, min_finger_angle_diff=0.001):
    
    traj_actions = []
    
    base_pos = traj_pos_ori[0][:3]
    base_ori_6d = traj_pos_ori[0][3:9]
    base_finger_angle = traj_pos_ori[0][9]
    base_feature_map = traj_feature_maps[0]
    base_gripper_pcd = traj_gripper_pcd[0]
    base_pc = traj_pc[0]
    base_pcd_mask = traj_pcd_masks[0]
    base_pos_ori = traj_pos_ori[0]
    if goal_gripper_pcd is not None:
        base_goal_gripper_pcd = goal_gripper_pcd[0]
    if displacement_gripper_to_object is not None:
        base_displacement_gripper_to_object = displacement_gripper_to_object[0]
    
    # base_pos = traj_pos_ori[0][:3]
    # base_ori_6d = traj_pos_ori[0][3:9]
    # base_finger_angle = traj_pos_ori[0][9]
    # base_gripper_pcd = traj_gripper_pcd[0]
    # base_pc = traj_pc[0]
    # base_pcd_mask = traj_pcd_masks[0]
    # base_pos_ori = traj_pos_ori[0]
    # if goal_gripper_pcd is not None:
    #     base_goal_gripper_pcd = goal_gripper_pcd[0]
    # if displacement_gripper_to_object is not None:
    #     base_displacement_gripper_to_object = displacement_gripper_to_object[0]
    
    filtered_pcs = []
    filtered_pos_oris = []
    filtered_feature_maps = []
    filtered_gripper_pcds = []
    filtered_pcd_masks = []
    filtered_goal_gripper_pcds = [] if goal_gripper_pcd is not None else None
    filtered_displacement_gripper_to_object = [] if displacement_gripper_to_object is not None else None
    traj_actions = []
    
    
    for i in range(len(traj_pos_ori) - 1):
        target_pos = traj_pos_ori[i+1][:3]
        delta_pos = np.array(target_pos) - np.array(base_pos)

        target_ori_6d = traj_pos_ori[i+1][3:9]
        base_ori_matrix = rotation_transfer_6D_to_matrix(base_ori_6d)
        target_ori_matrix = rotation_transfer_6D_to_matrix(target_ori_6d)

        delta_ori_matrix = base_ori_matrix.T @ target_ori_matrix
        delta_ori_6d = rotation_transfer_matrix_to_6D(delta_ori_matrix)
        
        target_finger_angle = traj_pos_ori[i+1][9]
        delta_finger_angle = target_finger_angle - base_finger_angle
        filter_action = False
        
        if i >= after_contact_idx and i < opening_start_idx:
            if np.abs(delta_finger_angle) < min_finger_angle_diff and filter_close_zero_action:
                filter_action = True
        
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
            if goal_gripper_pcd is not None:
                filtered_goal_gripper_pcds.append(base_goal_gripper_pcd)
            if displacement_gripper_to_object is not None:
                filtered_displacement_gripper_to_object.append(base_displacement_gripper_to_object)

            base_pc = traj_pc[i+1]
            base_pcd_mask = traj_pcd_masks[i+1]
            base_gripper_pcd = traj_gripper_pcd[i+1]
            base_feature_map = traj_feature_maps[i+1]
            base_pos_ori = traj_pos_ori[i+1]
            if goal_gripper_pcd is not None:
                base_goal_gripper_pcd = goal_gripper_pcd[i+1]
            if displacement_gripper_to_object is not None:
                base_displacement_gripper_to_object = displacement_gripper_to_object[i+1]

            base_pos = target_pos
            base_ori_6d = target_ori_6d
            base_finger_angle = target_finger_angle
    
    if goal_gripper_pcd is not None:
        filtered_goal_gripper_pcds = np.array(filtered_goal_gripper_pcds)
    if displacement_gripper_to_object is not None:
        filtered_displacement_gripper_to_object = np.array(filtered_displacement_gripper_to_object)
        
    # return np.array(filtered_pcs), np.array(filtered_pos_oris), np.array(traj_actions)
    return np.array(filtered_pcs), np.array(filtered_pos_oris), np.array(filtered_feature_maps), np.array(filtered_gripper_pcds), \
        np.array(filtered_pcd_masks), filtered_goal_gripper_pcds, filtered_displacement_gripper_to_object, np.array(traj_actions)


# # store goal grippre pcd and gripper distance to closest object point
# new_zarr_path = "/scratch/yufei/dp3_demo/0623-act3d-obj-45448-reach-to-contact-smoothed-per-step-combine-2-action-gripper-goal-displacement-to-closest-obj-point"
# zarr_path = "data/dp3_demo/0622-act3d-obj-45448-reach-to-contact-smoothed"
# demo_path = "data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_45448_2024-03-27-22-40-39/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-2-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"

# new_zarr_path = "/scratch/yufei/dp3_demo/0624-act3d-obj-46462-per-step-combine-2-action-gripper-goal-displacement-to-closest-obj-point-filtered-zero-closing-action"
# zarr_path = "data/dp3_demo/0531-act3d-obj-46462"
# demo_path = "data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46462_2024-03-27-23-35-10/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-4-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"

# new_zarr_path = "/scratch/yufei/dp3_demo/0626-act3d-obj-41510-per-step-combine-2-action-gripper-goal-displacement-to-closest-obj-point-filtered-zero-closing-action"
# zarr_path = "data/dp3_demo/0527-act3d-always-close"
# demo_path = "data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"

# zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0701-act3d-obj-45448-remove-reaching-collision-resize-2-full-dp3_goal_gripper_whole"
# demo_path = "/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_45448_2024-03-27-22-40-39/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-2-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
# new_zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0703-dp3-goal-whole"

# zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0701-act3d-obj-45448-remove-reaching-collision-resize-2-full-dp3_goal_gripper_part"
# demo_path = "/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_45448_2024-03-27-22-40-39/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-2-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
# new_zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0703-dp3-goal-part"

# zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0703-act3d-obj-45448-remove-reaching-collision-resize-2-full-dp3_goal_gripper_dense"
# demo_path = "/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_45448_2024-03-27-22-40-39/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-2-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
# new_zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0703-dp3-goal-dense"

# 1: 46732
zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0628-act3d-obj-46732"
demo_path = "/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46732_2024-03-27-18-46-00/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
new_zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0707-dp3-obj-46732-goal_gripper_on_agent"

# 2: 46801
zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0628-act3d-obj-46801"
demo_path = "/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46801_2024-03-27-20-37-05/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
new_zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0707-dp3-obj-46801-goal_gripper_on_agent"

# 3: 46874
zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0628-act3d-obj-46874"
demo_path = "/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46874_2024-03-27-13-57-49/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
new_zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0707-dp3-obj-46874-goal_gripper_on_agent"

# 4: 46922
zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0628-act3d-obj-46922"
demo_path = "/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46922_2024-03-27-19-42-45/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
new_zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0707-dp3-obj-46922-goal_gripper_on_agent"

# 5: 46966
zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0628-act3d-obj-46966"
demo_path = "/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46966_2024-03-27-16-55-33/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
new_zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0707-dp3-obj-46966-goal_gripper_on_agent"

# 6: 47570
zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0628-act3d-obj-47570"
demo_path = "/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_47570_2024-03-27-21-36-50/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
new_zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0707-dp3-obj-47570-goal_gripper_on_agent"

# 7: 47578
zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0628-act3d-obj-47578"
demo_path = "/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_47578_2024-03-27-14-56-07/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
new_zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0707-dp3-obj-47578-goal_gripper_on_agent"
 
# 8: 48700
zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/data/dp3_demo/0628-act3d-obj-48700"
demo_path = "/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_48700_2024-03-27-12-59-58/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
new_zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0707-dp3-obj-48700-goal_gripper_on_agent"

# ######################################

# # 0: 41510
# zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0705-dp3-obj-41510-goal_dense_gripper_on_pcd"
# demo_path = "/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_41510_2024-03-27-15-59-54/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
# new_zarr_path = "0706-dp3-obj-41510-goal_dense_gripper_on_pcd"

# # 1: 45448
# zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0705-dp3-obj-45448-goal_dense_gripper_on_pcd"
# demo_path = "/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_45448_2024-03-27-22-40-39/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-2-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
# new_zarr_path = "0706-dp3-obj-45448-goal_dense_gripper_on_pcd"

# # 2: 46462
# zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0705-dp3-obj-46462-goal_dense_gripper_on_pcd"
# demo_path = "/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46462_2024-03-27-23-35-10/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-4-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
# new_zarr_path = "0706-dp3-obj-46462-goal_dense_gripper_on_pcd"

# # 3: 46732
# zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0705-dp3-obj-46732-goal_dense_gripper_on_pcd"
# demo_path = "/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46732_2024-03-27-18-46-00/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
# new_zarr_path = "0706-dp3-obj-46732-goal_dense_gripper_on_pcd"

# # 4: 46801
# zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0705-dp3-obj-46801-goal_dense_gripper_on_pcd"
# demo_path = "/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46801_2024-03-27-20-37-05/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
# new_zarr_path = "0706-dp3-obj-46801-goal_dense_gripper_on_pcd"

# # 5: 46874
# zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0705-dp3-obj-46874-goal_dense_gripper_on_pcd"
# demo_path = "/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46874_2024-03-27-13-57-49/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
# new_zarr_path = "0706-dp3-obj-46874-goal_dense_gripper_on_pcd"

# # 6: 46922
# zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0705-dp3-obj-46922-goal_dense_gripper_on_pcd"
# demo_path = "/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46922_2024-03-27-19-42-45/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
# new_zarr_path = "0706-dp3-obj-46922-goal_dense_gripper_on_pcd"

# # 7: 46966
# zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0705-dp3-obj-46966-goal_dense_gripper_on_pcd"
# demo_path = "/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46966_2024-03-27-16-55-33/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
# new_zarr_path = "0706-dp3-obj-46966-goal_dense_gripper_on_pcd"

# # 8: 47570
# zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0705-dp3-obj-47570-goal_dense_gripper_on_pcd"
# demo_path = "/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_47570_2024-03-27-21-36-50/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
# new_zarr_path = "0706-dp3-obj-47570-goal_dense_gripper_on_pcd"

# # 9: 47578
# zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0705-dp3-obj-47578-goal_dense_gripper_on_pcd"
# demo_path = "/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_47578_2024-03-27-14-56-07/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
# new_zarr_path = "0706-dp3-obj-47578-goal_dense_gripper_on_pcd"

# # 10: 48700
# zarr_path = "/project_data/held/chialiak/RoboGen-sim2real/dp3_demo/0705-dp3-obj-48700-goal_dense_gripper_on_pcd"
# demo_path = "/project_data/held/chialiak/RoboGen-sim2real/data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_48700_2024-03-27-12-59-58/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0627-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
# new_zarr_path = "0706-dp3-obj-48700-goal_dense_gripper_on_pcd"



import argparse
args = argparse.ArgumentParser()
args.add_argument('--zarr_path', type=str, default=zarr_path)
args.add_argument('--demo_path', type=str, default=demo_path)
args.add_argument('--new_zarr_path', type=str, default=new_zarr_path)
args.add_argument('--add_gripper_goal_obs', type=int, default=True)
args.add_argument('--add_gripper_distance_to_closest_point', type=int, default=True)
# args.add_argument('--add_gripper_goal_obs', type=int, default=False)
# args.add_argument('--add_gripper_distance_to_closest_point', type=int, default=False)
args.add_argument('--combine_action_steps', type=int, default=2)
args.add_argument('--remove_collision', type=int, default=False)
args.add_argument('--filter_close_zero_action', type=int, default=True)
args = args.parse_args()

zarr_path = args.zarr_path
demo_path = args.demo_path
new_zarr_path = args.new_zarr_path
add_gripper_goal_obs = args.add_gripper_goal_obs
add_gripper_distance_to_closest_point = args.add_gripper_distance_to_closest_point
combine_action_steps = args.combine_action_steps
remove_collision = args.remove_collision
filter_close_zero_action = args.filter_close_zero_action

all_subfolder = os.listdir(zarr_path)
for string in ["action_dist", "demo_rgbs", "all_demo_path.txt", "meta_info.json", 'example_pointcloud']:
    if string in all_subfolder:
        all_subfolder.remove(string)
        
all_subfolder = sorted(all_subfolder)
zarr_paths = [os.path.join(zarr_path, subfolder) for subfolder in all_subfolder]
path_list = zarr_paths

per_episode_root = []
# keys = ['state', 'action', 'point_cloud']
keys = ['state', 'action', 'point_cloud']
keys += ['feature_map', 'gripper_pcd', 'pcd_mask']

# add_gripper_goal_obs = True
# add_gripper_distance_to_closest_point = True
# combine_action_steps = 2
# remove_collision = True
# filter_close_zero_action = True



for zarr_path in tqdm(path_list, desc='Processing'):
    exp_name = zarr_path.split('/')[-1]
    all_subfolder = os.listdir(os.path.join(demo_path, exp_name))
    all_subfolder = [d for d in all_subfolder if os.path.isdir(os.path.join(demo_path, exp_name, d))]
    d = all_subfolder[0]
    stage_lengths_json_file = os.path.join(demo_path, exp_name, d, 'stage_lengths.json')
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
        open_begin_t_idx = stage_lengths['reach_handle'] + stage_lengths["reach_to_contact"] + stage_lengths["close_gripper"]
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
        
    if remove_collision:
        contact_idx = stage_lengths['reach_handle'] + stage_lengths["reach_to_contact"]
        for key in keys:
            data[key] = np.concatenate([data[key][:contact_idx-2], data[key][contact_idx:]])
        
    if combine_action_steps > 1:
        if not remove_collision:
            new_after_contact_idx = (stage_lengths['reach_handle'] + stage_lengths["reach_to_contact"]) // combine_action_steps
            new_opening_start_idx = (stage_lengths['reach_handle'] + stage_lengths["reach_to_contact"] + stage_lengths["close_gripper"]) // combine_action_steps
        else:
            new_after_contact_idx = (stage_lengths['reach_handle'] + stage_lengths["reach_to_contact"] - 2) // combine_action_steps
            new_opening_start_idx = (stage_lengths['reach_handle'] + stage_lengths["reach_to_contact"] - 2 + stage_lengths["close_gripper"]) // combine_action_steps
          
    # #################
    # # For DP3 based #
    # #################
        
    #     filtered_pcs, filtered_pos_oris, traj_actions = filter_traj(None, 
    #                     None, 
    #                     data['point_cloud'][::combine_action_steps], 
    #                     None, 
    #                     data['state'][::combine_action_steps], 
    #                     None,
    #                     None,
    #                     after_contact_idx=new_after_contact_idx,
    #                     opening_start_idx=new_opening_start_idx,
    #                     filter_close_zero_action=filter_close_zero_action)
        
    #     data['point_cloud'] = filtered_pcs
    #     data['state'] = filtered_pos_oris
    #     data['action'] = traj_actions
          
    #     for i in range(len(traj_actions)):
    #         # if i < new_after_contact_idx:
    #         #     traj_actions[i][-1] = 0.006
    #         if i >= new_after_contact_idx:
    #             traj_actions[i][-1] = -0.006

    # # save new data
    # new_data_save_dir = os.path.join(new_zarr_path, exp_name)
    # print("Saving new data to: ", new_data_save_dir)
    # save_data(data['point_cloud'], data['state'], None, None, None, data['action'], 
    #           goal_gripper_pcd, 
    #           displacement_gripper_to_object,
    #           new_data_save_dir)

    ###################
    # For Act3D based #
    ###################

        filtered_pcs, filtered_pos_oris, filtered_feature_maps, filtered_gripper_pcds, filtered_pcd_masks, \
            filtered_goal_gripper_pcd, filtered_displacement_gripper_to_object, traj_actions = filter_traj_for_act3d(data['feature_map'][::combine_action_steps], 
                        data['gripper_pcd'][::combine_action_steps], 
                        data['point_cloud'][::combine_action_steps], 
                        data['pcd_mask'][::combine_action_steps], 
                        data['state'][::combine_action_steps], 
                        goal_gripper_pcd[::combine_action_steps],
                        displacement_gripper_to_object[::combine_action_steps],
                        after_contact_idx=new_after_contact_idx,
                        opening_start_idx=new_opening_start_idx,
                        filter_close_zero_action=filter_close_zero_action)

        # # reduced
        # filtered_feature_maps_reduced = []
        # for filtered_pc, filtered_feature_map, filtered_pcd_mask in zip(filtered_pcs, filtered_feature_maps, filtered_pcd_masks):
        #     N,H,W,C = filtered_feature_map.shape
        #     filtered_feature_map_flat = filtered_feature_map.reshape((-1, C))
        #     filtered_feature_map_reduced_flat = filtered_feature_map_flat[filtered_pcd_mask == 1]
        #     filtered_feature_maps_reduced.append(filtered_feature_map_reduced_flat)
        # filtered_feature_maps_stacked = np.stack(filtered_feature_maps_reduced, axis=0)
        
        data['feature_map'] = filtered_feature_maps
        # data['feature_map'] = filtered_feature_maps_reduced
        data['gripper_pcd'] = filtered_gripper_pcds
        # data['point_cloud'] = filtered_pcs
        data['pcd_mask'] = filtered_pcd_masks
        # data['state'] = filtered_pos_oris
        data['action'] = traj_actions
        goal_gripper_pcd = filtered_goal_gripper_pcd
        displacement_gripper_to_object = filtered_displacement_gripper_to_object

        # Chialiang for dp3 goal on agent pos

        new_states = []
        new_pcds = []
        for (filtered_pos_ori, goal_gpcd, gpcd, feature_map, pcd_mask) in zip(filtered_pos_oris, filtered_goal_gripper_pcd, filtered_gripper_pcds, filtered_feature_maps, filtered_pcd_masks):

            diff = goal_gpcd - gpcd
            diff_flat = diff.reshape(-1)

            new_state = np.concatenate((filtered_pos_ori, diff_flat), axis=0)
            new_states.append(new_state)

            original_feature_map_flat = feature_map.reshape(-1, 5)
            pcd_objrobot = np.zeros(original_feature_map_flat.shape[0]).astype(np.uint8)
            pcd_objrobot[np.where(original_feature_map_flat[:,0] == 1)] = 1
            pcd_objrobot[np.where(original_feature_map_flat[:,1] == 1)] = 1
            new_pcd = original_feature_map_flat[pcd_objrobot == 1][:,2:]

            h = min(9, np.log2(4500))
            kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(new_pcd[:, :3], 4500, h=h)
            new_pcd = new_pcd[kdline_fps_samples_idx]
            new_pcds.append(new_pcd)

        new_states = np.stack(new_states, axis=0)
        new_pcds = np.stack(new_pcds, axis=0)
        data['state'] = new_states
        data['point_cloud'] = new_pcds

        for i in range(len(traj_actions)):
            # if i < new_after_contact_idx:
            #     traj_actions[i][-1] = 0.006
            if i >= new_after_contact_idx:
                traj_actions[i][-1] = -0.006

    # save new data
    new_data_save_dir = os.path.join(new_zarr_path, exp_name)
    print("Saving new data to: ", new_data_save_dir)
    save_data(data['point_cloud'], data['state'], data['feature_map'], data['gripper_pcd'], data['pcd_mask'], data['action'], 
              goal_gripper_pcd, 
              displacement_gripper_to_object,
              new_data_save_dir)
    # save_data_per_step(data['point_cloud'], data['state'], data['feature_map'], data['gripper_pcd'], data['pcd_mask'], data['action'], 
    #           goal_gripper_pcd, 
    #           displacement_gripper_to_object,
    #           new_data_save_dir)
