import os, glob
import numpy as np
import open3d as o3d
import copy
import pickle
from termcolor import cprint
from tqdm import tqdm

def rotation_transfer_matrix_to_6D(rotate_matrix):
    if type(rotate_matrix) == list or type(rotate_matrix) == tuple:
        rotate_matrix = np.array(rotate_matrix, dtype=np.float64).reshape(3, 3)
    rotate_matrix = rotate_matrix.reshape(3, 3)
    
    a1 = rotate_matrix[:, 0]
    a2 = rotate_matrix[:, 1]

    orient = np.array([a1, a2], dtype=np.float64).flatten()
    return orient

def rotation_transfer_6D_to_matrix(orient):
    if type(orient) == list or type(orient) == tuple:
        orient = np.array(orient, dtype=np.float64)

    orient = orient.reshape(2, 3)
    a1 = orient[0]
    a2 = orient[1]

    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(a2, b1) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)

    rotate_matrix = np.array([b1, b2, b3], dtype=np.float64).T

    return rotate_matrix

def rotation_transfer_6D_to_matrix_batch(orient):

    # orient shape = (B, 6)
    # return shape = (3, B * 3)

    if type(orient) == list or type(orient) == tuple:
        orient = np.array(orient, dtype=np.float64)
    
    assert orient.shape[-1] == 6

    orient = orient.reshape(-1, 2, 3)
    a1 = orient[:,0]
    a2 = orient[:,1]

    b1 = a1 / np.linalg.norm(a1, axis=-1).reshape(-1,1)
    b2 = a2 - (np.sum(a2*b1, axis=-1).reshape(-1,1) * b1)
    b2 = b2 / np.linalg.norm(b2, axis=-1).reshape(-1,1)
    b3 = np.cross(b1, b2)

    rotate_matrix = np.hstack((b1, b2, b3))
    rotate_matrix = rotate_matrix.reshape(-1, 3).T

    return rotate_matrix

def rotation_transfer_matrix_to_6D_batch(rotate_matrix):

    # rotate_matrix.shape = (B, 9) or (B, 3, 3)
    # return shape = (B, 6)

    if type(rotate_matrix) == list or type(rotate_matrix) == tuple:
        rotate_matrix = np.array(rotate_matrix, dtype=np.float64).reshape(-1, 9)
    rotate_matrix = rotate_matrix.reshape(-1, 9)

    return rotate_matrix[:,:6]

def add_pose(pose, size=0.1):

    assert len(pose) == 10

    trans = np.identity(4)
    trans[:3, 3] = pose[:3]
    trans[:3,:3] = rotation_transfer_6D_to_matrix(pose[3:9])
    coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    coor.transform(trans)

    return coor

def add_sphere(pts, color=[1, 0, 0], radius=0.02):
    rets = []
    for pt in pts:
        pt_coor = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        pt_coor.translate(pt.reshape(-1, 1))
        pt_coor.paint_uniform_color(color)
        rets.append(pt_coor)

    return rets

def add_pcd(pts, color=[0,0,1]):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(np.full(pts.shape, color))

    return pcd


def combine_action(src_dir='one_traj/45526_pkl', dst_dir='one_traj/45526_pkl_post', combine_step=2, dense_steps_around_goal=3):

    assert os.path.exists(src_dir), f'{src_dir} not exists'
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)
    
    pkl_dirs = glob.glob(f'{src_dir}/*')

    for pkl_dir in pkl_dirs:

        traj_name = pkl_dir.split('/')[-1]
        
        wpt_groups = os.listdir(pkl_dir)
        wpt_groups = [x for x in wpt_groups if x.endswith('.npz')]
        wpt_groups = sorted(wpt_groups, key = lambda x: int(x.split('.')[0]))
        traj_length = len(wpt_groups)
        print(pkl_dir)
        pickle_path = f'{pkl_dir}/{wpt_groups[0]}'
        data = np.load(pickle_path)
        original_goal_gripper_pcd = data['goal_gripper_pcd']
        
        goal_pcd = copy.deepcopy(original_goal_gripper_pcd[0])

        # find goal switching point and save raw data
        raw_wpt_data = [data]
        for wpt_id, wpt_group in enumerate(wpt_groups[1:]):

            pickle_path = f'{pkl_dir}/{wpt_group}'
            data = np.load(pickle_path)
            raw_wpt_data.append(data)

        
        # combine action
        saved_data_list = []
        run_index = 0
        run_gripper_action = 0
        for wpt_id in range(traj_length):
            if wpt_id % combine_step == 0:

                run_index = wpt_id
                
                saved_data_list.append({
                    'state': raw_wpt_data[wpt_id]['state'],
                    'point_cloud': raw_wpt_data[wpt_id]['point_cloud'],
                    'gripper_pcd': raw_wpt_data[wpt_id]['gripper_pcd'],
                    'goal_gripper_pcd': raw_wpt_data[wpt_id]['goal_gripper_pcd'],
                })


                if len(saved_data_list) <= 1:
                    run_gripper_action = raw_wpt_data[wpt_id]['action'][0, 9]
                    continue

                last_state = saved_data_list[-2]['state']
                current_state = saved_data_list[-1]['state']

                last_rot_mat = rotation_transfer_6D_to_matrix(last_state[0, 3:9])
                current_rot_mat = rotation_transfer_6D_to_matrix(current_state[0, 3:9])
                delta_rot_mat = last_rot_mat.T @ current_rot_mat
                delta_rot_6d = rotation_transfer_matrix_to_6D(delta_rot_mat)

                action = np.zeros((1, 10))
                action[0, :3] = current_state[0, :3] - last_state[0, :3]
                action[0, 3:9] = delta_rot_6d
                action[0, 9] = run_gripper_action
                run_gripper_action = raw_wpt_data[wpt_id]['action'][0, 9]

                # last action will be computed in the current state
                saved_data_list[-2]['action'] = action
            
            else:

                run_gripper_action += raw_wpt_data[wpt_id]['action'][0, 9]

        # for last action only
        last_action = raw_wpt_data[run_index]['action']
        run_gripper_action = raw_wpt_data[run_index]['action'][0, 9]
        last_delta_rot_mat = rotation_transfer_6D_to_matrix(last_action[0, 3:9])
        for last_few_wpt_id in range(run_index + 1, traj_length):
            
            # position
            last_action[0, :3] += raw_wpt_data[last_few_wpt_id]['action'][0, :3]

            # rotation
            current_delta_rot_mat = rotation_transfer_6D_to_matrix(raw_wpt_data[last_few_wpt_id]['action'][0, 3:9])
            last_delta_rot_mat = last_delta_rot_mat @ current_delta_rot_mat

            # gripper
            run_gripper_action += raw_wpt_data[last_few_wpt_id]['action'][0, 9]
            
        last_action[0, 3:9] = rotation_transfer_matrix_to_6D(last_delta_rot_mat)
        last_action[0, 9] = run_gripper_action
        saved_data_list[-1]['action'] = last_action
        
        # save data
        for new_wpt_id, saved_data in enumerate(saved_data_list):
            target_dir = f'{dst_dir}/{traj_name}'
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
            new_pickle_data_save_path = f'{target_dir}/{new_wpt_id}.npz'
            np.savez_compressed(new_pickle_data_save_path, **saved_data)
            

if __name__=="__main__":

    src_dirs = ['plate']

    all_objs = os.listdir("data/aloha")
    all_objs = sorted(all_objs)
    src_dirs = all_objs

    dense_steps_around_goal = 0
    combine_step = 2
    
    src_dirs = [os.path.join("data/aloha", x) for x in src_dirs]
    dst_dirs = [x.replace("aloha", "aloha_combined_{}_step_{}".format(combine_step, dense_steps_around_goal)) for x in src_dirs]


    for src_dir, dst_dir in zip(src_dirs, dst_dirs):
        combine_action(
            src_dir=src_dir, 
            dst_dir=dst_dir, 
            combine_step=combine_step,
            dense_steps_around_goal=dense_steps_around_goal
        )
        