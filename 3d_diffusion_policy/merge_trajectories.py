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
        if not traj_name.startswith('202'):
            continue
        
        wpt_groups = os.listdir(pkl_dir)
        wpt_groups = [x for x in wpt_groups if x.endswith('.pkl')]
        wpt_groups = sorted(wpt_groups, key = lambda x: int(x.split('.')[0]))
        traj_length = len(wpt_groups)
        print(pkl_dir)
        pickle_path = f'{pkl_dir}/{wpt_groups[0]}'
        data = pickle.load(open(pickle_path, 'rb'))
        original_goal_gripper_pcd = data['goal_gripper_pcd']
        
        goal_switch_point = 0
        goal_pcd = copy.deepcopy(original_goal_gripper_pcd[0])

        # find goal switching point and save raw data
        raw_wpt_data = [data]
        for wpt_id, wpt_group in enumerate(wpt_groups[1:]):

            pickle_path = f'{pkl_dir}/{wpt_group}'
            data = pickle.load(open(pickle_path, 'rb'))
            raw_wpt_data.append(data)

            if goal_switch_point == 0 and np.linalg.norm((data['goal_gripper_pcd'][0] - goal_pcd)) > 1e-3:
                goal_switch_point = wpt_id
        
        # combine action
        saved_data_list = []
        run_index = 0
        run_gripper_action = 0
        for wpt_id in range(traj_length):
            
            # keep dense steps around goal
            if (wpt_id > goal_switch_point - dense_steps_around_goal) and (wpt_id < goal_switch_point + dense_steps_around_goal):
                
                run_index = wpt_id
                
                saved_data_list.append({
                    'state': raw_wpt_data[wpt_id]['state'],
                    'point_cloud': raw_wpt_data[wpt_id]['point_cloud'],
                    'gripper_pcd': raw_wpt_data[wpt_id]['gripper_pcd'],
                    'goal_gripper_pcd': raw_wpt_data[wpt_id]['goal_gripper_pcd'],
                    'displacement_gripper_to_object': raw_wpt_data[wpt_id]['displacement_gripper_to_object']
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
            
            elif wpt_id % combine_step == 0:

                run_index = wpt_id
                
                saved_data_list.append({
                    'state': raw_wpt_data[wpt_id]['state'],
                    'point_cloud': raw_wpt_data[wpt_id]['point_cloud'],
                    'gripper_pcd': raw_wpt_data[wpt_id]['gripper_pcd'],
                    'goal_gripper_pcd': raw_wpt_data[wpt_id]['goal_gripper_pcd'],
                    'displacement_gripper_to_object': raw_wpt_data[wpt_id]['displacement_gripper_to_object']
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
            new_pickle_data_save_path = f'{target_dir}/{new_wpt_id}.pkl'
            with open(new_pickle_data_save_path, "wb") as f:
                pickle.dump(saved_data, f)
                cprint(f"Saving new data to: {new_pickle_data_save_path}", "green")

if __name__=="__main__":

    src_dirs = [
        # bucket
        'bucket_100435', 'bucket_100441', 'bucket_100443', 'bucket_100444', 'bucket_100452',
        'bucket_100454', 'bucket_100460', 'bucket_100461', 'bucket_100462', 'bucket_100469',
        'bucket_100472', 'bucket_102352', 'bucket_102358', 'bucket_102365',

        # faucet
        'faucet_148', 'faucet_149', 'faucet_152', 'faucet_153', 'faucet_154', 'faucet_168',
        'faucet_811', 'faucet_822', 'faucet_857', 'faucet_908', 'faucet_929', 'faucet_960',
        'faucet_991', 'faucet_1028', 'faucet_1052', 'faucet_1053', 'faucet_1288', 'faucet_1343',
        'faucet_1370', 'faucet_1466', 'faucet_1492', 'faucet_1528', 'faucet_1626', 'faucet_1633',
        'faucet_1646', 'faucet_1668', 'faucet_1741', 'faucet_1794', 'faucet_1795', 'faucet_1802',
        'faucet_1885', 'faucet_1901', 'faucet_1903', 'faucet_1925', 'faucet_1961', 'faucet_1986',
        'faucet_2054',

        # foldingchair
        'foldingchair_100520', 'foldingchair_100521', 'foldingchair_100526', 'foldingchair_100531',
        'foldingchair_100532', 'foldingchair_100557', 'foldingchair_100561', 'foldingchair_100562',
        'foldingchair_100568', 'foldingchair_100579', 'foldingchair_100586', 'foldingchair_100590',
        'foldingchair_100599', 'foldingchair_100600', 'foldingchair_100608', 'foldingchair_100609',
        'foldingchair_100611', 'foldingchair_100616', 'foldingchair_102255', 'foldingchair_102263',
        'foldingchair_102269', 'foldingchair_102314',

        # laptop
        'laptop_9748', 'laptop_9912', 'laptop_9960', 'laptop_9968', 'laptop_9992', 'laptop_9996',
        'laptop_10040', 'laptop_10098', 'laptop_10101', 'laptop_10238', 'laptop_10243', 'laptop_10248',
        'laptop_10269', 'laptop_10270', 'laptop_10280', 'laptop_10289', 'laptop_10305', 'laptop_10306',
        'laptop_10383', 'laptop_10626', 'laptop_10697', 'laptop_10885', 'laptop_10915', 'laptop_11075',
        'laptop_11156', 'laptop_11242', 'laptop_11248', 'laptop_11395', 'laptop_11405', 'laptop_11406',
        'laptop_11429', 'laptop_11477', 'laptop_11581', 'laptop_11586', 'laptop_11691', 'laptop_11778',
        'laptop_11876', 'laptop_11888', 'laptop_11945', 'laptop_12073',

        # stapler
        'stapler_102990', 'stapler_103095', 'stapler_103099', 'stapler_103100', 'stapler_103104',
        'stapler_103111', 'stapler_103113', 'stapler_103271', 'stapler_103275', 'stapler_103276',
        'stapler_103280', 'stapler_103292', 'stapler_103293', 'stapler_103297', 'stapler_103299',
        'stapler_103301', 'stapler_103303', 'stapler_103305', 'stapler_103789', 'stapler_103792',

        # toilet
        'toilet_101320', 'toilet_102620', 'toilet_102621', 'toilet_102622', 'toilet_102630',
        'toilet_102634', 'toilet_102645', 'toilet_102648', 'toilet_102651', 'toilet_102652',
        'toilet_102654', 'toilet_102658', 'toilet_102663', 'toilet_102666', 'toilet_102667',
        'toilet_102668', 'toilet_102669', 'toilet_102670', 'toilet_102675', 'toilet_102676',
        'toilet_102677', 'toilet_102687', 'toilet_102689', 'toilet_102692', 'toilet_102694',
        'toilet_102697', 'toilet_102699', 'toilet_102701', 'toilet_102703', 'toilet_102707',
        'toilet_102708', 'toilet_103234',

         # storagefurniture_tasks
        # '0622-act3d-obj-45448-remove-reaching-collision-resize-2-full-per-step-gripper-goal-displacement-to-closest-obj-point',
        # '0624-act3d-obj-46462-per-step-combine-2-action-gripper-goal-displacement-to-closest-obj-point-filtered-zero-closing-action',
        # '0626-act3d-obj-41510-per-step-combine-2-action-gripper-goal-displacement-to-closest-obj-point-filtered-zero-closing-action',
        # '0628-act3d-obj-46732-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1',
        # '0628-act3d-obj-46801-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1',
        # '0628-act3d-obj-46874-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1',
        # '0628-act3d-obj-46922-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1',
        # '0628-act3d-obj-46966-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1',
        # '0628-act3d-obj-47570-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1',
        # '0628-act3d-obj-47578-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1',
    ]   


    all_objs = os.listdir("data/dp3_demo/seuss_gen_random")
    all_objs = sorted(all_objs)
    src_dirs = all_objs

    dense_steps_around_goal = 0
    combine_step = 2
    
    src_dirs = [os.path.join("data/dp3_demo/seuss_gen_random", x) for x in src_dirs]
    dst_dirs = [x.replace("dp3_demo", "dp3_demo_combined_{}_step_{}".format(combine_step, dense_steps_around_goal)) for x in src_dirs]


    for src_dir, dst_dir in zip(src_dirs, dst_dirs):
        combine_action(
            src_dir=src_dir, 
            dst_dir=dst_dir, 
            combine_step=combine_step,
            dense_steps_around_goal=dense_steps_around_goal
        )
        