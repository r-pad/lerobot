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

import subprocess

def find_directories(path, search_string):
    try:
        # Construct the find command
        command = ['find', path, '-type', 'd', '-name', f'*{search_string}*']
        
        # Execute the command
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Get the output and split it into lines
        directories = result.stdout.splitlines()
        
        return directories
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e.stderr}")
        return []

zarr_path_list = os.listdir("/scratch/chialiang/dp3_demo")
zarr_path_list = sorted(zarr_path_list)
zarr_path_list = [os.path.join("/scratch/chialiang/dp3_demo", zarr_path) for zarr_path in zarr_path_list]
old_objects = [
    "41510",
    "45448",
    "46462",
    "46732",
    "46801",
    "46874",
    "46922",
    "46966",
    "47570",
    "47578",
    "48700"
]


for zarr_path in zarr_path_list:
    if 'copy' in zarr_path:
        continue

    obj_id = zarr_path.split('-')[-1]
    date = zarr_path.split('-')[0]
    if '0705' in date:
        demo_path = "data/diverse_objects/open_the_door_{}/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first".format(obj_id)
    else:
        demo_path = "data/diverse_objects_2/open_the_door_{}/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0712-diverse-objects-2-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first/".format(obj_id)
    if obj_id in old_objects:
        continue
    
    all_subfolder = os.listdir(zarr_path)
    for string in ["action_dist", "demo_rgbs", "all_demo_path.txt", "meta_info.json", 'example_pointcloud']:
        if string in all_subfolder:
            all_subfolder.remove(string)
            
    all_subfolder = sorted(all_subfolder)
    # print(zarr_path, len(all_subfolder))

    keys = ['state', 'action', 'point_cloud', ]
    keys += ['gripper_pcd', 'displacement_gripper_to_object', 'goal_gripper_pcd']
    combine_action_steps = 2

    for traj in all_subfolder:

        zarr_path_traj = os.path.join(zarr_path, traj)
        exp_name = traj
        experiment_folder = os.path.join(demo_path, exp_name)
        if not os.path.exists(experiment_folder):
            # this is for handling the case that some demo does not have the corresponding experiment
            # found_directories = find_directories("/mnt/RoboGen_sim2real/data/diverse_objects/", traj)
            # found_obj = found_directories[0][len("/mnt/RoboGen_sim2real/data/diverse_objects/open_the_door_"):len("/mnt/RoboGen_sim2real/data/diverse_objects/open_the_door_")+5]
            # exists = os.path.exists(os.path.join("/project_data/held/yufeiw2/RoboGen_sim2real/data/dp3_demo/0705-obj-{}/{}".format(found_obj, traj)))
            # print(zarr_path, traj)
            continue

        
        all_subfolder = os.listdir(experiment_folder)
        all_subfolder = [d for d in all_subfolder if os.path.isdir(os.path.join(demo_path, exp_name, d))]
        d = all_subfolder[0]
        opened_angle_file = os.path.join(demo_path, exp_name, d, 'opened_angle.txt')

        
        with open(opened_angle_file, 'r') as f:
            all_lines = f.readlines()
            opened_angle = float(all_lines[0].strip())
            initial_angle = float(all_lines[1].strip())
            max_angle = float(all_lines[2].strip())
        
        if opened_angle < max_angle * 0.2:
            cmd = 'rm -rf {}'.format(zarr_path_traj)
            print(cmd)
            os.system(cmd)
    
# demo have more data than experiments. 
# 45661, 47669, 48177, 49025
# they are actually from 
# 47235, 49025, 
    