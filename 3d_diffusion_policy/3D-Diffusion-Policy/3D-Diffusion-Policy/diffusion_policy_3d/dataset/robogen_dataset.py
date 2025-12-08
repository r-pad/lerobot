from typing import Dict
import torch
import time
import numpy as np
import copy
import os
from tqdm import tqdm
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.sampler import (get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.dataset.Augmentations.aug_translation_xy import TranslationXY
from diffusion_policy_3d.dataset.Augmentations.aug_rotation_z import rotationZ
from diffusion_policy_3d.dataset.Augmentations.random_apply_numpy import RandomApplyNumpy
from termcolor import cprint
import random
import copy
from test_PointNet2.all_data import *
from scripts.datasets.randomize_partition_50_obj import *
from scripts.datasets.randomize_partition_100_obj import *
from scripts.datasets.randomize_partition_200_obj import *

import pybullet as p
from manipulation.utils import get_pc, get_pc_in_camera_frame, rotation_transfer_6D_to_matrix_batch, rotation_transfer_matrix_to_6D_batch, add_sphere, get_pixel_location, get_matrix_from_pos_rot
from diffuser_actor_3d.robogen_utils import gripper_pcd_to_10d_vector

articulated_new = [
    # Bucket
    "bucket_100443", "bucket_100444", "bucket_100452", "bucket_100454", "bucket_100460", "bucket_100461",
    "bucket_100462", "bucket_100469", "bucket_100472", "bucket_102352", "bucket_102358", "bucket_102365",

    # Faucet
    "faucet_148", "faucet_152", "faucet_153", "faucet_154", "faucet_168", "faucet_811", "faucet_822",
    "faucet_857", "faucet_908", "faucet_929", "faucet_1028", "faucet_1052", "faucet_1053", "faucet_1288",
    "faucet_1343", "faucet_1370", "faucet_1466", "faucet_1492", "faucet_1528", "faucet_1626", "faucet_1633",
    "faucet_1646", "faucet_1668", "faucet_1741", "faucet_1794", "faucet_1795", "faucet_1802", "faucet_1885",
    "faucet_1901", "faucet_1903", "faucet_1925", "faucet_1961", "faucet_1986", "faucet_2054",

    # Foldingchair
    "foldingchair_100531", "foldingchair_100532", "foldingchair_100557", "foldingchair_100561",
    "foldingchair_100562", "foldingchair_100568", "foldingchair_100579", "foldingchair_100586",
    "foldingchair_100590", "foldingchair_100599", "foldingchair_100600", "foldingchair_100608",
    "foldingchair_100609", "foldingchair_100611", "foldingchair_100616", "foldingchair_102255",
    "foldingchair_102263", "foldingchair_102269", "foldingchair_102314",

    # Laptop
    "laptop_9968", "laptop_9992", "laptop_9996", "laptop_10040", "laptop_10098", "laptop_10101",
    "laptop_10238", "laptop_10243", "laptop_10248", "laptop_10269", "laptop_10270", "laptop_10280",
    "laptop_10289", "laptop_10305", "laptop_10306", "laptop_10383", "laptop_10626", "laptop_10697",
    "laptop_10885", "laptop_10915", "laptop_11075", "laptop_11156", "laptop_11242", "laptop_11248",
    "laptop_11395", "laptop_11405", "laptop_11406", "laptop_11429", "laptop_11477", "laptop_11581",
    "laptop_11586", "laptop_11691", "laptop_11778", "laptop_11876", "laptop_11888", "laptop_11945",
    "laptop_12073",

    # Stapler
    "stapler_103099", "stapler_103100", "stapler_103104", "stapler_103111", "stapler_103113",
    "stapler_103271", "stapler_103275", "stapler_103276", "stapler_103280", "stapler_103292",
    "stapler_103293", "stapler_103297", "stapler_103299", "stapler_103301", "stapler_103303",
    "stapler_103305", "stapler_103789", "stapler_103792",

    # Toilet
    "toilet_102622", "toilet_102630", "toilet_102634", "toilet_102645", "toilet_102648",
    "toilet_102651", "toilet_102652", "toilet_102654", "toilet_102658", "toilet_102663",
    "toilet_102666", "toilet_102667", "toilet_102668", "toilet_102669", "toilet_102670",
    "toilet_102675", "toilet_102676", "toilet_102677", "toilet_102687", "toilet_102689",
    "toilet_102692", "toilet_102694", "toilet_102697", "toilet_102699", "toilet_102701",
    "toilet_102703", "toilet_102707", "toilet_102708", "toilet_103234"
    ]

def get_zarry_paths(zarr_path):
    # dataset_prefix = '/mnt/RoboGen_sim2real/data/dp3_demo_combined_2_step_0/165-obj'
    # dataset_prefix = "/project_data/held/chenyuah/RoboGen-sim2real/data/dp3_demo/165-obj"
    dataset_prefix = "/scratch/yufeiw2/dp3_demo_combined_2_step_0"

    if zarr_path == 'test_1':
        data_name = [save_data_name_0]
        all_zarr_paths = [
            "{}/{}".format(dataset_prefix, data_name[i]) for i in range(len(data_name))
        ]
    if zarr_path == 'articulated':
        data_name = [
            # save_data_name_0, save_data_name_1, save_data_name_2, save_data_name_3, save_data_name_4, 
            save_data_name_5, save_data_name_6, save_data_name_7, save_data_name_8, save_data_name_9,
            save_data_name_10, save_data_name_11, save_data_name_12, save_data_name_13, save_data_name_14, save_data_name_15, save_data_name_16, save_data_name_17, save_data_name_18, save_data_name_19,
            save_data_name_20, save_data_name_21, save_data_name_22, save_data_name_23, save_data_name_24, save_data_name_25, save_data_name_26, save_data_name_27, save_data_name_28, save_data_name_29,
            save_data_name_30, save_data_name_31, save_data_name_32, save_data_name_33, save_data_name_34, save_data_name_35, save_data_name_36, save_data_name_37, save_data_name_38, save_data_name_39,
            save_data_name_40, save_data_name_41, save_data_name_42, save_data_name_43, save_data_name_44, save_data_name_45, save_data_name_46, save_data_name_47, save_data_name_48, save_data_name_49,
            
        ] + articulated_new
        all_zarr_paths = [
            "{}/{}".format(dataset_prefix, data_name[i]) for i in range(len(data_name))
        ]
    if zarr_path == 'articulated_250':
        data_name = [
            # save_data_name_0, save_data_name_1, save_data_name_2, save_data_name_3, save_data_name_4, 
            save_data_name_5, save_data_name_6, save_data_name_7, save_data_name_8, save_data_name_9,
            save_data_name_10, save_data_name_11, save_data_name_12, save_data_name_13, save_data_name_14, save_data_name_15, save_data_name_16, save_data_name_17, save_data_name_18, save_data_name_19,
            save_data_name_20, save_data_name_21, save_data_name_22, save_data_name_23, save_data_name_24, save_data_name_25, save_data_name_26, save_data_name_27, save_data_name_28, save_data_name_29,
            save_data_name_30, save_data_name_31, save_data_name_32, save_data_name_33, save_data_name_34, save_data_name_35, save_data_name_36, save_data_name_37, save_data_name_38, save_data_name_39,
            save_data_name_40, save_data_name_41, save_data_name_42, save_data_name_43, save_data_name_44, save_data_name_45, save_data_name_46, save_data_name_47, save_data_name_48, save_data_name_49,
            save_data_name_50, save_data_name_51, save_data_name_52, save_data_name_53, save_data_name_54, save_data_name_55, save_data_name_56, save_data_name_57, save_data_name_58, save_data_name_59,
            save_data_name_60, save_data_name_61, save_data_name_62, save_data_name_63, save_data_name_64, save_data_name_65, save_data_name_66, save_data_name_67, save_data_name_68, save_data_name_69,
            save_data_name_70, save_data_name_71, save_data_name_72, save_data_name_73, save_data_name_74, save_data_name_75, save_data_name_76, save_data_name_77, save_data_name_78, save_data_name_79,
            save_data_name_80, save_data_name_81, save_data_name_82, save_data_name_83, save_data_name_84, save_data_name_85, save_data_name_86, save_data_name_87, save_data_name_88, save_data_name_89,
            save_data_name_90, save_data_name_91, save_data_name_92, save_data_name_93, save_data_name_94, save_data_name_95, save_data_name_96, save_data_name_97, save_data_name_98, save_data_name_99] + articulated_new
        all_zarr_paths = [
            "{}/{}".format(dataset_prefix, data_name[i]) for i in range(len(data_name))
        ]
    
    if zarr_path == 'articulated_full':
        all_zarr_paths_part_1 = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(246)]
        all_subfolders = sorted(os.listdir(dataset_prefix))
        object_other_categories_no_cam_rand = [x for x in all_subfolders if "1121-other-cat-no-cam-rand" in x]
        all_zarr_paths_part_2 = [f"{dataset_prefix}/{x}" for x in object_other_categories_no_cam_rand]
        all_zarr_paths_part_3 = articulated_new
        all_zarr_paths_part_3 = ["{}/{}".format(dataset_prefix, name) for name in all_zarr_paths_part_3]
        all_zarr_paths = all_zarr_paths_part_1 + all_zarr_paths_part_2 + all_zarr_paths_part_3
    if zarr_path == 'full_and_close':
        all_zarr_paths_part_1 = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(246)]
        all_subfolders = sorted(os.listdir(dataset_prefix))
        object_other_categories_no_cam_rand = [x for x in all_subfolders if "1121-other-cat-no-cam-rand" in x]
        all_zarr_paths_part_2 = [f"{dataset_prefix}/{x}" for x in object_other_categories_no_cam_rand]
        all_zarr_paths_part_3 = articulated_new
        all_zarr_paths_part_3 = ["{}/{}".format(dataset_prefix, name) for name in all_zarr_paths_part_3]
        all_zarr_paths = all_zarr_paths_part_1 + all_zarr_paths_part_2 + all_zarr_paths_part_3
        close_prefix = '/mnt/RoboGen_sim2real/data/dp3_demo_combined_2_step_0/invert'
        close_prefix_2 = '/mnt/RoboGen_sim2real/data/dp3_demo_combined_2_step_0/invert_new'
        close_names = os.listdir(close_prefix)
        close_names = [name for name in close_names if name[0].isalpha()]
        close_names_2 = os.listdir(close_prefix_2)
        close_obj_paths = [
            "{}/{}".format(close_prefix, close_names[i]) for i in range(len(close_names))
        ] + [
            "{}/{}".format(close_prefix_2, close_names_2[i]) for i in range(len(close_names_2))
        ]
        all_zarr_paths += close_obj_paths

    if zarr_path == '10_object_high_level':
        dataset_prefix = '/scratch/yufeiw2/dp3_demo'
        all_zarr_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(10)]
    if zarr_path == '50_object_high_level':
        dataset_prefix = '/scratch/yufeiw2/dp3_demo'
        all_zarr_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(50)]
    if zarr_path == '100_object_high_level':
        dataset_prefix = '/scratch/yufeiw2/dp3_demo'
        all_zarr_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(100)]
    if zarr_path == "200_object_high_level":
        dataset_prefix = '/scratch/yufeiw2/dp3_demo'
        all_zarr_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(200)]
    if zarr_path == "300_object_high_level": 
        dataset_prefix = '/scratch/yufeiw2/dp3_demo'
        all_zarr_paths_part_1 = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(246)]
        all_subfolders = sorted(os.listdir(dataset_prefix))
        object_other_categories_no_cam_rand = [x for x in all_subfolders if "1121-other-cat-no-cam-rand" in x]
        all_zarr_paths_part_2 = [f"{dataset_prefix}/{x}" for x in object_other_categories_no_cam_rand]
        all_zarr_paths = all_zarr_paths_part_1 + all_zarr_paths_part_2
    
    dataset_prefix = '/data/minon/dp3_demo_combined_2_step_0'
    # dataset_prefix = '/scratch/yufeiw2/dp3_demo_combined_2_step_0'
    # dataset_prefix = '/local/'
    
    if zarr_path == '10_object_low_level':
        all_zarr_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(10)]
    if zarr_path == '50_object_low_level':
        all_zarr_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(50)]
    if zarr_path == '100_object_low_level':
        all_zarr_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(100)]
    if zarr_path == "200_object_low_level":
        all_zarr_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(200)]
    if zarr_path == "300_object_low_level": 
        all_zarr_paths_part_1 = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(246)]
        all_subfolders = sorted(os.listdir(dataset_prefix))
        object_other_categories_no_cam_rand = [x for x in all_subfolders if "1121-other-cat-no-cam-rand" in x]
        all_zarr_paths_part_2 = [f"{dataset_prefix}/{x}" for x in object_other_categories_no_cam_rand]
        all_zarr_paths = all_zarr_paths_part_1 + all_zarr_paths_part_2
        
    if zarr_path == 'camera_random_50_obj_high_level':
        dataset_prefix = '/scratch/yufeiw2/dp3_demo'
        all_zarr_paths = ["{}/{}".format(dataset_prefix, globals()["camera_random_50_save_data_name_{}".format(i)]) for i in range(87)]
    if zarr_path == 'camera_random_100_obj_high_level':
        dataset_prefix = '/scratch/yufeiw2/dp3_demo'
        all_zarr_paths = ["{}/{}".format(dataset_prefix, globals()["camera_random_100_save_data_name_{}".format(i)]) for i in range(175)]
    if zarr_path == 'camera_random_200_obj_high_level':
        dataset_prefix = '/scratch/yufeiw2/dp3_demo'
        all_zarr_paths = ["{}/{}".format(dataset_prefix, globals()["camera_random_200_save_data_name_{}".format(i)]) for i in range(350)]
    if zarr_path == 'camera_random_500_obj_high_level' or zarr_path == "500_object_high_level":
        dataset_prefix = '/scratch/yufeiw2/dp3_demo'
        all_zarr_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(462)]
        
    if zarr_path == "mixed_old_and_real_world_noisy_1119": # for low-level
        dataset_prefix_1 = '/scratch/yufeiw2/dp3_demo_combined_2_step_0'
        dataset_prefix_2 = '/scratch/yufeiw2/dp3_demo_real_world_noise_pcd_combined_2_step_0'
    
        old_list = [i for i in range(50)]
        all_old_obj_paths = ["{}/{}".format(dataset_prefix_1, globals()["save_data_name_{}".format(i)]) for i in old_list]
        
        all_new_obj_paths = os.listdir(dataset_prefix_2)
        all_new_obj_paths = sorted(all_new_obj_paths)
        all_new_obj_paths = [os.path.join(dataset_prefix_2, x) for x in all_new_obj_paths]
        
        all_obj_paths = all_old_obj_paths + all_new_obj_paths
        all_zarr_paths = all_obj_paths
    if zarr_path == "mixed_old_and_real_world_noisy_1119_high_level":
        dataset_prefix_1 = '/scratch/yufeiw2/dp3_demo'
        dataset_prefix_2 = '/scratch/yufeiw2/dp3_demo_real_world_noise_pcd'
        
        old_list = [i * 3 for i in range(150)]
        all_old_obj_paths = ["{}/{}".format(dataset_prefix_1, globals()["save_data_name_{}".format(i)]) for i in old_list]
        
        all_new_obj_paths = os.listdir(dataset_prefix_2)
        all_new_obj_paths = sorted(all_new_obj_paths)
        all_new_obj_paths = [os.path.join(dataset_prefix_2, x) for x in all_new_obj_paths]
        
        all_obj_paths = all_old_obj_paths + all_new_obj_paths
        all_zarr_paths = all_obj_paths
    if zarr_path == '50_plus_1026_gripper_closed_at_beginning_low_level':
        dataset_prefix_1 = '/scratch/yufeiw2/dp3_demo_combined_2_step_0'
        dataset_prefix_2 = '/scratch/yufeiw2/dp3_demo_combined_2_step_0'
    
        old_list = [i for i in range(50)] # first 50 objects
        new_list = [i for i in range(463, 569)] # all other cases where the gripper starts closed
        all_old_obj_paths = ["{}/{}".format(dataset_prefix_1, globals()["save_data_name_{}".format(i)]) for i in old_list]
        all_new_obj_paths = ["{}/{}".format(dataset_prefix_1, globals()["save_data_name_{}".format(i)]) for i in new_list]
        
        all_obj_paths = all_old_obj_paths + all_new_obj_paths
        all_zarr_paths = all_obj_paths
    if zarr_path == '500_plus_normal_other_cat':
        pass
    
    if zarr_path == "sriram_plate":
        dataset_prefix = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data/aloha"
        all_obj_paths = os.listdir(dataset_prefix)
        all_obj_paths = sorted(all_obj_paths)
        all_obj_paths = [os.path.join(dataset_prefix, x) for x in all_obj_paths]
        print(all_obj_paths)
        all_zarr_paths = all_obj_paths
        
    if zarr_path == 'sriram_plate_combine_2':
        dataset_prefix = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/articubot_multitask/RoboGen-sim2real/data/aloha_combined_2_step_0"
        all_obj_paths = os.listdir(dataset_prefix)
        all_obj_paths = sorted(all_obj_paths)
        all_obj_paths = [os.path.join(dataset_prefix, x) for x in all_obj_paths]
        print(all_obj_paths)
        all_zarr_paths = all_obj_paths
    
    return all_zarr_paths

class RobogenDataset(BaseDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.1,
            train_ratio=0.9,
            max_train_episodes=None,
            task_name=None,
            observation_mode='segmask',
            enumerate=False,
            is_pickle=False,
            dataset_keys=None,
            augmentation_pcd=False,
            augmentation_scale=False,
            scale_scene_by_pcd=False,
            use_absolute_waypoint=False,
            augmentation_rot=False,
            object_augmentation_high_level = False,
            mean_x_augmentation_high_level = None, 
            mean_y_augmentation_high_level = None, 
            std_x_augmentation_high_level = None, 
            std_y_augmentation_high_level = None,
            mean_angle_z_augmentation_high_level = None, 
            std_rot_z_augmentation_high_level = None,
            augmentation_goal_gripper_pcd=False,
            prob_x = None,
            prob_y = None,
            prob_rot_z = None,
            prediction_target='action',
            use_repr_10d=False, # 10D Representation for Low Level Policy
            pos_ori_imp=False, #10D Representation for High Level Policy
            dp3=False,
            goal_always_open=False,
            **kwargs
            ):
        super().__init__()

        self.task_name = task_name
        self.observation_mode = observation_mode
        self.augmentation_rot = augmentation_rot
        self.augmentation_pcd = augmentation_pcd
        self.augmentation_scale = augmentation_scale
        self.augmentation_goal_gripper_pcd = augmentation_goal_gripper_pcd
        self.scale_scene_by_pcd = scale_scene_by_pcd
        self.use_absolute_waypoint = use_absolute_waypoint
        self.is_pickle = is_pickle
        self.object_augmentation_high_level = object_augmentation_high_level
        self.prediction_target = prediction_target
        self.use_repr_10d=use_repr_10d
        self.pos_ori_imp=pos_ori_imp
        self.dp3 = dp3
        self.goal_always_open = goal_always_open

        cprint(f"Using 10D representation {self.use_repr_10d}", "red")
        
        if dataset_keys is None:
            keys = ['state', 'action', 'point_cloud']
            if 'act3d' in observation_mode:
                keys += ['feature_map', 'gripper_pcd', 'pcd_mask']
                if 'goal' in observation_mode:
                    keys += ['goal_gripper_pcd']
                if 'displacement_gripper_to_object' in observation_mode:
                    keys += ['displacement_gripper_to_object']
            elif 'act3d_pointnet' == observation_mode:
                keys += ['gripper_pcd']
        else:
            cprint(f"specifying dataset_keys: {dataset_keys}", "red")
            keys = dataset_keys
        
        self.keys_ = keys
        
        # try to get kept_in_disk from kwargs, if not, set it to False
        if 'kept_in_disk' in kwargs:
            self.kept_in_disk = kwargs['kept_in_disk']
        else:
            self.kept_in_disk = False 
            
        self.load_per_step = kwargs.get('load_per_step', False)

        self.only_reach_stage = kwargs.get('only_reach_stage', False)

        if self.kept_in_disk:
            cprint("loading dataset in disk, need a lot of I/O", "red")
            
        if not enumerate:
            from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
            self.replay_buffer = ReplayBuffer.copy_from_path(
                zarr_path, keys=keys)
            self.val_mask = np.zeros(self.replay_buffer.n_episodes, dtype=bool)
            self.val_mask[-int(self.replay_buffer.n_episodes*val_ratio):] = True
            
            train_mask = np.zeros(self.replay_buffer.n_episodes, dtype=bool)
            train_mask[:int(self.replay_buffer.n_episodes*train_ratio)] = True
        else:
            # import pdb; pdb.set_trace()

            # if type(zarr_path) != list:
            #     zarr_path = [zarr_path]
            
            if type(zarr_path) == list:
                all_zarr_paths = copy.deepcopy(zarr_path)
            else:
                all_zarr_paths = get_zarry_paths(zarr_path)
            
            
            all_paths = []
            train_masks = []
            val_masks = []
            for zarr_path in tqdm(all_zarr_paths):
                all_subfolder = os.listdir(zarr_path)
                # import pdb; pdb.set_trace()
                for string in ["action_dist", "demo_rgbs", "all_demo_path.txt", "meta_info.json", 'example_pointcloud', '.zgroup']:
                    if string in all_subfolder:
                        all_subfolder.remove(string)
                all_subfolder = sorted(all_subfolder)
                n_episodes = len(all_subfolder)
                num_load_episodes = kwargs.get('num_load_episodes', n_episodes)
                num_load_episodes = min(num_load_episodes, n_episodes)
                all_subfolder = all_subfolder[:num_load_episodes]
                # zarr_paths = [os.path.join(zarr_path, subfolder) for subfolder in all_subfolder]
                zarr_paths = []
                for subfolder in all_subfolder:
                    if len(os.listdir(os.path.join(zarr_path, subfolder))) > 10:
                        zarr_paths.append(os.path.join(zarr_path, subfolder))
                all_paths += zarr_paths
                folder_train_mask = np.zeros(num_load_episodes, dtype=bool)
                folder_train_mask[:int(num_load_episodes*train_ratio)] = True
                train_masks.append(folder_train_mask)
                folder_val_mask = np.zeros(num_load_episodes, dtype=bool)
                folder_val_mask[-int(num_load_episodes*val_ratio):] = True
                val_masks.append(folder_val_mask)
            
            if not self.kept_in_disk:
                from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
                self.replay_buffer = ReplayBuffer.copy_from_multiple_path(all_paths, keys=keys)
            else:
                cprint(f'keep in disk and load per step, load_per_step:{self.load_per_step}', 'green')
                from diffusion_policy_3d.common.replay_buffer_disk import ReplayBuffer
                self.replay_buffer = ReplayBuffer.copy_from_multiple_path(all_paths, keys=keys, load_per_step=self.load_per_step, 
                                                                        only_reach_stage=self.only_reach_stage, is_pickle=self.is_pickle,
                                                                        target_action=self.prediction_target, dp3=dp3)
                self.action_welford = self.replay_buffer.action_welford
                self.pcd_welford = self.replay_buffer.pcd_welford
                self.agent_pos_welford = self.replay_buffer.agent_pos_welford
            
            # self.val_mask = np.zeros(self.replay_buffer.n_episodes, dtype=bool)
            # self.val_mask[-int(self.replay_buffer.n_episodes*val_ratio):] = True
            # train_mask = np.zeros(self.replay_buffer.n_episodes, dtype=bool)
            # train_mask[:int(self.replay_buffer.n_episodes*train_ratio)] = True
            train_mask = np.concatenate(train_masks)
            self.val_mask = np.concatenate(val_masks)

        
        if not self.kept_in_disk:
            from diffusion_policy_3d.common.sampler import SequenceSampler
            self.sampler = SequenceSampler(
                replay_buffer=self.replay_buffer, 
                sequence_length=horizon,
                pad_before=pad_before, 
                pad_after=pad_after,
                episode_mask=train_mask)
        else:
            from diffusion_policy_3d.common.sampler_disk import SequenceSampler
            self.sampler = SequenceSampler(
                replay_buffer=self.replay_buffer, 
                sequence_length=horizon,
                pad_before=pad_before, 
                pad_after=pad_after,
                episode_mask=train_mask)
        
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after 

        if self.object_augmentation_high_level:
            print("High level Augmentation Setup")
            trans_x = TranslationXY(mean_x_augmentation_high_level, mean_y_augmentation_high_level, std_x_augmentation_high_level, std_y_augmentation_high_level, True, False)
            trans_y = TranslationXY(mean_x_augmentation_high_level, mean_y_augmentation_high_level, std_x_augmentation_high_level, std_y_augmentation_high_level, False, True)
            rot_z = rotationZ(mean_angle_z_augmentation_high_level, std_rot_z_augmentation_high_level)
            #probs = [0.4, 0.6, 0.3]
            transforms_and_probs = [[trans_x,prob_x], [trans_y, prob_y], [rot_z, prob_rot_z]]
            self.rand_apply = RandomApplyNumpy(transforms_and_probs)
        # [Chialiang]   
        cprint('dataset has been loaded', 'green')
            
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        if not self.kept_in_disk:
            from diffusion_policy_3d.common.sampler import SequenceSampler
            val_set.sampler = SequenceSampler(
                replay_buffer=self.replay_buffer, 
                sequence_length=self.horizon,
                pad_before=self.pad_before, 
                pad_after=self.pad_after,
                episode_mask=self.val_mask
                )
        else:
            from diffusion_policy_3d.common.sampler_disk import SequenceSampler
            val_set.sampler = SequenceSampler(
                replay_buffer=self.replay_buffer, 
                sequence_length=self.horizon,
                pad_before=self.pad_before, 
                pad_after=self.pad_after,
                episode_mask=self.val_mask
                )
        val_set.train_mask = self.val_mask
        return val_set
    

    def get_normalizer(self, mode='limits', **kwargs):
        # TODO: do we need to normalize the agent_pos and point cloud?
        # or just center point cloud to be at robot gripper?
        if not self.kept_in_disk:
            if 'act3d' not in self.observation_mode:
                data = {
                    'action': self.replay_buffer['action'],
                    'agent_pos': self.replay_buffer['state'][...,:],
                    'point_cloud': self.replay_buffer['point_cloud'],
                }
            else:
                # only normalizes actions, to make sure that the relative attention makes sense
                data = {
                    'action': self.replay_buffer['action'],
                }
            normalizer = LinearNormalizer()
            normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
            return normalizer
        else:
            normalizer = LinearNormalizer()
            if self.dp3: keys = ['action', 'point_cloud', 'gripper_pcd', 'agent_pos']
            else: keys = ['action']
            for key in keys:
                if key == 'action':
                    welford = self.action_welford
                if key == 'point_cloud' or key == 'gripper_pcd':
                    welford = self.pcd_welford
                if key == 'agent_pos':
                    welford = self.agent_pos_welford
                
                input_min = welford.get_min()
                input_max = welford.get_max()
                input_mean = welford.get_mean()
                input_std = welford.get_std()
                input_range = input_max - input_min
                range_eps = 1e-4
                output_min = -1
                output_max = 1
                ignore_dim = input_range < range_eps
                input_range[ignore_dim] = output_max - output_min
                scale = (output_max - output_min) / input_range
                offset = output_min - scale * input_min
                offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
                scale = torch.from_numpy(scale).float()
                offset = torch.from_numpy(offset).float()
                this_params = torch.nn.ParameterDict({
                    'scale': scale,
                    'offset': offset,
                    'input_stats': torch.nn.ParameterDict({
                        'min': input_min,
                        'max': input_max,
                        'mean': input_mean,
                        'std': input_std
                    })
                })
                for p in this_params.values():
                    p.requires_grad = False
                    
                if key == 'action':
                    normalizer.params_dict[self.prediction_target] = this_params
                else:
                    normalizer.params_dict[key] = this_params

            # [DebugNormalize] [Chialiang]
            if self.augmentation_rot:
                value = self.action_welford.get_max_norm_3d()
                value = torch.from_numpy(value).float()
                additional_params = torch.nn.ParameterDict({
                    'max_norm_3d': value
                })
                for p in additional_params.values():
                    p.requires_grad = False

                normalizer.params_dict['additional_params'] = additional_params

            return normalizer
    
    def __len__(self) -> int:
        return len(self.sampler)
    
    def _sample_to_data(self, sample):

        # get data
        agent_pos = copy.deepcopy(sample['state'][:,])
        point_cloud = copy.deepcopy(sample['point_cloud'][:,])
        action = copy.deepcopy(sample['action'])
        cat_idx = copy.deepcopy(sample['cat_idx'])
        cat_weights = copy.deepcopy(sample['cat_weights'])


        #10D GRIPPER BASELINE EXPERIMENT
        if self.pos_ori_imp:
            open_close = sample['state'][:-1,9]
            gripper_pcd_10d = []
            for i in range(self.horizon):
                gripper_pcd_10d.append(get_gripper_pos_orient_from_4_points_torch(sample['gripper_pcd'][i]))
            gripper_pcd = copy.deepcopy(np.array(gripper_pcd_10d))
            gripper_pcd = np.column_stack((gripper_pcd, open_close))
            gripper_pcd = np.expand_dims(gripper_pcd, axis=-1)
            goal_gripper_pcd_10d =[]
            open_close = sample['action'][:-1,9]
            for i in range(self.horizon):
                goal_gripper_pcd_10d.append(get_gripper_pos_orient_from_4_points_torch(sample['goal_gripper_pcd'][i]))
            goal_gripper_pcd = copy.deepcopy(np.array(goal_gripper_pcd_10d))
            goal_gripper_pcd = np.column_stack((goal_gripper_pcd, open_close))
            goal_gripper_pcd = np.expand_dims(goal_gripper_pcd, axis=-1)
            displacement_gripper_to_object = []
            for i in range(self.horizon):
                #print("000000000000000000", sample['displacement_gripper_to_object'].shape)
                object_point = sample['gripper_pcd'][i][0] + sample['displacement_gripper_to_object'][i][0]
                #print("111111111111", object_point.shape)
                displacement_gripper_to_object.append(object_point - gripper_pcd[i, :3, :].flatten())
                #print("222222222222", gripper_pcd.shape, gripper_pcd[i, :3, :].flatten().shape, (object_point - gripper_pcd[i, :3, :].flatten()).shape) 
            displacement_gripper_to_object = np.array(displacement_gripper_to_object).reshape(self.horizon, -1, 3)
            #print("HEREEEEEEE", gripper_pcd.shape, goal_gripper_pcd.shape, displacement_gripper_to_object.shape)



        if self.object_augmentation_high_level:
            gripper_pcd = copy.deepcopy(sample['gripper_pcd'][:,])
            goal_gripper_pcd = copy.deepcopy(sample['goal_gripper_pcd'][:,])
        agent_pos_old = copy.deepcopy(agent_pos)


        

        # augmentation
        ###########################################
        debug = False
        if debug:
            np.save('/project_data/held/chialiak/RoboGen-sim2real/one_traj/debug/agent_pos_before.npy', agent_pos)
            np.save('/project_data/held/chialiak/RoboGen-sim2real/one_traj/debug/point_cloud_before.npy', point_cloud)
            np.save('/project_data/held/chialiak/RoboGen-sim2real/one_traj/debug/action_before.npy', action)
            # np.save('/project_data/held/chialiak/RoboGen-sim2real/one_traj/debug/feature_map_before.npy', feature_map)
            np.save('/project_data/held/chialiak/RoboGen-sim2real/one_traj/debug/gripper_pcd_before.npy', gripper_pcd)
            np.save('/project_data/held/chialiak/RoboGen-sim2real/one_traj/debug/goal_gripper_pcd_before.npy', goal_gripper_pcd)
            np.save('/project_data/held/chialiak/RoboGen-sim2real/one_traj/debug/displacement_gripper_to_object_before.npy', displacement_gripper_to_object)
            start = time.time()
        ###########################################

        if self.augmentation_pcd:
            point_cloud = point_cloud + np.random.normal(0, 0.003, point_cloud.shape) # [AugTODO] add more 
            
        # if self.augmentation_goal_gripper_pcd:
        #     p = np.random.rand()
        #     print((point_cloud.shape[0], 2, 4))
        #     print(sample['goal_gripper_pcd'][:,].shape)
        #     if p < 0.33:
        #         random_noise = np.random.normal(0, 0.05, (point_cloud.shape[0], 2, 4))
        #         sample['goal_gripper_pcd'][:,] += random_noise
        #     if p >= 0.33 and p < 0.66:
        #         random_noise = np.random.normal(0, 0.02, (point_cloud.shape[0], 2, 4, 3))
        #         sample['goal_gripper_pcd'][:,] += random_noise
            
                

        if self.augmentation_rot:
            # random rotation
            random_trans = np.identity(4)
            random_zrot = (np.random.rand() * 2 - 1) * 10 * np.pi / 180 # -10 degree to 10 degree in raduis
            
            ###########################################
            if debug:
                random_zrot = 45 * np.pi / 180 
            ###########################################

            random_rotmat = p.getMatrixFromQuaternion(p.getQuaternionFromEuler([0, 0, random_zrot]))
            random_rotmat = np.asarray(random_rotmat).reshape(3, 3)
            random_trans[:3, :3] = random_rotmat

            # agent pos
            agent_pos_old = copy.deepcopy(agent_pos)
            agent_trans = np.identity(4).repeat(self.horizon, 1)
            pos_index = np.asarray([4*i+3 for i in range(self.horizon)]).astype(np.uint16)
            agent_trans[:3, pos_index] = agent_pos[:, :3].T
            rot_index = np.asarray([[4*i, 4*i+1, 4*i+2] for i in range(self.horizon)]).astype(np.uint16).reshape(-1)
            agent_trans[:3, rot_index] = rotation_transfer_6D_to_matrix_batch(agent_pos[:,3:9]) # should be 6D rotation representation
            agent_trans = random_trans @ agent_trans
            agent_pos[:, :3] = agent_trans[:3, pos_index].T
            agent_pos[:, 3:9] = rotation_transfer_matrix_to_6D_batch(agent_trans[:3, rot_index].T)

            # point cloud
            point_cloud_homo = np.ones((point_cloud.shape[0] * point_cloud.shape[1], 4))
            point_cloud_homo[:,:3] = point_cloud.reshape((-1, 3))
            point_cloud = (point_cloud_homo @ random_trans.T)[:, :3]
            point_cloud = point_cloud.reshape(self.horizon, -1, 3)

            # action
            action[:,:3] = action[:,:3] @ random_rotmat.T

            if 'act3d' in self.observation_mode:

                gripper_pcd_copy = copy.deepcopy(gripper_pcd)
                gripper_pcd_homo = np.ones((gripper_pcd.shape[0] * gripper_pcd.shape[1], 4))
                gripper_pcd_homo[:,:3] = gripper_pcd.reshape((-1, 3))
                gripper_pcd = (gripper_pcd_homo @ random_trans.T)[:, :3]
                gripper_pcd = gripper_pcd.reshape(self.horizon, -1, 3)

                # if 'mlp' not in self.observation_mode:
                #     feature_num = feature_map.shape[0] * feature_map.shape[1]
                #     feature_dim = feature_map.shape[2]
                #     feature_map = feature_map.reshape((-1, feature_dim))
                #     feature_map_homo = np.ones((feature_num, 4))
                #     feature_map_homo[:,:3] = feature_map[:,2:]
                #     feature_map[:,2:] = (feature_map_homo @ random_trans.T)[:, :3]
                #     feature_map = feature_map.reshape(self.horizon, -1, feature_dim)

                if 'goal' in self.observation_mode:
                    goal_gripper_pcd_homo = np.ones((goal_gripper_pcd.shape[0] * goal_gripper_pcd.shape[1], 4))
                    goal_gripper_pcd_homo[:,:3] = goal_gripper_pcd.reshape((-1, 3))
                    goal_gripper_pcd = (goal_gripper_pcd_homo @ random_trans.T)[:, :3]
                    goal_gripper_pcd = goal_gripper_pcd.reshape(self.horizon, -1, 3)
                
                if 'displacement_gripper_to_object' in self.observation_mode:
                    goal_gripper_to_pcd = gripper_pcd_copy + displacement_gripper_to_object
                    goal_gripper_to_pcd_homo = np.ones((goal_gripper_to_pcd.shape[0] * goal_gripper_to_pcd.shape[1], 4))
                    goal_gripper_to_pcd_homo[:,:3] = goal_gripper_to_pcd.reshape((-1, 3))
                    goal_gripper_to_pcd = (goal_gripper_to_pcd_homo @ random_trans.T)[:, :3]
                    goal_gripper_to_pcd = goal_gripper_to_pcd.reshape(self.horizon, -1, 3)
                    displacement_gripper_to_object = goal_gripper_to_pcd - gripper_pcd
            
            elif 'act3d_pointnet' == self.observation_mode:

                gripper_pcd_homo = np.ones((gripper_pcd.shape[0] * gripper_pcd.shape[1], 4))
                gripper_pcd_homo[:,:3] = gripper_pcd.reshape((-1, 3))
                gripper_pcd = (gripper_pcd_homo @ random_trans.T)[:, :3]
                gripper_pcd = gripper_pcd_homo.reshape(self.horizon, -1, 3)

        if self.object_augmentation_high_level:
            #print("APPLYING DATA AUGMENTATION")
            data = {
                'point_cloud': point_cloud.astype(np.float32), # T, 1280, 
                'agent_pos': agent_pos.astype(np.float32), # T, D_pos
                'gripper_pcd' : gripper_pcd.astype(np.float32),
                'goal_gripper_pcd': goal_gripper_pcd.astype(np.float32)
        }
            data = self.rand_apply(data)
            point_cloud = data["point_cloud"]
            gripper_pcd = data["gripper_pcd"]
            goal_gripper_pcd = data["goal_gripper_pcd"]



        # change to absolute waypoints
        if self.use_absolute_waypoint:

            absolute_action = copy.deepcopy(agent_pos)
            
            # position
            absolute_action[:, :3] += action[:, :3]
            
            # rotation
            current_rotations = rotation_transfer_6D_to_matrix_batch(agent_pos[:, 3:9]).T.reshape((-1, 3, 3)) # (H, 6) -> (3, 3*H) -> (3*H, 3) -> (H, 3, 3)
            current_rotations = np.transpose(current_rotations, (0, 2, 1)) # (H, 3, 3) make row vector column vector
            delta_rotations = rotation_transfer_6D_to_matrix_batch(action[:, 3:9]).T.reshape((-1, 3, 3)) # (H, 6) -> (3, 3*H) -> (3*H, 3) -> (H, 3, 3)
            delta_rotations = np.transpose(delta_rotations, (0, 2, 1)) # (H, 3, 3) make row vector column vector
            next_rotations = np.matmul(current_rotations, delta_rotations) # (H, 3, 3)
            row_1 = next_rotations[:, :3, 0] # (H, 3)
            row_2 = next_rotations[:, :3, 1] # (H, 3)
            rot_6d = np.hstack((row_1, row_2)) # (H, 6)
            absolute_action[:, 3:9] = rot_6d

            # eef
            absolute_action[:, 9] += action[:, 9]

            action = absolute_action

        if self.augmentation_scale:

            max_difference = 0.2
            random_scale = 1 + max_difference * (2 * np.random.rand() - 1) # [1 - max_difference, 1 + max_difference]

            point_cloud[...,:3] *= random_scale
            agent_pos[...,:3] *= random_scale
            action[...,:3] *= random_scale
            
            if 'act3d' in self.observation_mode:
                gripper_pcd[...,:3] *= random_scale
                if 'goal' in self.observation_mode:
                    goal_gripper_pcd[...,:3] *= random_scale
                if 'displacement_gripper_to_object' in self.observation_mode:
                    displacement_gripper_to_object[...,:3] *= random_scale
            
            elif 'act3d_pointnet' == self.observation_mode:
                gripper_pcd[...,:3] *= random_scale

        if self.scale_scene_by_pcd:

            max_scale = np.max(np.linalg.norm(point_cloud, axis=-1))

            point_cloud[...,:3] /= max_scale
            agent_pos[...,:3] /= max_scale
            action[...,:3] /= max_scale

            if 'act3d' in self.observation_mode:
                gripper_pcd[...,:3] /= max_scale
                if 'goal' in self.observation_mode:
                    goal_gripper_pcd[...,:3] /= max_scale
                if 'displacement_gripper_to_object' in self.observation_mode:
                    displacement_gripper_to_object[...,:3] /= max_scale
            
            elif 'act3d_pointnet' == self.observation_mode:
                gripper_pcd[...,:3]  /= max_scale
        if self.object_augmentation_high_level:
            data = {
            'obs': {
                'point_cloud': point_cloud.astype(np.float32), # T, 1280, 
                'agent_pos': agent_pos.astype(np.float32), # T, D_pos
                'gripper_pcd': gripper_pcd.astype(np.float32),
                'goal_gripper_pcd': goal_gripper_pcd.astype(np.float32)
                },
                'action': action.astype(np.float32),
                'cat_idx': cat_idx.astype(np.int64),
                'cat_weights': cat_weights.astype(np.float32)
            }
            for key in self.keys_:
                if key not in ['state', 'action', 'point_cloud', 'gripper_pcd', 'goal_gripper_pcd']:
                    data['obs'][key] = copy.deepcopy(sample[key][:,].astype(np.float32))
                    
        elif self.pos_ori_imp:
            data = {
                'obs': {
                    'point_cloud': point_cloud.astype(np.float32), # T, 1280, 
                    'agent_pos': agent_pos.astype(np.float32), # T, D_pos
                    'gripper_pcd': gripper_pcd.astype(np.float32),
                    'goal_gripper_pcd': goal_gripper_pcd.astype(np.float32),
                    'displacement_gripper_to_object': displacement_gripper_to_object.astype(np.float32)
                },
                'action': action.astype(np.float32),
                'cat_idx': cat_idx.astype(np.int64),
                'cat_weights': cat_weights.astype(np.float32)
            }
            for key in self.keys_:
                if key not in ['state', 'action', 'point_cloud', 'gripper_pcd', 'goal_gripper_pcd', 'displacement_gripper_to_object']:
                    data['obs'][key] = copy.deepcopy(sample[key][:,].astype(np.float32))[:self.horizon,:,:]
        else:
            # assign to dict
            data = {
                'obs': {
                    'point_cloud': point_cloud.astype(np.float32), # T, 1280, 
                    'agent_pos': agent_pos.astype(np.float32), # T, D_pos
                },
                'action': action.astype(np.float32),
                'cat_idx': cat_idx.astype(np.int64),
                'cat_weights': cat_weights.astype(np.float32)
            }

            # if 'act3d' in self.observation_mode:
            #     data['obs']['gripper_pcd'] = gripper_pcd.astype(np.float32)
            #     if 'mlp' not in self.observation_mode:
            #         data['obs']['feature_map'] = feature_map.astype(np.float32)
            #         data['obs']['pcd_mask'] = pcd_mask.astype(np.uint8)
            #     if 'goal' in self.observation_mode:
            #         data['obs']['goal_gripper_pcd'] = goal_gripper_pcd.astype(np.float32)
            #     if 'displacement_gripper_to_object' in self.observation_mode:
            #         data['obs']['displacement_gripper_to_object'] = displacement_gripper_to_object.astype(np.float32)
            for key in self.keys_:
                if key not in ['state', 'action', 'point_cloud']:
                    data['obs'][key] = copy.deepcopy(sample[key][:,].astype(np.float32))
            
            if self.goal_always_open:
                # import pdb; pdb.set_trace()
                from test_PointNet2.dataset_from_disk import change_goal_gripper_pcd_to_open
                for idx in range(len(data['obs']['goal_gripper_pcd'])):
                    goal_gripper = data['obs']['goal_gripper_pcd'][idx]
                    new_goal_gripper = change_goal_gripper_pcd_to_open(goal_gripper)
                    data['obs']['goal_gripper_pcd'][idx] = new_goal_gripper
                
        if self.prediction_target == 'delta_to_goal_gripper':
            data['obs']['delta_to_goal_gripper'] = data['obs']['goal_gripper_pcd'] - data['obs']['gripper_pcd']
        
        if self.use_repr_10d:
            data['obs']['goal_gripper_10d_repr'] = gripper_pcd_to_10d_vector(data['obs']['goal_gripper_pcd'])
        return data

    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
