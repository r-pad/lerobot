import os
import hydra
import torch
from omegaconf import OmegaConf
from train_ddp import TrainDP3Workspace
from diffusion_policy_3d.common.pytorch_util import dict_apply
from manipulation.utils import build_up_env, save_numpy_as_gif
import numpy as np
from copy import deepcopy
from manipulation.robogen_wrapper import RobogenPointCloudWrapper
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
import json
import yaml
import argparse
from typing import Optional
from collections import deque

def construct_env(cfg, config_file, env_name, init_state_file, obj_translation=None, real_world_camera=False, noise_real_world_pcd=False,
                  randomize_camera=False):
    config = yaml.safe_load(open(config_file, "r"))
    for config_dict in config:
        if 'name' in config_dict:
            object_name = config_dict['name'].lower()
    env, _ = build_up_env(
                    task_config=config_file,
                    env_name=env_name,
                    restore_state_file=init_state_file,
                    # render=False, 
                    render=False, 
                    randomize=False,
                    obj_id=0,
                    horizon=600,
                    random_object_translation=obj_translation,
            )
    env.reset()
    pointcloud_env = RobogenPointCloudWrapper(env, object_name, in_gripper_frame=cfg.task.env_runner.in_gripper_frame, 
                                                gripper_num_points=cfg.task.env_runner.gripper_num_points, add_contact=cfg.task.env_runner.add_contact,
                                                num_points=cfg.task.env_runner.num_point_in_pc,
                                                use_joint_angle=cfg.task.env_runner.use_joint_angle, 
                                                use_segmask=cfg.task.env_runner.use_segmask,
                                                only_handle_points=cfg.task.env_runner.only_handle_points,
                                                observation_mode=cfg.task.env_runner.observation_mode,
                                                real_world_camera=real_world_camera,
                                                noise_real_world_pcd=noise_real_world_pcd,
                                                )
        
    if randomize_camera:
        pointcloud_env.reset_random_cameras()
        
    env = MultiStepWrapper(pointcloud_env, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps, 
                        max_episode_steps=600, reward_agg_method='sum')
    
    return env

def run_eval_non_parallel(cfg, policy, goal_prediction_model, num_worker, save_path, cat_idx, exp_beg_idx=0,
                          exp_end_idx=1000, embedding_dim=None, pool=None, horizon=150,  exp_beg_ratio=None, exp_end_ratio=None,
                          dataset_index=None, calculate_distance_from_gt=False, output_obj_pcd_only=False, obj_translation: Optional[list]= None,
                          update_goal_freq=1, real_world_camera=False, noise_real_world_pcd=False,
                          randomize_camera=False, pos_ori_imp=False):
    
    for dataset_idx, (experiment_folder, experiment_name, demo_experiment_path) in enumerate(zip(cfg.task.env_runner.experiment_folder, cfg.task.env_runner.experiment_name, cfg.task.env_runner.demo_experiment_path)):
        
        if dataset_index is not None:
            dataset_idx = dataset_index

        if calculate_distance_from_gt:
            all_obj_distances = []

        # if dataset_idx == 0:
        #     continue

        init_state_files = []
        config_files = []
        experiment_folder = "{}/{}".format(os.environ['PROJECT_DIR'], experiment_folder)
        experiment_name = experiment_name
        experiment_path = os.path.join(experiment_folder, "experiment", experiment_name)
        all_experiments = os.listdir(experiment_path)
        all_experiments = sorted(all_experiments)
        
        if demo_experiment_path is not None:
            # demo_experiment_path = demo_experiment_path[demo_experiment_path.find("RoboGen_sim2real/") + len("RoboGen_sim2real/"):]
            all_subfolder = os.listdir(demo_experiment_path)
            for string in ["action_dist", "demo_rgbs", "all_demo_path.txt", "meta_info.json", 'example_pointcloud']:
                if string in all_subfolder:
                    all_subfolder.remove(string)
            all_subfolder = sorted(all_subfolder)
            all_experiments = all_subfolder
    
        expert_opened_angles = []
        for experiment in all_experiments:
            if "meta" in experiment:
                continue

            exp_folder = os.path.join(experiment_path, experiment)
            if os.path.exists(os.path.join(exp_folder, "label.json")):
                with open(os.path.join(exp_folder, "label.json"), 'r') as f:
                    label = json.load(f)
                if not label['good_traj']: continue
                
            states_path = os.path.join(exp_folder, "states")
            if not os.path.exists(states_path):
                continue
            if len(os.listdir(states_path)) <= 1 or not os.path.exists(os.path.join(exp_folder, "all.gif")):
                continue
            expert_states = [f for f in os.listdir(states_path) if f.startswith("state")]
            if len(expert_states) == 0:
                continue
            stage_lengths = os.path.join(exp_folder, "stage_lengths.json")
            with open(stage_lengths, "r") as f:
                stage_lengths = json.load(f)
            open_time_idx = stage_lengths['reach_handle'] + stage_lengths["reach_to_contact"] + stage_lengths["close_gripper"]
            if len(expert_states) - open_time_idx < 5: # if the opening time is too short, skip this trajectory
                continue  
                
            expert_opened_angle_file = os.path.join(experiment_path, experiment, "opened_angle.txt")
            if os.path.exists(expert_opened_angle_file):
                with open(expert_opened_angle_file, "r") as f:
                    angles = f.readlines()
                    expert_opened_angle = float(angles[0].lstrip().rstrip())
                    # max_angle = float(angles[-1].lstrip().rstrip())
                    # ratio = expert_opened_angle / max_angle+0.001)
                # if ratio < 0.65:
                #     continue
            expert_opened_angles.append(expert_opened_angle)
                
            init_state_file = os.path.join(states_path, "state_0.pkl")
            init_state_files.append(init_state_file)
            config_file = os.path.join(experiment_path, experiment, "task_config.yaml")
            config_files.append(config_file)

        opened_joint_angles = {}

        if exp_end_ratio is not None:
            exp_end_idx = int(exp_end_ratio * len(config_files))
        if exp_beg_ratio is not None:
            exp_beg_idx = int(exp_beg_ratio * len(config_files))

        angle_threshold = np.quantile(expert_opened_angles, 0.5)
        selected_idx = [i for i, angle in enumerate(expert_opened_angles) if angle > angle_threshold]
        config_files = [config_files[i] for i in selected_idx]
        init_state_files = [init_state_files[i] for i in selected_idx]
        expert_opened_angles = [expert_opened_angles[i] for i in selected_idx]

        config_files = config_files[exp_beg_idx:exp_end_idx]
        init_state_files = init_state_files[exp_beg_idx:exp_end_idx]
        expert_opened_angles = expert_opened_angles[exp_beg_idx:exp_end_idx]
        # import pdb; pdb.set_trace()
        all_distances = []
        all_grasp_distances = []

        if args.category_embedding_type == "siglip":
            siglip_text_features = torch.load("/project_data/held/chenyuah/RoboGen-sim2real/siglip_text_features.pt")
        cat_idx_cuda = torch.tensor(cat_idx).to('cuda')

        for exp_idx, (config_file, init_state_file) in enumerate(zip(config_files, init_state_files)):

            env = construct_env(cfg, config_file, "articulated", init_state_file, obj_translation, real_world_camera, noise_real_world_pcd, 
                                randomize_camera)
            
            obs = env.reset(open_gripper_at_reset=True)
            rgb = env.env.render()
            info = env.env._env._get_info()

            initial_info = info
            all_rgbs = [rgb]
            goal_stage = 'first'
            first_step_outputs = None
            gripper_close_accumulation_buffer = deque(maxlen=5)
            last_goal = None
            for t in range(1, horizon):
                parallel_input_dict = obs
                parallel_input_dict = dict_apply(parallel_input_dict, lambda x: torch.from_numpy(x).to('cuda'))
                
                # print("step: ", t)
                
                for key in obs:
                    parallel_input_dict[key] = parallel_input_dict[key].unsqueeze(0)
                
                if t == 1 or (not args.predict_two_goals):
                    if t == 1 or t % update_goal_freq == 0:
                        with torch.no_grad():
                            pointcloud = parallel_input_dict['point_cloud'][:, -1, :, :]
                            gripper_pcd = parallel_input_dict['gripper_pcd'][:, -1, :]
                            if args.category_embedding_type == "one_hot":
                                cat_embedding = torch.nn.functional.one_hot(cat_idx_cuda, num_classes=embedding_dim).float().to(pointcloud.device)
                            elif args.category_embedding_type == "siglip":
                                cat_embedding = siglip_text_features[cat_idx].float().to(pointcloud.device)
                            else:
                                cat_embedding = None
                            if not args.predict_two_goals:
                                inputs = torch.cat([pointcloud, gripper_pcd], dim=1)
                            else:
                                inputs = pointcloud
                                
                            if args.add_one_hot_encoding:
                                # for pointcloud, we add (1, 0)
                                # for gripper_pcd, we add (0, 1)
                                pointcloud_one_hot = torch.zeros(pointcloud.shape[0], pointcloud.shape[1], 2).float().to(pointcloud.device)
                                pointcloud_one_hot[:, :, 0] = 1
                                pointcloud_ = torch.cat([pointcloud, pointcloud_one_hot], dim=2)
                                gripper_pcd_one_hot = torch.zeros(gripper_pcd.shape[0], gripper_pcd.shape[1], 2).float().to(pointcloud.device)
                                gripper_pcd_one_hot[:, :, 1] = 1
                                gripper_pcd_ = torch.cat([gripper_pcd, gripper_pcd_one_hot], dim=2)
                                inputs = torch.cat([pointcloud_, gripper_pcd_], dim=1) # B, N+4, 5
                                
                            inputs = inputs.to('cuda')
                            inputs_ = inputs.permute(0, 2, 1)
                            if args.model_transformer:
                                outputs = goal_prediction_model(inputs)
                            else:
                                outputs = goal_prediction_model(inputs_, cat_embedding)
                            weights = outputs[:, :, -1] # B, N
                            outputs = outputs[:, :, :-1] # B, N, 12
                            if output_obj_pcd_only:
                                # cprint("using only obj pcd output!", "red")
                                weights = weights[:, :-4]
                                outputs = outputs[:, :-4, :]
                                inputs = inputs[:, :-4, :]

                            B, N, _ = outputs.shape
                            if not args.predict_two_goals:
                                outputs = outputs.view(B, N, 4, 3)
                            else:
                                outputs = outputs.view(B, N, 8, 3)
                                if first_step_outputs is None:
                                    first_step_outputs = deepcopy(outputs)
                                    
                                if goal_stage == 'first':
                                    outputs = outputs[:, :, :4, :]
                                elif goal_stage == 'second':
                                    outputs = outputs[:, :, 4:, :]
                                    
                            outputs = outputs + inputs[:, :, :3].unsqueeze(2)
                            weights = torch.nn.functional.softmax(weights, dim=1)
                            outputs = outputs * weights.unsqueeze(-1).unsqueeze(-1)
                            outputs = outputs.sum(dim=1)
                            outputs = outputs.unsqueeze(1)
                            # visualize weights
                            # import matplotlib.pyplot as plt
                            # fig = plt.figure()
                            # ax = fig.add_subplot(111, projection='3d')
                            # pointcloud = pointcloud.squeeze(0).cpu().numpy()
                            # print("pointcloud: ", pointcloud.shape)
                            # print("weights: ", weights[0].shape)
                            # if pointcloud.shape[0] != weights[0].shape[0]:
                            #     import pdb; pdb.set_trace()
                            # ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], c=weights[0].cpu().numpy(), cmap='seismic')
                            # ax.view_init(elev=24, azim=-117) 
                            # ax.axis("equal")
                            # plt.show()
                        last_goal = outputs
                    else:
                        outputs = last_goal
                        
                else:
                    outputs = first_step_outputs
                    if goal_stage == 'first':
                        outputs = outputs[:, :, :4, :]
                    elif goal_stage == 'second':
                        outputs = outputs[:, :, 4:, :]
                            
                    outputs = outputs + inputs.unsqueeze(2)
                    weights = torch.nn.functional.softmax(weights, dim=1)
                    outputs = outputs * weights.unsqueeze(-1).unsqueeze(-1)
                    outputs = outputs.sum(dim=1)
                    outputs = outputs.unsqueeze(1)
                    
                np_predicted_goal = outputs.detach().to('cpu').numpy()
                
                predicted_goal = outputs.repeat(1, 2, 1, 1)

                #parallel_input_dict['goal_gripper_pcd'] = predicted_goal
                if pos_ori_imp:
                    from diffuser_actor_3d.robogen_utils import gripper_pcd_to_10d_vector
                    #import pdb; pdb.set_trace();
                    predicted_goal = predicted_goal.reshape(-1,4,3)
                    predicted_goal = torch.from_numpy(gripper_pcd_to_10d_vector(predicted_goal.cpu().numpy())).to(torch.float32).cuda()
                    predicted_goal = predicted_goal.reshape(1,-1,10)
                    
                    parallel_input_dict['goal_gripper_10d_repr'] = predicted_goal
                else:
                    parallel_input_dict['goal_gripper_pcd'] = predicted_goal
                #import pdb; pdb.set_trace()
                with torch.no_grad():
                    batched_action = policy.predict_action(parallel_input_dict)
                    gripper_close_actions = batched_action['action'][:, :, -1].detach().cpu().numpy()
                    gripper_close_accumulation_buffer.append(np.sum(gripper_close_actions))
                    if np.sum(gripper_close_accumulation_buffer) < -0.006:
                        # cprint("changing goal!", 'red')
                        goal_stage = 'second'
                    
                    
                np_batched_action = dict_apply(batched_action, lambda x: x.detach().to('cpu').numpy())
                np_batched_action = np_batched_action['action']
                
                obs, reward, done, info = env.step(np_batched_action.squeeze(0))
                if calculate_distance_from_gt:
                    predicted_goal = np_predicted_goal.squeeze(0)[0].reshape(4, 3)
                    gt_goal = env.env.goal_gripper_pcd
                    import pdb; pdb.set_trace()
                    distance = np.linalg.norm(predicted_goal - gt_goal, axis=1).mean()
                    all_distances.append(distance)
                    grasp_distance = np.linalg.norm(predicted_goal[-1] - gt_goal[-1])
                    all_grasp_distances.append(grasp_distance)
                    break
                env.env.goal_gripper_pcd = np_predicted_goal.squeeze(0)[0].reshape(4, 3)
                rgb = env.env.render()
                all_rgbs.append(rgb)
            
            env.env._env.close()

            if calculate_distance_from_gt:
                break
            
            opened_joint_angles[config_file] = \
            {
                "final_door_joint_angle": float(info['opened_joint_angle'][-1]), 
                "expert_door_joint_angle": expert_opened_angles[exp_idx], 
                "initial_joint_angle": float(info['initial_joint_angle'][-1]),
                "ik_failure": float(info['ik_failure'][-1]),
                'oversized_joint_distance': float(info['oversized_joint_distance'][-1]),
                'grasped_handle': float(info['grasped_handle'][-1]),
                "exp_idx": exp_idx, 
            }
            # gif_save_exp_name = experiment_folder.split("/")[-1] 
            gif_save_exp_name = experiment_folder.split("/")[-2] + "/" + experiment_folder.split("/")[-1]
            gif_save_folder = "{}/{}".format(save_path, gif_save_exp_name)                 
            if not os.path.exists(gif_save_folder):
                os.makedirs(gif_save_folder, exist_ok=True)

            with open("{}/opened_joint_angles.json".format(gif_save_folder), "w") as f:
                json.dump(opened_joint_angles, f, indent=4)
   
            gif_save_path = "{}/{}_{}.gif".format(gif_save_folder, exp_idx, 
                    float(info["improved_joint_angle"][-1]))
            save_numpy_as_gif(np.array(all_rgbs), gif_save_path)

        if calculate_distance_from_gt:
            print("average distance: {}".format(np.mean(all_distances)))
            print("average grasp distance: {}".format(np.mean(all_grasp_distances)))
            all_obj_distances.append(all_distances)

    if calculate_distance_from_gt:
        print("average distance over all objects: {}".format(np.mean(all_obj_distances)))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--low_level_exp_dir', type=str, default=None)
    parser.add_argument('--low_level_ckpt_name', type=str, default=None)
    parser.add_argument("--high_level_ckpt_name", type=str, default=None)
    parser.add_argument("--pointnet_class", type=str, default="PointNet2")
    parser.add_argument("--eval_exp_name", type=str, default=None)
    parser.add_argument("--use_predicted_goal", type=bool, default=True)
    parser.add_argument("--test_cross_category", type=bool, default=False)
    parser.add_argument("--model_invariant", type=bool, default=True)
    parser.add_argument("--model_transformer", action='store_true')
    parser.add_argument('--predict_two_goals', action='store_true')
    parser.add_argument('--output_obj_pcd_only', action='store_true')
    parser.add_argument("--update_goal_freq", type=int, default=1)
    parser.add_argument("--noise_real_world_pcd", type=int, default=0)
    parser.add_argument("--randomize_camera", type=int, default=0)
    parser.add_argument("--real_world_camera", type=int, default=0)
    parser.add_argument('-n', '--noise', type=float, default=None, nargs=2, help='bounds for noise. e.g. `--noise -0.1 0.1')
    parser.add_argument('--keep_gripper_in_fps', type=int, default=0)
    parser.add_argument('--add_one_hot_encoding', type=int, default=0)
    parser.add_argument('--pos_ori_imp', action='store_true', help='Set the flag for 10D representation Training')
    parser.add_argument('--exp_dir', type=str, help='Experiment directory')
    parser.add_argument('--num_categories', type=int, default=7)
    parser.add_argument('--category_embedding_type', type=str, default="none")
    args = parser.parse_args()
    
    num_worker = 30
    pool=None

    categories = ['bucket', 'faucet', 'foldingchair', 'laptop', 'stapler', 'toilet']
    cat_idx = 0
    for i, cat in enumerate(categories):
        if cat in args.exp_dir:
            cat_idx = i + 1
            break

    if args.low_level_exp_dir is None:
        # best 50 objects
        exp_dir = "/project_data/held/chialiak/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/07201526-act3d_goal_mlp-horizon-8-num_load_episodes-1000/2024.07.20/15.26.54_train_dp3_robogen_open_door"
        checkpoint_name = 'latest.ckpt'
    else:
        exp_dir = args.low_level_exp_dir
        checkpoint_name = args.low_level_ckpt_name

    with hydra.initialize(config_path='diffusion_policy_3d/config'):  # same config_path as used by @hydra.main
        recomposed_config = hydra.compose(
            config_name="dp3.yaml",  # same config_name as used by @hydra.main
            overrides=OmegaConf.load("{}/.hydra/overrides.yaml".format(exp_dir)),
        )
    cfg = recomposed_config
    
    workspace = TrainDP3Workspace(cfg)
    checkpoint_dir = "{}/checkpoints/{}".format(exp_dir, checkpoint_name)
    workspace.load_checkpoint(path=checkpoint_dir, )

    #Low level policy loading 
    policy = deepcopy(workspace.model)
    if workspace.cfg.training.use_ema:
        policy = deepcopy(workspace.ema_model)
    policy.eval()
    policy.reset()
    policy = policy.to('cuda')
    ''''data/diverse_objects/open_the_door_44962/task_open_the_door_of_the_storagefurniture_by_its_handle',
        'data/diverse_objects/open_the_door_45132/task_open_the_door_of_the_storagefurniture_by_its_handle',
        'data/diverse_objects/open_the_door_45219/task_open_the_door_of_the_storagefurniture_by_its_handle',
        'data/diverse_objects/open_the_door_45243/task_open_the_door_of_the_storagefurniture_by_its_handle',
        'data/diverse_objects/open_the_door_45332/task_open_the_door_of_the_storagefurniture_by_its_handle',
        'data/diverse_objects/open_the_door_45378/task_open_the_door_of_the_storagefurniture_by_its_handle',
        'data/diverse_objects/open_the_door_45384/task_open_the_door_of_the_storagefurniture_by_its_handle',
        'data/diverse_objects/open_the_door_45463/task_open_the_door_of_the_storagefurniture_by_its_handle', 
        'data/diverse_objects/open_the_door_40147/task_open_the_door_of_the_storagefurniture_by_its_handle', '''
    # cfg.task.env_runner.experiment_name = ['0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first' for _ in range(10)]
    # cfg.task.env_runner.experiment_folder = [
    #     'data/diverse_objects/open_the_door_44962/task_open_the_door_of_the_storagefurniture_by_its_handle',
        
    #     ]
    if 'diverse_objects' in args.exp_dir:
        cfg.task.env_runner.experiment_name = ['0705' for _ in range(1)]
    else: 
        cfg.task.env_runner.experiment_name = ['165-obj' for _ in range(1)]

    cfg.task.env_runner.experiment_folder = [
        args.exp_dir,
        # bucket_tasks
        # 'data/bucket/100444',
        # 'data/bucket/100452',
        # 'data/bucket/100454',
        # 'data/bucket/100460',
        # 'data/bucket/100461',
        # 'data/bucket/100462',
        # 'data/bucket/100469',
        # 'data/bucket/100472',
        # 'data/bucket/102352',
        # 'data/bucket/102365',

        # faucet_tasks
        # 'data/faucet/148',
        # 'data/faucet/149',
        # 'data/faucet/152',
        # 'data/faucet/153',
        # 'data/faucet/154',
        # 'data/faucet/168',
        # 'data/faucet/811',
        # 'data/faucet/857',
        # 'data/faucet/960',
        # 'data/faucet/991',

        # foldingchair_tasks
        # 'data/foldingchair/100520',
        # 'data/foldingchair/100521',
        # 'data/foldingchair/100526',
        # 'data/foldingchair/100562',
        # 'data/foldingchair/100586',
        # 'data/foldingchair/100590',
        # 'data/foldingchair/100599',
        # 'data/foldingchair/102263',
        # 'data/foldingchair/102269',
        # 'data/foldingchair/102314',

        # # laptop_tasks
        # 'data/laptop/9748',
        # 'data/laptop/9912',
        # 'data/laptop/9960',
        # 'data/laptop/9968',
        # 'data/laptop/9992',
        # 'data/laptop/9996',
        # 'data/laptop/10040',
        # 'data/laptop/10098',
        # 'data/laptop/10101',
        # 'data/laptop/10238',

        # # stapler_tasks
        # 'data/stapler/103095',
        # 'data/stapler/103099',
        # 'data/stapler/103100',
        # 'data/stapler/103104',
        # 'data/stapler/103111',
        # 'data/stapler/103292',
        # 'data/stapler/103293',
        # 'data/stapler/103297',
        # 'data/stapler/103299',
        # 'data/stapler/103301',

        # # toilet_tasks
        # 'data/toilet/101320',
        # 'data/toilet/102621',
        # 'data/toilet/102622',
        # 'data/toilet/102630',
        # 'data/toilet/102634',
        # 'data/toilet/102645',
        # 'data/toilet/102648',
        # 'data/toilet/102651',
        # 'data/toilet/102652',
        # 'data/toilet/102658',
    ]
    cfg.task.env_runner.demo_experiment_path = [None for _ in range(1)]
    # cfg.task.env_runner.experiment_name += ['0822-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first' for _ in range(6)]
    # cfg.task.env_runner.experiment_folder += [
    #     "data/diverse_objects_other/open_the_door_7167/task_open_the_door_of_the_storagefurniture_by_its_handle",
    #     "data/diverse_objects_other/open_the_door_7263/task_open_the_door_of_the_storagefurniture_by_its_handle",
    #     "data/diverse_objects_other/open_the_door_7290/task_open_the_door_of_the_storagefurniture_by_its_handle",
    #     "data/diverse_objects_other/open_the_door_7310/task_open_the_door_of_the_storagefurniture_by_its_handle",
    #     "data/diverse_objects_other/open_the_door_12092/task_open_the_door_of_the_storagefurniture_by_its_handle",
    #     "data/diverse_objects_other/open_the_door_12606/task_open_the_door_of_the_storagefurniture_by_its_handle",
    # ]
    # cfg.task.env_runner.demo_experiment_path += [None for _ in range(6)]
    
    load_model_path = args.high_level_ckpt_name
        
    
    num_class = 13 if not args.predict_two_goals else 25
    input_channel = 5 if args.add_one_hot_encoding else 3
    print(args.pointnet_class)
    if args.category_embedding_type == "one_hot":
        embedding_dim = args.num_categories
    elif args.category_embedding_type == "siglip":
        embedding_dim = 768
    else:
        embedding_dim = None
    if args.model_transformer:
        from ptv3.highlevel_ptv3 import HighlevelPTv3
        config_file = "ptv3/configs/highlevel_ptv3.yaml"
        if os.path.exists(config_file):
            model_cfg = OmegaConf.load(config_file)
            
            # if there are no args, we use default values
            if hasattr(args, 'in_channels'):
                model_cfg.in_channels = args.in_channels
            else:
                model_cfg.in_channels = 3  # default value
                
            if hasattr(args, 'patch_size'):
                model_cfg.patch_size = args.patch_size  
            else:
                model_cfg.patch_size = 48  # default value
            pointnet2_model: HighlevelPTv3 = hydra.utils.instantiate(model_cfg.model)
            pointnet2_model = pointnet2_model.to('cuda')
        
    elif not args.model_invariant:
        from test_PointNet2.model import PointNet2_small2, PointNet2, PointNet2_super
        if args.pointnet_class == "PointNet2":
            pointnet2_model = PointNet2_small2(num_classes=num_class).to('cuda')
        elif args.pointnet_class == "PointNet2_large":
            pointnet2_model = PointNet2(num_classes=num_class).to('cuda')
        elif args.pointnet_class == "PointNet2_super":
            pointnet2_model = PointNet2_super(num_classes=num_class, keep_gripper_in_fps=args.keep_gripper_in_fps, input_channel=input_channel, embedding_dim=embedding_dim).to("cuda")
        
    else:
        from test_PointNet2.model_invariant import PointNet2, PointNet2_super, PointNet2_superplus
        if args.pointnet_class == 'PointNet2_large':
            pointnet2_model = PointNet2(num_classes=num_class).to('cuda')
        elif args.pointnet_class == 'PointNet2_super':
            pointnet2_model = PointNet2_super(num_classes=num_class, keep_gripper_in_fps=args.keep_gripper_in_fps, input_channel=input_channel, embedding_dim=embedding_dim).to("cuda")
        elif args.pointnet_class == "PointNet2_superplus":
            pointnet2_model = PointNet2_superplus(num_classes=13).to("cuda")
            
    #High Level Policy Loading    
    if args.model_transformer:
        state_dict = torch.load(load_model_path)['model']
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        pointnet2_model.load_state_dict(state_dict)
    else:
        pointnet2_model.load_state_dict(torch.load(load_model_path))
    pointnet2_model.eval()
    
    checkpoint_dir = "{}/checkpoints/{}".format(exp_dir, checkpoint_name)
    
    save_path = "data/{}/rollout".format(args.eval_exp_name)
    if args.noise is not None:
        save_path = "data/{}_{}_{}".format(args.eval_exp_name, args.noise[0], args.noise[1])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    checkpoint_info = {
        "low_level_policy": checkpoint_dir,
        "low_level_policy_checkpoint": checkpoint_name,
        "high_level_policy_checkpoint": args.high_level_ckpt_name,
    }
    checkpoint_info.update(args.__dict__)
    with open("{}/checkpoint_info.json".format(save_path), "w") as f:
        json.dump(checkpoint_info, f, indent=4)
    cfg.task.env_runner.observation_mode = "act3d_goal_displacement_gripper_to_object"
    cfg.task.dataset.observation_mode = "act3d_goal_displacement_gripper_to_object"
    run_eval_non_parallel(
            cfg, policy, pointnet2_model,
            num_worker, save_path, cat_idx,
            pool=pool, 
            horizon=35,
            exp_beg_idx=0,
            exp_end_idx=25,
            embedding_dim=embedding_dim,
            obj_translation=args.noise,
            output_obj_pcd_only=args.output_obj_pcd_only,
            update_goal_freq=args.update_goal_freq,
            real_world_camera=args.real_world_camera,
            noise_real_world_pcd=args.noise_real_world_pcd,
            randomize_camera=args.randomize_camera,
            pos_ori_imp=args.pos_ori_imp
    )


# python eval_robogen_with_goal_PointNet.py --high_level_ckpt_name /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps/pointnet2_super_model_invariant_2024-09-30_use_75_episodes_200-obj/model_39.pth --eval_exp_name eval_yufei_weighted_displacement_pointnet_large_200_invariant_reproduce --pointnet_class PointNet2_super --model_invariant True