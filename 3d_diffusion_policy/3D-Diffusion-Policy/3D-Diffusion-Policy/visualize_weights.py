import os
import hydra
import torch
from omegaconf import OmegaConf
from train_ddp import TrainDP3Workspace
from diffusion_policy_3d.common.pytorch_util import dict_apply
from manipulation.utils import build_up_env
import numpy as np
from copy import deepcopy
from manipulation.robogen_wrapper import RobogenPointCloudWrapper
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
import json
import yaml
import argparse
from typing import Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio.v2 as imageio
from io import BytesIO
from tqdm import tqdm

def construct_env(cfg, config_file, env_name, init_state_file, obj_translation=None, real_world_camera=False, noise_real_world_pcd=False,
                  randomize_camera=False):
    config = yaml.safe_load(open(config_file, "r"))
    link_name = 'link_0'
    for config_dict in config:
        if 'name' in config_dict:
            object_name = config_dict['name'].lower()
        if 'link_name' in config_dict:
            link_name = config_dict['link_name']
    env, _ = build_up_env(
                    task_config=config_file,
                    env_name=env_name,
                    object_name=object_name,
                    link_name=link_name,
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

def run_eval_non_parallel(cfg, goal_prediction_model, save_path, cat_idx, exp_beg_idx=0,
                          exp_end_idx=1000, exp_beg_ratio=None, exp_end_ratio=None,
                          dataset_index=None, output_obj_pcd_only=False, obj_translation: Optional[list]= None, real_world_camera=False, noise_real_world_pcd=False,
                          randomize_camera=False):
    if args.category_embedding_type == "siglip":
            siglip_text_features = torch.load("/project_data/held/chenyuah/RoboGen-sim2real/siglip_text_features.pt").to('cuda')
    cat_idx_cuda = torch.tensor(cat_idx).to('cuda')
    for dataset_idx, (experiment_folder, experiment_name, demo_experiment_path) in enumerate(zip(cfg.task.env_runner.experiment_folder, cfg.task.env_runner.experiment_name, cfg.task.env_runner.demo_experiment_path)):
        
        if dataset_index is not None:
            dataset_idx = dataset_index
        
        experiment_folder = "{}/{}".format(os.environ['PROJECT_DIR'], experiment_folder)
        experiment_name = experiment_name
        experiment_path = os.path.join(experiment_folder, "experiment", experiment_name)
        all_experiments = os.listdir(experiment_path)
        all_experiments = sorted(all_experiments)
        
        if demo_experiment_path is not None:
            all_subfolder = os.listdir(demo_experiment_path)
            for string in ["action_dist", "demo_rgbs", "all_demo_path.txt", "meta_info.json", 'example_pointcloud']:
                if string in all_subfolder:
                    all_subfolder.remove(string)
            all_subfolder = sorted(all_subfolder)
            all_experiments = all_subfolder

        if exp_end_ratio is not None:
            exp_end_idx = int(exp_end_ratio * len(all_experiments))
        if exp_beg_ratio is not None:
            exp_beg_idx = int(exp_beg_ratio * len(all_experiments))
        
        expert_opened_angles = []
        selected_experiments = []
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
            selected_experiments.append(experiment)
        angle_threshold = np.quantile(expert_opened_angles, 0.1)
        selected_idx = [i for i, angle in enumerate(expert_opened_angles) if angle > angle_threshold]
        all_experiments = [selected_experiments[i] for i in selected_idx]
        all_experiments = all_experiments[exp_beg_idx:exp_end_idx]
        # all_experiments = ['2025-05-13-08-46-19']
        # print(f"Evaluating {len(all_experiments)} experiments: {all_experiments}")
        cnt = 0
        avg_error = []

        for experiment in all_experiments:
            exp_folder = os.path.join(experiment_path, experiment)
            states_path = os.path.join(exp_folder, "states")
            stage_lengths = os.path.join(exp_folder, "stage_lengths.json")
            expert_states = [f for f in os.listdir(states_path) if f.startswith("state")]
            with open(stage_lengths, "r") as f:
                stage_lengths = json.load(f)
            open_time_idx = stage_lengths['reach_handle'] + stage_lengths["reach_to_contact"] + stage_lengths["close_gripper"]
            if 'stage' in stage_lengths:
                reaching_phase = stage_lengths.get('open_gripper', 0) + stage_lengths['grasp_handle']
            else:
                reaching_phase = stage_lengths['reach_handle']

            config_file = os.path.join(experiment_path, experiment, "task_config.yaml")

            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            for config_dict in config:
                if 'name' in config_dict:
                    object_name = config_dict['name'].lower()

            goal_stage = 'first'

            all_pointclouds = []
            all_gripper_pcds = []
            all_weights = []
            all_outputs = []
            all_errors = []
            with tqdm(total=len(expert_states), desc=f"Processing {experiment}") as pbar:
                for state_idx in range(len(expert_states)):
                    pbar.update(1)
                    if state_idx % 4 != 0:
                        continue
                    # print(f"Processing {state_idx}/{len(expert_states)}")
                    state_file = os.path.join(states_path, f"state_{state_idx}.pkl")
                    env = construct_env(cfg, config_file, "articulated", state_file, obj_translation, real_world_camera, noise_real_world_pcd, 
                                    randomize_camera)
                    
                    obs = env.reset()
                    env.env._env.close()
                    parallel_input_dict = dict_apply(obs, lambda x: torch.from_numpy(x).to('cuda'))
                    for key in obs:
                        parallel_input_dict[key] = parallel_input_dict[key].unsqueeze(0)
                    with torch.no_grad():
                        pointcloud = parallel_input_dict['point_cloud'][:, -1, :, :]
                        gripper_pcd = parallel_input_dict['gripper_pcd'][:, -1, :]
                        if args.category_embedding_type == "one_hot":
                            cat_embedding = torch.nn.functional.one_hot(cat_idx_cuda, num_classes=embedding_dim).float()
                        elif args.category_embedding_type == "siglip":
                            cat_embedding = siglip_text_features[cat_idx].float()
                        else:
                            cat_embedding = None
                        if not args.predict_two_goals:
                            inputs = torch.cat([pointcloud, gripper_pcd], dim=1)
                        else:
                            inputs = pointcloud
                            
                        if args.add_one_hot_encoding:
                            pointcloud_one_hot = torch.zeros(pointcloud.shape[0], pointcloud.shape[1], 2).float().to(pointcloud.device)
                            pointcloud_one_hot[:, :, 0] = 1
                            pointcloud_ = torch.cat([pointcloud, pointcloud_one_hot], dim=2)
                            gripper_pcd_one_hot = torch.zeros(gripper_pcd.shape[0], gripper_pcd.shape[1], 2).float().to(pointcloud.device)
                            gripper_pcd_one_hot[:, :, 1] = 1
                            gripper_pcd_ = torch.cat([gripper_pcd, gripper_pcd_one_hot], dim=2)
                            inputs = torch.cat([pointcloud_, gripper_pcd_], dim=1) # B, N+4, 5
                            
                        inputs = inputs.to('cuda')
                        inputs_ = inputs.permute(0, 2, 1)
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
                            
                            if state_idx >= reaching_phase:
                                goal_stage = 'second'
                            if goal_stage == 'first':
                                outputs = outputs[:, :, :4, :]
                            elif goal_stage == 'second':
                                outputs = outputs[:, :, 4:, :]
                                
                        outputs = outputs + inputs[:, :, :3].unsqueeze(2)
                        weights = torch.nn.functional.softmax(weights, dim=1)
                        outputs = outputs * weights.unsqueeze(-1).unsqueeze(-1)
                        outputs = outputs.sum(dim=1)
                        outputs = outputs.unsqueeze(1)
                        # print(outputs.shape)

                    # save data
                    all_pointclouds.append(pointcloud.squeeze(0).cpu().numpy())
                    all_gripper_pcds.append(gripper_pcd.squeeze(0,1).cpu().numpy())
                    all_weights.append(weights[0].cpu().numpy())
                    all_outputs.append(outputs.squeeze(0,1).cpu().numpy())
                    
    
            goal_gripper_pcd_at_grasping = all_gripper_pcds[min(open_time_idx, len(all_gripper_pcds)-1)]
            goal_gripper_pcd_at_end =  all_gripper_pcds[-1]

            frames_weight = []
            frames_goal = []
            with tqdm(total=len(all_pointclouds), desc=f"Visualizing {experiment}") as pbar:
                for idx, (pointcloud, weight, output, gripper_pc) in enumerate(zip(all_pointclouds, all_weights, all_outputs, all_gripper_pcds)):
                    goal_gripper_pcd = goal_gripper_pcd_at_grasping if idx < open_time_idx else goal_gripper_pcd_at_end
                    all_errors.append(((output - goal_gripper_pcd) ** 2).sum(axis=1).mean().item())
                    # visualize weights
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    if pointcloud.shape[0] != weight.shape[0]:
                        import pdb; pdb.set_trace()
                    ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], c=weight, cmap='seismic')
                    ax.view_init(elev=24, azim=-117) 
                    ax.axis("equal")
                    ax.set_title("frame {}".format(idx))
                    # plt.show()
                    buf = BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    frames_weight.append(imageio.imread(buf))
                    buf.close()
                    plt.close()

                    # visualize gripper pcd
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], color='lightgray', s=1, label='PointCloud')
                    ax.scatter(output[:, 0], output[:, 1], output[:, 2], color='blue', s=30, label='Output')
                    ax.scatter(goal_gripper_pcd[:, 0], goal_gripper_pcd[:, 1], goal_gripper_pcd[:, 2], color='green', s=30, label='Goal')
                    ax.scatter(gripper_pc[:, 0], gripper_pc[:, 1], gripper_pc[:, 2], color='red', s=30, label='Gripper PCD')
                    ax.view_init(elev=24, azim=-117) 
                    ax.axis("equal")
                    ax.set_title("frame {}".format(idx))
                    buf = BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    frames_goal.append(imageio.imread(buf))
                    buf.close()
                    plt.close()
                    pbar.update(1)

            # save gif
            if not os.path.exists(os.path.join(save_path, f'{experiment_path.split("/")[-4]}/{experiment_path.split("/")[-3]}', 'weights_visualization')):
                os.makedirs(os.path.join(save_path, f'{experiment_path.split("/")[-4]}/{experiment_path.split("/")[-3]}', 'weights_visualization'))
            gif_save_path = os.path.join(save_path, f'{experiment_path.split("/")[-4]}/{experiment_path.split("/")[-3]}', 'weights_visualization', f'{cnt}.gif')
            imageio.mimsave(gif_save_path, frames_weight, duration=0.05)
            print(f"Weight GIF saved to {gif_save_path}")

            if not os.path.exists(os.path.join(save_path, f'{experiment_path.split("/")[-4]}/{experiment_path.split("/")[-3]}', 'goal_visualization')):
                os.makedirs(os.path.join(save_path, f'{experiment_path.split("/")[-4]}/{experiment_path.split("/")[-3]}', 'goal_visualization'))
            gif_save_path = os.path.join(save_path, f'{experiment_path.split("/")[-4]}/{experiment_path.split("/")[-3]}', 'goal_visualization', f'{cnt}.gif')
            imageio.mimsave(gif_save_path, frames_goal, duration=0.05)
            print(f"Goal GIF saved to {gif_save_path}")
            # compute average error
            avg_error.append(sum(all_errors) / len(all_errors))
            cnt += 1
        save_avg_error_path = os.path.join(save_path, f'{experiment_path.split("/")[-4]}/{experiment_path.split("/")[-3]}', 'error.json')
        if not os.path.exists(os.path.join(save_path, f'{experiment_path.split("/")[-4]}/{experiment_path.split("/")[-3]}')):
            os.makedirs(os.path.join(save_path, f'{experiment_path.split("/")[-4]}/{experiment_path.split("/")[-3]}'))
        err_data = {
            "avg": sum(avg_error) / len(avg_error),
            "all_errors": avg_error,
        }
        with open(save_avg_error_path, 'w') as f:
            json.dump(err_data, f, indent=4)

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

        # # foldingchair_tasks
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

    if not args.model_invariant:
        from test_PointNet2.model import PointNet2_small2, PointNet2, PointNet2_super
        if args.pointnet_class == "PointNet2":
            pointnet2_model = PointNet2_small2(num_classes=num_class).to('cuda')
        elif args.pointnet_class == "PointNet2_large":
            pointnet2_model = PointNet2(num_classes=num_class).to('cuda')
        elif args.pointnet_class == "PointNet2_super":
            pointnet2_model = PointNet2_super(num_classes=num_class, keep_gripper_in_fps=args.keep_gripper_in_fps, input_channel=input_channel).to("cuda")
        
    else:
        from test_PointNet2.model_invariant import PointNet2, PointNet2_super, PointNet2_superplus
        if args.pointnet_class == 'PointNet2_large':
            pointnet2_model = PointNet2(num_classes=num_class).to('cuda')
        elif args.pointnet_class == 'PointNet2_super':
            pointnet2_model = PointNet2_super(num_classes=num_class, keep_gripper_in_fps=args.keep_gripper_in_fps, input_channel=input_channel, embedding_dim=embedding_dim).to("cuda")
        elif args.pointnet_class == "PointNet2_superplus":
            pointnet2_model = PointNet2_superplus(num_classes=13).to("cuda")
            
    #High Level Policy Loading    
    pointnet2_model.load_state_dict(torch.load(load_model_path))
    pointnet2_model.eval()
    
    checkpoint_dir = "{}/checkpoints/{}".format(exp_dir, checkpoint_name)
    
    save_path = "data/{}".format(args.eval_exp_name)
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
            cfg, pointnet2_model, save_path, cat_idx,
            exp_beg_idx=0,
            exp_end_idx=10,
            obj_translation=args.noise,
            output_obj_pcd_only=args.output_obj_pcd_only,
            real_world_camera=args.real_world_camera,
            noise_real_world_pcd=args.noise_real_world_pcd,
            randomize_camera=args.randomize_camera,
    )


