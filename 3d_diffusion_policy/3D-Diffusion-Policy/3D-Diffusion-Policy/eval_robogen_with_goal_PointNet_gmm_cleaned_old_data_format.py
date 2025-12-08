import os
import hydra
import torch
from omegaconf import OmegaConf
from train_ddp import TrainDP3Workspace
from diffusion_policy_3d.common.pytorch_util import dict_apply
from manipulation.utils import build_up_env, save_numpy_as_gif, save_env
import numpy as np
from copy import deepcopy
from termcolor import cprint
from manipulation.robogen_wrapper import RobogenPointCloudWrapper
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
import json
import yaml
import argparse
from typing import Optional
from collections import deque
from manipulation.utils import load_env

def infer_pointnetplus_model(inputs, goal_prediction_model, cat_embedding=None, high_level_args=None, args=None):
    inputs = inputs.to('cuda')
    pred_dict = goal_prediction_model(inputs, cat_embedding) 
    outputs = pred_dict['pred_offsets']
    pred_points = pred_dict['pred_points'] 
    weights = pred_dict['pred_scores'].squeeze(-1)
    inputs = pred_points
    B, N, _, _ = outputs.shape
    outputs = outputs.view(B, N, -1)
    
    if args.output_obj_pcd_only:
        # cprint("using only obj pcd output!", "red")
        weights = weights[:, :-4]
        outputs = outputs[:, :-4, :]
        inputs = inputs[:, :-4, :]

    outputs = outputs.view(B, N, 4, 3)
    
    
    if 'gmm' in high_level_args.articubot and high_level_args.articubot.gmm:
        ### sample an displacement according to the weight
        probabilities = weights  # Must sum to 1
        probabilities = torch.nn.functional.softmax(weights, dim=1)

        # Sample one index based on the probabilities
        if not args.argmax:
            sampled_index = torch.multinomial(probabilities, num_samples=1)
            sampled_index = sampled_index.item()
        else:
            sampled_index = torch.argmax(probabilities.squeeze(0))
        displacement_mean = outputs[:, sampled_index, :, :] # B, 4, 3
        input_point_pos = inputs[:, sampled_index, :] # B, 3
        prediction = input_point_pos.unsqueeze(1) + displacement_mean # B, 4, 3
    else:
        outputs = outputs.view(B, N, 4, 3)
        outputs = outputs + inputs[:, :, :3].unsqueeze(2)
        weights = torch.nn.functional.softmax(weights, dim=1)
        outputs = outputs * weights.unsqueeze(-1).unsqueeze(-1)
        outputs = outputs.sum(dim=1)
        prediction = outputs
        
    return prediction
    

def construct_env(cfg, config_file, env_name, init_state_file, real_world_camera=False, noise_real_world_pcd=False,
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
            )
    env.reset()
    pointcloud_env = RobogenPointCloudWrapper(env, object_name,
                                                num_points=cfg.task.env_runner.num_point_in_pc,
                                                observation_mode=cfg.task.env_runner.observation_mode,
                                                real_world_camera=real_world_camera,
                                                noise_real_world_pcd=noise_real_world_pcd,
                                                )
        
    if randomize_camera:
        pointcloud_env.reset_random_cameras()
        
    env = MultiStepWrapper(pointcloud_env, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps, 
                        max_episode_steps=600, reward_agg_method='sum')
    
    return env
            
def run_eval_non_parallel(cfg, policy, goal_prediction_model, save_path, cat_idx, exp_beg_idx=0,
                          exp_end_idx=1000, embedding_dim=None, horizon=150,  exp_beg_ratio=None, exp_end_ratio=None,
                          dataset_index=None, calculate_distance_from_gt=False, output_obj_pcd_only=False,
                          update_goal_freq=1, real_world_camera=False, noise_real_world_pcd=False,
                          randomize_camera=False, high_level_args=None, args=None):
    
    for dataset_idx, (experiment_folder, experiment_name) in \
        enumerate(zip(cfg.task.env_runner.experiment_folder, cfg.task.env_runner.experiment_name)):
        
        
        if dataset_index is not None:
            dataset_idx = dataset_index

        if calculate_distance_from_gt:
            all_obj_distances = []
            
        init_state_files = []
        config_files = []
        experiment_folder = "{}/{}".format(os.environ['PROJECT_DIR'], experiment_folder)
        experiment_name = experiment_name
        experiment_path = os.path.join(experiment_folder, "experiment", experiment_name)
        all_experiments = os.listdir(experiment_path)
        all_experiments = sorted(all_experiments)
        
                    
        expert_opened_angles = []
        for experiment in all_experiments:
            if "meta" in experiment:
                continue
            
            exp_folder = os.path.join(experiment_path, experiment, "grasp_the_handle_of_the_storage_furniture_door_primitive")
            # print(exp_folder)
            # import pdb; pdb.set_trace()
            if os.path.exists(os.path.join(exp_folder, "label.json")):
                with open(os.path.join(exp_folder, "label.json"), 'r') as f:
                    label = json.load(f)
                if not label['good_traj']: continue
                
            states_path = os.path.join(exp_folder, "states")
            if not os.path.exists(states_path):
                continue
            if len(os.listdir(states_path)) <= 1: #or not os.path.exists(os.path.join(exp_folder, "all.gif")):
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
     
            expert_opened_angle_file = os.path.join(exp_folder, "opened_angle.txt")
            # import pdb; pdb.set_trace()
            if os.path.exists(expert_opened_angle_file):
                with open(expert_opened_angle_file, "r") as f:
                    angles = f.readlines()
                    expert_opened_angle = float(angles[0].lstrip().rstrip())
                    # max_angle = float(angles[-1].lstrip().rstrip())
                    # ratio = expert_opened_angle / max_angle
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
        
        all_distances = []
        all_grasp_distances = []

        if high_level_args is not None and high_level_args.general.category_embedding_type == "siglip":
            siglip_text_features = torch.load("/project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/siglip_text_features.pt")
            
        cat_idx_cuda = torch.tensor(cat_idx).to('cuda')

        for exp_idx, (config_file, init_state_file) in enumerate(zip(config_files, init_state_files)):
                
            env = construct_env(cfg, config_file, "articulated", init_state_file, real_world_camera, noise_real_world_pcd, 
                                randomize_camera)
            obs = env.reset()
            rgb = env.env.render()
            info = env.env._env._get_info()
            
            initial_info = info
            all_rgbs = [rgb]
            first_step_outputs = None
            last_goal = None
            for t in range(1, horizon):
                parallel_input_dict = obs
                parallel_input_dict = dict_apply(parallel_input_dict, lambda x: torch.from_numpy(x).to('cuda'))
                            
                for key in obs:
                    parallel_input_dict[key] = parallel_input_dict[key].unsqueeze(0)
                    
              
                if t == 1 or t % update_goal_freq == 0:
                    with torch.no_grad():
                        pointcloud = parallel_input_dict['point_cloud'][:, -1, :, :]
                        gripper_pcd = parallel_input_dict['gripper_pcd'][:, -1, :]
                        if high_level_args and high_level_args.general.category_embedding_type == "one_hot":
                            cat_embedding = torch.nn.functional.one_hot(cat_idx_cuda, num_classes=embedding_dim).float().to(pointcloud.device)
                        elif high_level_args and high_level_args.general.category_embedding_type == "siglip":
                            cat_embedding = siglip_text_features[cat_idx].float().to(pointcloud.device).unsqueeze(0)
                        else:
                            cat_embedding = None
                            
                        inputs = torch.cat([pointcloud, gripper_pcd], dim=1)    
                        if high_level_args and high_level_args.articubot.add_one_hot_encoding:
                            # for pointcloud, we add (1, 0)
                            # for gripper_pcd, we add (0, 1)
                            pointcloud_one_hot = torch.zeros(pointcloud.shape[0], pointcloud.shape[1], 2).float().to(pointcloud.device)
                            pointcloud_one_hot[:, :, 0] = 1
                            pointcloud_ = torch.cat([pointcloud, pointcloud_one_hot], dim=2)
                            gripper_pcd_one_hot = torch.zeros(gripper_pcd.shape[0], gripper_pcd.shape[1], 2).float().to(pointcloud.device)
                            gripper_pcd_one_hot[:, :, 1] = 1
                            gripper_pcd_ = torch.cat([gripper_pcd, gripper_pcd_one_hot], dim=2)
                            inputs = torch.cat([pointcloud_, gripper_pcd_], dim=1) # B, N+4, 5
                        
                        if args.model_type == "pointnet++":
                            prediction = infer_pointnetplus_model(inputs.permute(0, 2, 1), goal_prediction_model, cat_embedding, high_level_args, args)
                        elif args.model_type == "m2t2":
                            ### TODO: implement m2t2 model inference
                            with torch.no_grad():
                                data_input = {
                                    "inputs": inputs,
                                }
                                # import pdb; pdb.set_trace()
                                prediction, weights = goal_prediction_model.infer(data_input, None)
                                
                        elif args.model_type == 'ptv3':
                            ### TODO: implement ptv3 model inference
                            prediction = infer_pointnetplus_model(inputs, goal_prediction_model, cat_embedding, high_level_args, args)

                        elif args.model_type == '3dfa':
                            prediction_len = 1
                            rgbs = torch.from_numpy(np.zeros((1, 2, 3, 256, 256)))
                            pcds = pointcloud
                            instruction = ["open the door of the storage furniture"]
                            instruction = tokenizer(instruction).cuda()
                            from articubot_3dfa.datasets.articubot_dataset import get_gripper_pos_orient_from_4_points
                            cur_pos, cur_orient = get_gripper_pos_orient_from_4_points(gripper_pcd.cpu().numpy().reshape(4, 3))
                            proprio = np.array([*cur_pos, *cur_orient], dtype=np.float32)
                            proprio = torch.from_numpy(proprio).view(1, 1, 1, 7).cuda() # B, history=1, 1, 8
                            with torch.no_grad():
                                output = high_level_model(
                                            None,
                                            torch.full([1, prediction_len, 1], False).cuda(non_blocking=True),
                                            rgbs.cuda(),
                                            None,
                                            pcds.cuda(),
                                            instruction,
                                            # gripper[:, :, None, :7],
                                            proprio,
                                            run_inference=True
                                        ).view(1, prediction_len, 8)

                            output = output.view(8).detach().cpu().numpy()
                            predicted_pos = output[:3]
                            predicted_quat = output[3:7]
                            open_finger = output[7].round()
                            gripper_q = 0.08 if open_finger > 0 else 0.01

                            from articubot_3dfa.test_trained_model import get_4_points_from_gripper_pos_orient
                            four_points = get_4_points_from_gripper_pos_orient(predicted_pos, predicted_quat, gripper_q)
                            prediction = torch.from_numpy(four_points).float().unsqueeze(0).to('cuda') # B, 4, 3

                        # handle the ambiguity between the two finger points
                        if args.flip_goal:
                            cur_gripper = parallel_input_dict['gripper_pcd'][0, -1].reshape(4, 3)
                            distance_1 = torch.norm(prediction.squeeze(0) - cur_gripper, dim=-1).mean()
                            distance_2 = torch.norm(prediction.squeeze(0)[[0, 2, 1, 3]] - cur_gripper, dim=-1).mean()
                            # print("distance_1: ", distance_1)
                            # print("distance_2: ", distance_2)
                            # import pdb; pdb.set_trace()


                            if distance_1 > distance_2:
                                print("flip the predicted goal")
                                prediction = prediction[:, [0, 2, 1, 3], :]
                        
                        
                        outputs = prediction.unsqueeze(1) # B, history=1, 4, 3
                        # import pdb; pdb.set_trace()
                        
                    last_goal = outputs
                else:
                    outputs = last_goal

                    
                np_predicted_goal = outputs.detach().to('cpu').numpy()
                
                predicted_goal = outputs.repeat(1, 2, 1, 1)

                parallel_input_dict['goal_gripper_pcd'] = predicted_goal

                with torch.no_grad():
                    batched_action = policy.predict_action(parallel_input_dict)
                    gripper_close_actions = batched_action['action'][:, :, -1].detach().cpu().numpy()
                    
                    
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
            gif_save_exp_name = experiment_folder.split("/")[-3] + "/" + experiment_folder.split("/")[-2]
            gif_save_folder = "{}/{}".format(save_path, gif_save_exp_name)                 
            if not os.path.exists(gif_save_folder):
                os.makedirs(gif_save_folder, exist_ok=True)

            with open("{}/opened_joint_angles.json".format(gif_save_folder), "w") as f:
                json.dump(opened_joint_angles, f, indent=4)
   
            gif_save_path = "{}/{}_{}.gif".format(gif_save_folder, exp_idx, 
                    float(info["improved_joint_angle"][-1]))
            save_numpy_as_gif(np.array(all_rgbs), gif_save_path)
            # with open("{}/opened_joint_angles_{}.json".format(save_path, dataset_idx), "w") as f:
            #     json.dump(opened_joint_angles, f, indent=4)
            
            # gif_save_exp_name = experiment_folder.split("/")[-2]
            # gif_save_folder = "{}/{}".format(save_path, gif_save_exp_name)
            # if not os.path.exists(gif_save_folder):
            #     os.makedirs(gif_save_folder, exist_ok=True)
            # gif_save_path = "{}/{}_{}.gif".format(gif_save_folder, exp_idx, 
            #         float(info["improved_joint_angle"][-1]))
            
            # save_numpy_as_gif(np.array(all_rgbs), gif_save_path)

        if calculate_distance_from_gt:
            print("average distance: {}".format(np.mean(all_distances)))
            print("average grasp distance: {}".format(np.mean(all_grasp_distances)))
            all_obj_distances.append(all_distances)

    if calculate_distance_from_gt:
        print("average distance over all objects: {}".format(np.mean(all_obj_distances)))

def load_high_level_model(path):
    from omegaconf import OmegaConf
    import json
    ckpt_path = os.path.dirname(path)
    config_path = os.path.join(ckpt_path, "config.json")
    cfg = json.load(open(config_path, "r"))
    cfg = OmegaConf.create(cfg)
    args = cfg
    
    device = torch.device("cuda")
    general_args = args.general
    input_channel = 5 if general_args.add_one_hot_encoding else 3
    output_dim = 13 
    from test_PointNet2.model_invariant import PointNet2_super_multitask
    
    if "category_embedding_type" not in general_args:
        general_args.category_embedding_type = None
    if general_args.category_embedding_type == "one_hot":
        embedding_dim = args.num_categories
    elif general_args.category_embedding_type == "siglip":
        embedding_dim = 768
    else:
        embedding_dim = None
    
    model = PointNet2_super_multitask(num_classes=output_dim, keep_gripper_in_fps=general_args.keep_gripper_in_fps, input_channel=input_channel,
                                      first_sa_point=general_args.get("first_sa_point", 2048),
                                      fp_to_full=general_args.get("fp_to_full", False),
                                      replace_bn_w_gn=general_args.get("replace_bn_with_gn", False),
                                      replace_bn_w_in=general_args.get("replace_bn_with_in", False),
                                      embedding_dim=embedding_dim,
                                      film_in_sa_and_fp=general_args.get("film_in_sa_and_fp", False),
                                      embedding_as_input=general_args.get("embedding_as_input", False),
                                      replace_bn_w_ln=general_args.get("replace_bn_with_ln", False),
                                      ).to(device)
    
    model.load_state_dict(torch.load(path, map_location=device)['model'])
    print("Successfully load model from: ", path)
    model.eval()
    # model.train()
        
    return model, args

def load_3dfa_models(args, checkpoint_path):
    print("Loading model from", checkpoint_path, flush=True)

    ### TODO: change to be the actual 3dfa package path
    from articubot_3dfa.modeling.policy.denoise_actor_pcd import DenoiseActor as DenoiseActorpcd
    from articubot_3dfa.modeling.encoder.text import fetch_tokenizers

    model = DenoiseActorpcd(
        backbone=args.backbone,
        num_vis_instr_attn_layers=args.num_vis_instr_attn_layers,
        fps_subsampling_factor=args.fps_subsampling_factor,
        embedding_dim=args.embedding_dim,
        num_attn_heads=args.num_attn_heads,
        nhist=args.num_history,
        nhand=2 if args.bimanual else 1,
        num_shared_attn_layers=args.num_shared_attn_layers,
        relative=args.relative_action,
        rotation_format=args.rotation_format,
        denoise_timesteps=args.denoise_timesteps,
        denoise_model=args.denoise_model
    )

    # Load model weights
    model_dict = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    model_dict_weight = {}
    for key in model_dict["weight"]:
        _key = key[7:]
        model_dict_weight[_key] = model_dict["weight"][key]
    model.load_state_dict(model_dict_weight, strict=False)
    model.eval()

    tokenizer = fetch_tokenizers("clip")

    return model.cuda(), tokenizer


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--low_level_exp_dir', type=str, default=None)
    parser.add_argument('--low_level_ckpt_name', type=str, default=None)
    parser.add_argument("--high_level_ckpt_name", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="pointnet++")
    parser.add_argument("--eval_exp_name", type=str, default=None)
    parser.add_argument("--use_predicted_goal", type=bool, default=True)
    parser.add_argument('--output_obj_pcd_only', action='store_true')
    parser.add_argument("--update_goal_freq", type=int, default=1)
    parser.add_argument("--noise_real_world_pcd", type=int, default=0)
    parser.add_argument("--randomize_camera", type=int, default=0)
    parser.add_argument("--real_world_camera", type=int, default=0)
    parser.add_argument('--keep_gripper_in_fps', type=int, default=0)
    parser.add_argument('--add_one_hot_encoding', type=int, default=0)
    parser.add_argument('--fixed_variance', type=float, default=0.05)
    parser.add_argument('--argmax', type=int, default=1)
    parser.add_argument('--gmm', type=int, default=0)
    parser.add_argument('--flip_goal', type=int, default=0)
    parser.add_argument('--exp_dir', type=str, help='Experiment directory')
    parser.add_argument('--num_categories', type=int, default=7)
    parser.add_argument('--category_embedding_type', type=str, default="none")
    args = parser.parse_args()
    
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

    policy = deepcopy(workspace.model)
    if workspace.cfg.training.use_ema:
        policy = deepcopy(workspace.ema_model)
    policy.eval()
    policy.reset()
    policy = policy.to('cuda')
    
    if 'diverse_objects' in args.exp_dir:
        cfg.task.env_runner.experiment_name = ['0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first' for _ in range(1)]
    else: 
        cfg.task.env_runner.experiment_name = ['165-obj' for _ in range(1)]

    cfg.task.env_runner.experiment_folder = [
        args.exp_dir,
    ]

    if args.model_type == 'pointnet++':
        load_model_path = args.high_level_ckpt_name
        high_level_model, model_args = load_high_level_model(load_model_path)
    elif args.model_type == 'm2t2':
        from m2t2.m2t2_articubot import M2T2
        load_model_path = args.high_level_ckpt_name
        load_model_dir = os.path.dirname(load_model_path)
        load_config = os.path.join(load_model_dir, "config.yaml")
        m2t2_config = OmegaConf.load(load_config)
        high_level_model = M2T2.from_config(m2t2_config.m2t2)
        ckpt = torch.load(load_model_path)
        high_level_model.load_state_dict(ckpt['model'])
        high_level_model = high_level_model.cuda().eval()
        model_args = None
        # import pdb; pdb.set_trace()
    elif args.model_type == 'ptv3':
        # import pdb; pdb.set_trace()
        from ptv3.highlevel_ptv3 import HighlevelPTv3
        import hydra
        
        load_model_path = args.high_level_ckpt_name
        load_model_dir = os.path.dirname(load_model_path)
        load_config = os.path.join(load_model_dir, ".hydra/config.yaml") # TODO: implement the overrides
        model_cfg = OmegaConf.load(load_config)
        pointnet2_model: HighlevelPTv3 = hydra.utils.instantiate(model_cfg.model)
        pointnet2_model = pointnet2_model.to('cuda')
        
        state_dict = torch.load(load_model_path)['model']
        pointnet2_model.load_state_dict(state_dict)
        high_level_model = pointnet2_model
        high_level_model.eval()
        model_args = model_cfg
    elif args.model_type == '3dfa':
        load_model_path = "/project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/articubot_3dfa/train_logs/2025-0817-test_articubot_50/best.pth"
        load_model_dir = os.path.dirname(load_model_path)
        load_config = os.path.join(load_model_dir, "config.yaml") # TODO: implement the overrides
        model_cfg = OmegaConf.load(load_config)
        high_level_model, tokenizer = load_3dfa_models(model_cfg, load_model_path)
        model_args = None
        
    
    checkpoint_dir = "{}/checkpoints/{}".format(exp_dir, checkpoint_name)
    save_path = "data_yufei/{}".format(args.eval_exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
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
            cfg, policy, high_level_model,
            save_path, 
            cat_idx,
            horizon=35,
            exp_beg_idx=0,
            exp_end_idx=25,
            embedding_dim=None,
            output_obj_pcd_only=args.output_obj_pcd_only,
            update_goal_freq=args.update_goal_freq,
            real_world_camera=args.real_world_camera,
            noise_real_world_pcd=args.noise_real_world_pcd,
            randomize_camera=args.randomize_camera,
            high_level_args=model_args,
            args=args
    )


# python eval_robogen_with_goal_PointNet.py --high_level_ckpt_name /project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps/pointnet2_super_model_invariant_2024-09-30_use_75_episodes_200-obj/model_39.pth --eval_exp_name eval_yufei_weighted_displacement_pointnet_large_200_invariant_reproduce --pointnet_class PointNet2_super --model_invariant True