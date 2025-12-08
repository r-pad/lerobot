import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from train import TrainDP3Workspace
from diffusion_policy_3d.common.pytorch_util import dict_apply
from manipulation.utils import build_up_env, save_numpy_as_gif
import pybullet as p
import numpy as np
from copy import deepcopy
import pytorch3d.ops as torch3d_ops
import sys
from termcolor import cprint
from manipulation.robogen_wrapper import RobogenPointCloudWrapper
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from manipulation.gpt_reward_api import get_joint_state
import tqdm
import json
import time
import yaml

def eval(cfg):
    for i in range(12):
        rgbs = main(cfg, i)
        save_numpy_as_gif(np.array(rgbs), f"open_door_{i}.gif")

# @hydra.main(
#     version_base=None,
#     config_path=str(pathlib.Path(__file__).parent.joinpath(
#         'diffusion_policy_3d', 'config'))
# )
def main():
    # print(str(pathlib.Path(__file__).parent.joinpath(
    #     'diffusion_policy_3d', 'config')))
    with hydra.initialize(config_path='diffusion_policy_3d/config'):  # same config_path as used by @hydra.main
        recomposed_config = hydra.compose(
            config_name="dp3.yaml",  # same config_name as used by @hydra.main
            overrides=OmegaConf.load("/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first/2024.05.11/14.45.04_train_dp3_robogen_open_door/.hydra/overrides.yaml"),
            # overrides=OmegaConf.load("/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0512-vary-obj-loc-ori-init-more-points-longer-horizon-depend/2024.05.12/01.50.59_train_dp3_robogen_open_door/.hydra/overrides.yaml"),
            # overrides=OmegaConf.load("/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0513-vary-obj-loc-ori-init-joint-angle/2024.05.13/02.21.51_train_dp3_robogen_open_door/.hydra/overrides.yaml"),
        )
    cfg = recomposed_config
    # cfg.task.env_runner.num_point_in_pc = 4500
    # cfg.task.env_runner.num_point_in_pc = 8000
    # cfg.task.env_runner.use_joint_angle = 0
    
    workspace = TrainDP3Workspace(cfg)
    checkpoint_dir = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0511-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first/2024.05.11/14.45.04_train_dp3_robogen_open_door/checkpoints/latest.ckpt"
    # checkpoint_dir = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0512-vary-obj-loc-ori-init-more-points-longer-horizon-depend/2024.05.12/01.50.59_train_dp3_robogen_open_door/checkpoints/epoch-600-test_mean_score=-0.046.ckpt"
    # checkpoint_dir = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0513-vary-obj-loc-ori-init-joint-angle/2024.05.13/02.21.51_train_dp3_robogen_open_door/checkpoints/epoch-2100-test_mean_score=-0.020.ckpt"
    workspace.load_checkpoint(path=checkpoint_dir)

    policy = deepcopy(workspace.model)
    if workspace.cfg.training.use_ema:
        policy = deepcopy(workspace.ema_model)
    policy.eval()

    experiment_folder = cfg.task.env_runner.experiment_folder
    experiment_name = cfg.task.env_runner.experiment_name
    
    after_reaching_init_state_files = []
    init_state_files = []
    config_files = []
    experiment_folder = "{}/{}".format(os.environ['PROJECT_DIR'], experiment_folder)
    experiment_name = experiment_name
    experiment_path = os.path.join(experiment_folder, "experiment", experiment_name)
    all_experiments = os.listdir(experiment_path)
    all_experiments = sorted(all_experiments)

    all_substeps_path = os.path.join(experiment_folder, "substeps.txt")
    with open(all_substeps_path, "r") as f:
        substeps = f.readlines()
        first_step = substeps[0].lstrip().rstrip()
    

    expert_opened_angles = []
    for experiment in all_experiments:
        if "meta" in experiment:
            continue
        
        first_step_folder = first_step.replace(" ", "_") + "_primitive"
        first_step_folder = os.path.join(experiment_path, experiment, first_step_folder)
        if os.path.exists(os.path.join(first_step_folder, "label.json")):
            with open(os.path.join(first_step_folder, "label.json"), 'r') as f:
                label = json.load(f)
            if not label['good_traj']: continue
            
        first_stage_states_path = os.path.join(first_step_folder, "states")
        expert_states = os.listdir(first_stage_states_path)
        if len(expert_states) == 0:
            continue
            
        expert_opened_angle_file = os.path.join(experiment_path, experiment, first_step_folder, "opened_angle.txt")
        if os.path.exists(expert_opened_angle_file):
            with open(expert_opened_angle_file, "r") as f:
                angles = f.readlines()
                expert_opened_angle = float(angles[0].lstrip().rstrip())
                max_angle = float(angles[-1].lstrip().rstrip())
                ratio = expert_opened_angle / max_angle
            if ratio < 0.5:
                continue
        expert_opened_angles.append(expert_opened_angle)
        
        first_stage_states_path = os.path.join(first_step_folder, "states")
        stage_lengths = os.path.join(first_step_folder, "stage_lengths.json")
        with open(stage_lengths, "r") as f:
            stage_lengths = json.load(f)
        
        if 'stage' in stage_lengths:
            reaching_phase = stage_lengths.get('open_gripper', 0) + stage_lengths['grasp_handle']
        else:
            reaching_phase = stage_lengths['reach_handle']
        after_init_state_file = os.path.join(first_stage_states_path, "state_{}.pkl".format(reaching_phase))
        after_reaching_init_state_files.append(after_init_state_file)
        init_state_file = os.path.join(first_stage_states_path, "state_0.pkl")
        init_state_files.append(init_state_file)
        config_file = os.path.join(experiment_path, experiment, "task_config.yaml")
        config_files.append(config_file)
                
    after_reaching_init_state_files = after_reaching_init_state_files
    config_files = config_files
    init_state_files = init_state_files

    opened_joint_angles = {}
    # config_files = config_files[9:10]
    # init_state_files = init_state_files[9:10]
    for idx in tqdm.tqdm(range(len(config_files))):
        cprint("config path: {}".format(config_files[idx]), "green")
        config_file = config_files[idx]
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        solution_path = [x['solution_path'] for x in config if "solution_path" in x][0]
        all_substeps_path = os.path.join(os.environ['PROJECT_DIR'], solution_path, "substeps.txt")
        with open(all_substeps_path, "r") as f:
            substeps = f.readlines()
            first_step = substeps[0].lstrip().rstrip()
        
        env, _ = build_up_env(
            config_files[idx],
            solution_path,
            first_step.replace(" ", "_"),
            init_state_files[idx] if not cfg.task.env_runner.start_after_reaching else after_reaching_init_state_files[idx],
            render=False, 
            randomize=False,
            obj_id=0,
            horizon=600,
        )
        
        object_name = "StorageFurniture"
        env.reset()
        
        pointcloud_env = RobogenPointCloudWrapper(env, object_name, in_gripper_frame=cfg.task.env_runner.in_gripper_frame, 
                                                  gripper_num_points=cfg.task.env_runner.gripper_num_points, add_contact=cfg.task.env_runner.add_contact,
                                                  num_points=cfg.task.env_runner.num_point_in_pc,
                                                  use_joint_angle=cfg.task.env_runner.use_joint_angle, 
                                                  use_segmask=cfg.task.env_runner.use_segmask,
                                                  only_handle_points=cfg.task.env_runner.only_handle_points,
                                                  )
        
        env = MultiStepWrapper(pointcloud_env, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps, 
                            max_episode_steps=600, reward_agg_method='sum')
        obs = env.reset()
        import pdb; pdb.set_trace()
        policy.reset()
        final_rgbs = []
        episode_reward = 0
        horizon = 150
        for _ in tqdm.tqdm(range(horizon)):
            np_obs_dict = dict(obs)
            obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to('cuda'))
            # run policy
            beg = time.time()
            with torch.no_grad():
                obs_dict_input = {}  # flush unused keys
                obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                action_dict = policy.predict_action(obs_dict_input)
            end = time.time()
            np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())
            action = np_action_dict['action'].squeeze(0)
            beg = time.time()
            obs, reward, done, info = env.step(action)
            end = time.time()
            done = np.all(done)
            episode_reward += reward
            beg = time.time()
            final_rgbs.append(env.env.render())
            end = time.time()
            if done:
                break

        joint_angle = float(info["improved_joint_angle"][-1])
        cprint(f"improved joint angle: {joint_angle}", "blue")
        max_improvement = expert_opened_angles[idx] - env.env._env.init_joint_angle
        pointcloud_env._env.close()
        opened_joint_angles[config_file] = [joint_angle, max_improvement, joint_angle / max_improvement]
    
        checkpoint_name_start_idx = checkpoint_dir.find("3D-Diffusion-Policy/data/")  + len("3D-Diffusion-Policy/data/")
        save_path = "data/eval_results/{}".format(checkpoint_dir[checkpoint_name_start_idx:].replace("/", "_"))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open("{}/opened_joint_angles.json".format(save_path), "w") as f:
            json.dump(opened_joint_angles, f, indent=4)
        save_numpy_as_gif(np.array(final_rgbs), "{}/open_door_{}_{:.3f}.gif".format(save_path, idx, joint_angle))



if __name__ == "__main__":
    # import cProfile, pstats, io
    # pr = cProfile.Profile()
    # pr.enable()
    main()
    # pr.disable()
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    # ps.print_stats(50)
    # print(s.getvalue())
    # ps = pstats.Stats(pr, stream=s).sort_stats('time')
    # ps.print_stats(50)
    # print(s.getvalue())