#!/usr/bin/env python
"""Replay one trajectory from YingYuan0414/debug_dataset_syringe_3 on the Franka+LEAP robot."""

import argparse
import pathlib
import time
from pathlib import Path
import dill
import hydra

import numpy as np
import pyarrow.parquet as pq
import torch
import pytorch3d.transforms as transforms

from lerobot.common.robot_devices.cameras.configs import AzureKinectCameraConfig
from lerobot.common.robot_devices.robots.configs import FrankaLeapRobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.policies.robot_adapters import FrankaLeapAdapter
from lerobot.common.robot_devices.control_utils import add_eef_pose
import sys
from collections import deque

import sys
import select
import termios
import tty
import threading


class RuntimeFlags:
    def __init__(self):
        self.gravity_comp = False
        self.quit = False


flags = RuntimeFlags()


def keyboard_listener():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)

        while not flags.quit:

            dr, _, _ = select.select([sys.stdin], [], [], 0.05)

            if dr:
                key = sys.stdin.read(1)

                if key == "g":
                    flags.gravity_comp = True
                    print("\n[KEYBOARD] Gravity compensation ENABLED")

                elif key == "d":
                    flags.gravity_comp = False
                    print("\n[KEYBOARD] Gravity compensation DISABLED")

                elif key == "q":
                    flags.quit = True
                    print("\n[KEYBOARD] Quit requested")

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


threading.Thread(target=keyboard_listener, daemon=True).start()

# syringe
# NONE_ABS_POLICY_ROOT = "~/Desktop/lerobot_restore/lerobot/outputs/train/real_syringe-dp3-none-abs_seed0/checkpoints/latest.ckpt"  # for loading policy config and checkpoints
# GOAL_ABS_POLICY_ROOT = "~/Desktop/lerobot_restore/lerobot/outputs/train/real_syringe-dp3-goal-abs_seed0/checkpoints/latest.ckpt"  # for loading policy config and checkpoints
# spraybottle
NONE_ABS_POLICY_ROOT = "~/Desktop/lerobot_restore/lerobot/outputs/train/real_syringe0515-dp3-none-abs_seed0/checkpoints/latest.ckpt"  # for loading policy config and checkpoints
GOAL_ABS_POLICY_ROOT = "~/Desktop/lerobot_restore/lerobot/outputs/train/real_syringe0515-dp3-goal-abs_seed0/checkpoints/latest.ckpt"

NONE_REL_POLICY_ROOT = "~/Desktop/lerobot_restore/lerobot/outputs/train/real_syringe-dp3-none-rel-diff4_seed0/checkpoints/latest.ckpt"  # for loading policy config and checkpoints
GOAL_REL_POLICY_ROOT = "~/Desktop/lerobot_restore/lerobot/outputs/train/real_syringe-dp3-goal-rel-diff2_seed0/checkpoints/latest.ckpt"
GOAL_DELTA_POLICY_ROOT = "~/Desktop/lerobot_restore/lerobot/outputs/train/real_syringe-dp3-goal-delta_seed0/checkpoints/latest.ckpt"
NONE_DELTA_POLICY_ROOT = None

# syringe
# GOAL_ABS_PRETRAIN_POLICY = "~/Desktop/lerobot_restore/lerobot/outputs/train/real_syringe-dp3-finetune-abs_seed0/checkpoints/latest.ckpt"
# spraybottle
GOAL_ABS_PRETRAIN_POLICY = "~/Desktop/lerobot_restore/lerobot/outputs/train/real_syringe0515-dp3-finetune-dummyaa_seed0/checkpoints/latest.ckpt"

# syringe
DEFAULT_DATASET_ROOT = Path(
    "~/.cache/huggingface/lerobot/YingYuan0414/syringe_0515_5"
).expanduser()
# spraybottle
# DEFAULT_DATASET_ROOT = Path(
#     "~/.cache/huggingface/lerobot/YingYuan0414/spraybottle_8"
# ).expanduser()

DEFAULT_FPS = 15  # matches meta/info.json


def rel_to_abs(rel_action, prev_target):
    obs_rot6d, obs_pos, obs_tail = prev_target[:6], prev_target[6:9], prev_target[9:]
    act_rot6d, act_pos, act_tail = rel_action[:6], rel_action[6:9], rel_action[9:]

    r_obs = transforms.rotation_6d_to_matrix(torch.from_numpy(obs_rot6d).float())
    r_act = transforms.rotation_6d_to_matrix(torch.from_numpy(act_rot6d).float())
    # r_new = torch.matmul(r_obs, r_act) # old
    r_new = torch.matmul(r_act, r_obs) # new
    rot6d_new = transforms.matrix_to_rotation_6d(r_new).numpy()

    pos_new = obs_pos + act_pos
    tail_new = obs_tail + act_tail
    print(act_tail, obs_tail)
    return np.concatenate([rot6d_new, pos_new, tail_new], axis=0)


def stack_last_n_obs(all_obs, n_steps):
    assert(len(all_obs) > 0)
    all_obs = list(all_obs)
    if isinstance(all_obs[0], np.ndarray):
        result = np.zeros((n_steps,) + all_obs[-1].shape, 
            dtype=all_obs[-1].dtype)
        start_idx = -min(n_steps, len(all_obs))
        result[start_idx:] = np.array(all_obs[start_idx:])
        if n_steps > len(all_obs):
            # pad
            result[:start_idx] = result[start_idx]
    elif isinstance(all_obs[0], torch.Tensor):
        result = torch.zeros((n_steps,) + all_obs[-1].shape, 
            dtype=all_obs[-1].dtype)
        start_idx = -min(n_steps, len(all_obs))
        result[start_idx:] = torch.stack(all_obs[start_idx:])
        if n_steps > len(all_obs):
            # pad
            result[:start_idx] = result[start_idx]
    else:
        raise RuntimeError(f'Unsupported obs type {type(all_obs[0])}')
    return result



def get_all_keys(d, prefix=""):
    keys = []
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        keys.append(full_key)
        if isinstance(v, dict):
            keys.extend(get_all_keys(v, full_key))
    return keys


def load_payload(model, payload, prefer_ema=True, exclude_keys=None, strict=False):                                                                                                                                                                                             
      """Load a native 3D-Diffusion-Policy workspace checkpoint into a fresh DP3 model.                                                                                                                                                                                           
                                                                                                                                                                                                                                                                                  
      Checkpoint layout (from diffusion_policy_3d/train.py:save_checkpoint):                                                                                                                                                                                                      
          payload['state_dicts'] = {                                                                                                                                                                                                                                              
              'model':     <flat state_dict of the DP3 model>,
              'ema_model': <flat state_dict of the EMA copy>,    # preferred for inference                                                                                                                                                                                        
              'optimizer': <optimizer state_dict>,                # ignored here
          }                                                                                                                                                                                                                                                                       
          payload['pickles'] = {'_output_dir': ..., 'global_step': ..., 'epoch': ...}
                                                                                                                                                                                                                                                                                  
      The DP3.load_state_dict call recurses into obs_encoder / model / normalizer
      / noise_scheduler etc. — DictOfTensorMixin's custom _load_from_state_dict                                                                                                                                                                                                   
      hook rebuilds the normalizer's nested ParameterDict from the prefixed keys.                                                                                                                                                                                                 
      """                                                                                                                                                                                                                                                                         
      sd_key = "ema_model"                                                                                                                                                                                 
      state_dict = payload["state_dicts"][sd_key]                                                                                                                                                                                                                                 
                  
      if exclude_keys:                                                                                                                                                                                                                                                            
          state_dict = {k: v for k, v in state_dict.items() if k not in tuple(exclude_keys)}
                                                                                                                                                                                                                                                                                  
      missing, unexpected = model.load_state_dict(state_dict, strict=strict)                                                                                                                                                                                                      
      print(
          f"[load_payload] loaded '{sd_key}'  "                                                                                                                                                                                                                                   
          f"missing={len(missing)}  unexpected={len(unexpected)}"
      )                                                                                                                                                                                                                                                                           
      if missing:
          print(f"  missing[:5] = {missing[:5]}")                                                                                                                                                                                                                                 
      if unexpected:                                                                                                                                                                                                                                                              
          print(f"  unexpected[:5] = {unexpected[:5]}")
                                                                                                                                                                                                                                                                                  
      # Restore any dilled workspace fields whose names happen to also exist as                                                                                                                                                                                                   
      # model attributes. For real_syringe-* ckpts the only pickles are
      # _output_dir / global_step / epoch — none of which are model attrs — so                                                                                                                                                                                                    
      # this is typically a no-op. Kept for completeness.
      for pkey, pbytes in payload.get("pickles", {}).items():                                                                                                                                                                                                                     
          if hasattr(model, pkey):
              setattr(model, pkey, dill.loads(pbytes))                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                  
      return missing, unexpected


def _resolve_episode_file(dataset_root: Path, episode_index: int) -> Path:
    candidates = sorted(
        dataset_root.glob(f"data/chunk-*/episode_{episode_index:06d}.parquet")
    )
    if not candidates:
        raise FileNotFoundError(
            f"Could not find episode_{episode_index:06d}.parquet under "
            f"{dataset_root}/data/chunk-*"
        )
    return candidates[0]


def load_initial_state_and_actions(dataset_root: Path, episode_index: int):
    """Return (initial_state [23], actions [T, 23]) as float32 numpy arrays."""
    episode_file = _resolve_episode_file(dataset_root, episode_index)
    table = pq.read_table(episode_file, columns=["observation.state", "action.right_eef_pose", "action"])  # "action"
    if table.num_rows == 0:
        raise RuntimeError(f"Episode file is empty: {episode_file}")

    initial_state = np.asarray(
        table.column("observation.state")[0].as_py(), dtype=np.float32
    )
    actions = np.stack(
        [np.asarray(a.as_py(), dtype=np.float32) for a in table.column("action.right_eef_pose")],  # "action"
        axis=0,
    )[1:] # skip the first action since it's a no-op (dataset convention)
    joint_actions = np.stack(
        [np.asarray(a.as_py(), dtype=np.float32) for a in table.column("action")],  # "action"
        axis=0,
    )
    print(f"[replay] episode_file: {episode_file}")
    print(f"[replay] initial_state shape: {initial_state.shape}")
    print(f"[replay] actions shape:       {actions.shape}")
    print(f"[replay] joint_actions shape: {joint_actions.shape}")
    return initial_state, actions, joint_actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--episode-index", type=int, default=1)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--use-goal-condition", action="store_true", help="Whether to condition the policy on the goal state (if enabled in the policy config)")
    parser.add_argument("--use-relative", action="store_true", help="Whether to use relative EEF pose as input feature and action space (if supported by the policy config)")
    parser.add_argument("--use-delta", action="store_true", help="Whether to use relative EEF pose as input feature and action space (if supported by the policy config)")
    parser.add_argument("--use-pretrained", action="store_true", help="Whether to use the pre-trained policy checkpoint (if enabled in the policy config and available)")
    args = parser.parse_args()

    # Load DP3 config via Hydra from this package's `configs/dp3.yaml` so
    # `cfg` is available for policy instantiation below.
    configs_dir = pathlib.Path(__file__).resolve().parents[1] / "configs"
    with hydra.initialize_config_dir(config_dir=str(configs_dir)):
        cfg = hydra.compose(config_name="dp3.yaml")
    print(f"[replay] Loaded Hydra config from {configs_dir}/dp3.yaml")

    # Try to import DP3 from a local 3D-Diffusion-Policy checkout (if present)
    dp3_repo = pathlib.Path("~/Desktop/3D-Diffusion-Policy/3D-Diffusion-Policy").expanduser()
    if dp3_repo.exists():
        sys.path.insert(0, str(dp3_repo))
    try:
        from diffusion_policy_3d.policy.dp3 import DP3  # type: ignore
        from diffusion_policy_3d.model.vision.articubot import PointNet2_super
        from train_high_level import compute_weighted_displacement
        from diffusion_policy_3d.common.pytorch_util import dict_apply
    except Exception as e:
        DP3 = None  # type: ignore
        print(f"[replay] Could not import DP3 from 3D-Diffusion-Policy: {e}")

    if args.use_goal_condition:
        cfg.policy.goal_mode = "high_level"
        if args.use_delta:
            policy_path = pathlib.Path(GOAL_DELTA_POLICY_ROOT).expanduser()
        elif args.use_relative:
            policy_path = pathlib.Path(GOAL_REL_POLICY_ROOT).expanduser()
        else:
            if args.use_pretrained:
                policy_path = pathlib.Path(GOAL_ABS_PRETRAIN_POLICY).expanduser()
                cfg.task.shape_meta.obs.agent_pos.shape=[91]
                cfg.task.shape_meta.action.shape=[28]
                cfg.policy.use_manual_adaptor=True
            else:
                policy_path = pathlib.Path(GOAL_ABS_POLICY_ROOT).expanduser()
    else:
        cfg.policy.goal_mode = "None"
        if args.use_delta:
            policy_path = pathlib.Path(NONE_DELTA_POLICY_ROOT).expanduser()
        elif args.use_relative:
            policy_path = pathlib.Path(NONE_REL_POLICY_ROOT).expanduser()
        else:
            policy_path = pathlib.Path(NONE_ABS_POLICY_ROOT).expanduser()
    
    model: DP3 = hydra.utils.instantiate(cfg.policy)
    if DP3 is not None and policy_path.exists():
        payload = torch.load(policy_path.open('rb'), pickle_module=dill, map_location='cpu')
        print(f"[replay] Loaded policy payload keys: {list(payload.keys())}")

        # Instantiate DP3 model if possible. Try several fallbacks.
        load_payload(model, payload) # , exclude_keys=['_dummy_variable'])
        print("[replay] Policy weights loaded into DP3 model")
    model.to('cuda')
    model.eval()

    # load high level model
    if args.use_goal_condition:
        num_classes = 16 * 3 + 1
        high_level_ckpt = '/home/leap/Desktop/lerobot_restore/lerobot/outputs/train/syringe_high_level_0515/dp3_epoch_19.pt'
        # high_level_ckpt = '/home/leap/Desktop/lerobot_restore/lerobot/outputs/train/syringe_high_level_0423/23-04-45/dp3_epoch_16.pt' # syringe
        # high_level_ckpt = '/home/leap/Desktop/lerobot_restore/lerobot/outputs/train/real_spraybottle_high_level/dp3_epoch_40.pt' # spraybottle
        high_level_model = PointNet2_super(num_classes=num_classes, input_channel=6)
        high_level_model.load_state_dict(torch.load(high_level_ckpt))
        high_level_model.to('cuda')
        high_level_model.eval()

    initial_state, actions, joint_actions = load_initial_state_and_actions(
        args.dataset_root, args.episode_index
    )

    # Build robot with the Azure Kinect cameras used by this replay setup.
    config = FrankaLeapRobotConfig(
        cameras={
            "cam_azure_kinect_front": AzureKinectCameraConfig(
                device_id=0,
                fps=30,
                width=1280,
                height=720,
                use_transformed_depth=True,
                wired_sync_mode="master",
            ),
            "cam_azure_kinect_side": AzureKinectCameraConfig(
                device_id=1,
                fps=30,
                width=1280,
                height=720,
                use_transformed_depth=True,
                wired_sync_mode="subordinate",
                subordinate_delay_off_master_usec=200,
            ),
        }
    )
    robot = make_robot_from_config(config)
    robot_adapter = FrankaLeapAdapter("right_eef")

    # connect() will run interactive calibration (Franka matches GELLO start pose).
    robot.connect()

    import rerun as rr
    rr.init("franka_leap_replay", spawn=True)

    ALPHA = 0.0   # 0.0 = no smoothing, 1.0 = max smoothing, 0.6
    _smoothed_joint = None
    _smoothed_eef   = None
    RUN_ACT_STEPS = 4
    GRAVITY_BIAS_Z = +0.01   # tune empirically  

    try:
        # Move arm + hand to the dataset's initial state before replay begins.
        arm_init = initial_state[:7].astype(np.float64)
        hand_init = initial_state[7:].astype(np.float64)
        print(f"[replay] Moving Franka to initial arm state: {np.round(arm_init, 4)}")
        robot._smooth_move_to(arm_init)
        print(f"[replay] Moving LEAP hand to initial state:  {np.round(hand_init, 4)}")
        robot._move_hand_to(hand_init)

        # input("\n[replay] Press Enter to start replaying the trajectory...")
        print("\n[replay] Starting rollout...")
        print("Press:")
        print("  g -> enable gravity compensation")
        print("  d -> disable gravity compensation")
        print("  q -> quit")

        dt_target = 1.0 / args.fps
        obs_queue = deque([], maxlen=3)
        for idx in range(0, 50 * len(actions), RUN_ACT_STEPS):
            print(idx, "/", 50 * len(actions))
            if idx >= len(actions) - RUN_ACT_STEPS:
                action_chunk = actions[-RUN_ACT_STEPS:]
            else:
                action_chunk = actions[idx:idx+RUN_ACT_STEPS]
            t0 = time.perf_counter()
            observation = robot.capture_observation()
            observation["observation.right_eef_pose"] = add_eef_pose(robot, observation['observation.state'])
            prev_target = observation["observation.right_eef_pose"].cpu().numpy().copy()  # for computing relative action if needed
            observation = robot.get_pointcloud_obs(observation)
            cur_state = observation['observation.state']

            pcd = observation.get('observation.points.point_cloud')
            pcd_scene = pcd[:4000]
            pcd_scene_onehot = torch.zeros_like(pcd_scene)
            pcd_scene_onehot[:, 0] = 1.0
            pcd_scene = torch.cat([pcd_scene, pcd_scene_onehot], dim=-1)
            pcd_hand = pcd[4000:]
            pcd_hand_onehot = torch.zeros_like(pcd_hand)
            pcd_hand_onehot[:, 1] = 1.0
            pcd_hand = torch.cat([pcd_hand, pcd_hand_onehot], dim=-1)
            pcd_with_onehot = torch.cat([pcd_scene, pcd_hand], dim=0)

            if args.use_goal_condition and idx % 200 == 0:
                imagined_pcd = observation['observation.points.gripper_pcds']
                pcd = torch.cat([pcd, torch.zeros_like(pcd)], axis=-1)
                pcd[..., 3] = 1.0  # Add a feature dimension
                imagined_pcd = torch.cat([imagined_pcd, torch.zeros_like(imagined_pcd)], axis=-1)
                imagined_pcd[..., 4] = 1.0  # Add a feature dimension
                high_level_obs = torch.cat([pcd, imagined_pcd], axis=0)[None].permute(0,2,1).to('cuda').float()
                with torch.no_grad():
                    pred = high_level_model(high_level_obs)
                pred_points = compute_weighted_displacement(high_level_obs, pred, "16points", is_gmm=True)

            this_obs = {
                'agent_pos': observation["observation.right_eef_pose"],
                'point_cloud': pcd_with_onehot,
                'imagin_robot': observation["observation.points.gripper_pcds"],
                'goal_gripper_pcd': pred_points.squeeze() if args.use_goal_condition else observation["observation.points.gripper_pcds"],
            }
            obs_queue.append(this_obs)

            # Log point cloud (scene + imagined hand)
            pc = pcd_with_onehot
            if pc is not None:
                pc_np = pc.cpu().numpy() if hasattr(pc, 'cpu') else np.asarray(pc)
                rr.log('point_cloud', rr.Points3D(pc_np[:, :3], colors=[100, 100, 100], radii=0.002))
                rr.log('hand_pcd', rr.Points3D(observation['observation.points.gripper_pcds'].cpu().numpy(), colors=[0, 0, 255], radii=0.002))
                rr.log('goal_pcd', rr.Points3D(this_obs['goal_gripper_pcd'].cpu().numpy(), colors=[255, 0, 0], radii=0.002))

            # # # Log observed EEF frame, if available
            obs_eef = observation.get('observation.right_eef_pose')
            if obs_eef is not None:
                obs_np = obs_eef.cpu().numpy() if hasattr(obs_eef, 'cpu') else np.asarray(obs_eef)

                rot6d = torch.from_numpy(obs_np[:6].astype(np.float32))
                rot_mat = transforms.rotation_6d_to_matrix(rot6d).numpy()

                trans = obs_np[6:9]

                rr.log("world/deoxys_eef", rr.Transform3D(
                    translation=trans,
                    mat3x3=rot_mat,
                ))

                # draw axes in LOCAL frame (this is the key)
                rr.log(
                    "world/deoxys_eef/axes",
                    rr.Arrows3D(
                        origins=[[0,0,0]] * 3,
                        vectors=np.eye(3) * 0.1,   # axis length
                        colors=[[255,0,0],[0,255,0],[0,0,255]],
                    )
                )

            # # Log EEF frame from URDF, if available
            # T = observation.get('observation.urdf_eef_pose')
            # if T is not None:
            #     T = T.cpu().numpy() if hasattr(T, 'cpu') else np.asarray(T)

            #     trans = T[:3, 3]
            #     rot_mat = T[:3, :3]

            #     rr.log("world/eef", rr.Transform3D(
            #         translation=trans,
            #         mat3x3=rot_mat,
            #     ))

            #     # draw axes in LOCAL frame (this is the key)
            #     rr.log(
            #         "world/eef/axes",
            #         rr.Arrows3D(
            #             origins=[[0,0,0]] * 3,
            #             vectors=np.eye(3) * 0.1,   # axis length
            #             colors=[[255,0,0],[0,255,0],[0,0,255]],
            #         )
            #     )
                    

            # # Log action EEF frame (from dataset)
            # try:
            #     act_np = np.asarray(eef_action)

            #     rot6d = torch.from_numpy(act_np[:6].astype(np.float32))
            #     rot_mat = transforms.rotation_6d_to_matrix(rot6d).numpy()

            #     trans = act_np[6:9]

            #     rr.log("world/action_eef", rr.Transform3D(
            #         translation=trans,
            #         mat3x3=rot_mat,
            #     ))

            #     # draw axes in LOCAL frame (this is the key)
            #     rr.log(
            #         "world/action_eef/axes",
            #         rr.Arrows3D(
            #             origins=[[0,0,0]] * 3,
            #             vectors=np.eye(3) * 0.1,   # axis length
            #             colors=[[255,0,0],[0,255,0],[0,0,255]],
            #         )
            #     )
            # except Exception:
            #     pass

            observation = dict()
            for key in this_obs.keys():
                observation[key] = stack_last_n_obs(
                    [obs[key] for obs in obs_queue],
                    2
                ).unsqueeze(0)  # add batch dim


            action_dict = model.predict_action(observation)
            np_action_dict = dict_apply(action_dict,
                                                lambda x: x.detach().squeeze(0))
            policy_actions = np_action_dict['action']

            for a_t in range(RUN_ACT_STEPS):
                eef_action = action_chunk[a_t]

                # -------------------------------------------
                # scale the action for relative
                if args.use_relative:
                    WARMUP = 1                                                                                                                                                                                                                                                                     
                    GAIN = 1.0 + (1.4 - 1.0) * min(1.0, idx / WARMUP)
                    rel_action = policy_actions[a_t]
                    rel_action[6:9] = rel_action[6:9] * GAIN
                    if a_t in [1,2,3]:
                        rel_action[9:25] = 0
                    # rel_action[9:25] = rel_action[9:25] * GAIN
                    R_rel = transforms.rotation_6d_to_matrix(rel_action[None, :6]).squeeze(0)
                    axis_angle = transforms.matrix_to_axis_angle(R_rel) * GAIN
                    R_rel_scaled = transforms.axis_angle_to_matrix(axis_angle)
                    rel_action[:6] = transforms.matrix_to_rotation_6d(R_rel_scaled[None]).squeeze(0)
                    # ---------------------------
                    # just for debugging
                    # rel_action[:6] = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32)  # zero rotation
                    # rel_action[6:] = 0
                    # ---------------------------
                    policy_action = rel_to_abs(rel_action.cpu().numpy(), prev_target)                                                                                                                                                                                                                                  
                    policy_action[6:9][2] += GRAVITY_BIAS_Z
                elif args.use_delta:
                    policy_action = policy_actions[a_t].cpu().numpy()
                    WARMUP = 15                                                                                                                                                                                                                                                                     
                    GAIN = 1.0 + (1.4 - 1.0) * min(1.0, idx / WARMUP) 
                    policy_action = torch.from_numpy(policy_action)  
                    policy_action[6:9] = obs_eef[6:9] + GAIN * policy_action[6:9]
                    policy_action[9:25] = obs_eef[9:25] + GAIN * policy_action[9:25]

                    R_obs = transforms.rotation_6d_to_matrix(obs_eef[None, :6]).squeeze(0)
                    R_rel = transforms.rotation_6d_to_matrix(policy_action[None, :6]).squeeze(0)
                    axis_angle = transforms.matrix_to_axis_angle(R_rel) * GAIN
                    R_rel_scaled = transforms.axis_angle_to_matrix(axis_angle)
                    R_act_scaled = R_obs @ R_rel_scaled
                    policy_action[:6] = transforms.matrix_to_rotation_6d(R_act_scaled[None]).squeeze(0)
                    policy_action = policy_action.numpy()
                    policy_action[6:9][2] += GRAVITY_BIAS_Z
                else:
                    policy_action = policy_actions[a_t] # .cpu().numpy()

                    # convert policy action from axis angle to rot6d
                    R_act = transforms.axis_angle_to_matrix(policy_action[None, 3:6])
                    rot6d = transforms.matrix_to_rotation_6d(R_act).squeeze(0)
                    policy_action = torch.cat([rot6d, policy_action[:3], policy_action[6:]]).cpu().numpy()

                    WARMUP = 50 # 15                                                                                                                                                                                                                                                                     
                    GAIN = 1.0 + (1.42 - 1.0) * min(1.0, idx / WARMUP)      # 1.42       
                    if flags.gravity_comp:
                        GAIN = 1.5                                                                                                                                                                                                                                                                                                                                                                                                                                              
                    policy_action = torch.from_numpy(policy_action)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
                    # Position: linear interpolation between current_eef and target.                                                                                                                                                                                                            
                    policy_action[6:9] = obs_eef[6:9] + torch.clip(GAIN * (policy_action[6:9] - obs_eef[6:9]), -0.01, 0.01)                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                    
                    # Hand: linear interpolation in joint space.                                                                                                                                                                                                                                    
                    policy_action[9:25] = obs_eef[9:25] + GAIN * (policy_action[9:25] - obs_eef[9:25])                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                                    
                    # Rotation: scale the *relative rotation* between obs and target, then re-compose.                                                                                                                                                                                                                                                                                                                                                                                                                                            
                    R_obs = transforms.rotation_6d_to_matrix(obs_eef[None, :6]).squeeze(0)                                                                                                                                                                                                               
                    R_act = transforms.rotation_6d_to_matrix(policy_action[None, :6]).squeeze(0)                                                                                                                                                                                                                   
                    R_rel = R_obs.transpose(-2, -1) @ R_act                                                                                                                                                                                                                                         
                    axis_angle = transforms.matrix_to_axis_angle(R_rel) * GAIN # * 1.5                                                                                                                                                                                                                               
                    R_rel_scaled = transforms.axis_angle_to_matrix(axis_angle)                                                                                                                                                                                                                               
                    R_act_scaled = R_obs @ R_rel_scaled
                    policy_action[:6] = transforms.matrix_to_rotation_6d(R_act_scaled[None]).squeeze(0)
                    policy_action = policy_action.numpy()
                    # policy_action[6:9][2] += GRAVITY_BIAS_Z
                    if flags.gravity_comp:
                        policy_action[6:9][2] += GRAVITY_BIAS_Z
                # -------------------------------------------


                # visualize the difference between policy action and dataset action in RPY + XYZ + tail
                act_np = np.asarray(eef_action)
                rot6d = torch.from_numpy(act_np[:6].astype(np.float32))
                rot_mat = transforms.rotation_6d_to_matrix(rot6d).numpy()
                trans = act_np[6:9]
                rr.log("world/gt_eef", rr.Transform3D(
                    translation=trans,
                    mat3x3=rot_mat,
                ))
                rr.log(
                    "world/gt_eef/axes",
                    rr.Arrows3D(
                        origins=[[0,0,0]] * 3,
                        vectors=np.eye(3) * 0.1,   # axis length
                        colors=[[255,0,0],[0,255,0],[0,0,255]],
                    )
                )

                act_np = np.asarray(policy_action)
                rot6d = torch.from_numpy(act_np[:6].astype(np.float32))
                rot_mat = transforms.rotation_6d_to_matrix(rot6d).numpy()
                trans = act_np[6:9]
                rr.log("world/policy_eef", rr.Transform3D(
                    translation=trans,
                    mat3x3=rot_mat,
                ))
                rr.log(
                    "world/policy_eef/axes",
                    rr.Arrows3D(
                        origins=[[0,0,0]] * 3,
                        vectors=np.eye(3) * 0.1,   # axis length
                        colors=[[255,0,0],[0,255,0],[0,0,255]],
                    )
                )

                # print('eef', eef_action[9:])
                # print('policy', policy_action[9:])
                # print("Action diff:   ", policy_action[9:25] - eef_action[9:25])
                # print('-----------')
                prev_target = policy_action.copy()
                joint_action = robot_adapter.transform_action(torch.from_numpy(policy_action), cur_state).squeeze().numpy()

                # EMA smoothing for control
                if _smoothed_joint is None:                                                                                                                                                                                                                                
                    _smoothed_joint = joint_action.copy()
                    _smoothed_eef   = policy_action.copy()                                                                                                                                                                                                                                   
                                                                                                                                                                                                                                                                                                
                _smoothed_joint = ALPHA * _smoothed_joint + (1 - ALPHA) * joint_action                                                                                                                                                                                                      
                _smoothed_eef   = ALPHA * _smoothed_eef   + (1 - ALPHA) * policy_action

                robot.send_action(joint_action)
                busy_wait(dt_target - (time.perf_counter() - t0))
                if (idx * RUN_ACT_STEPS + a_t) % args.fps == 0:
                    print(f"[replay] step {idx * RUN_ACT_STEPS + a_t}/{len(actions)}")

        print("[replay] Done.")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
