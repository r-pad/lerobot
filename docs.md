# Notes

- `pyk4a` needs to be installed from source to build it with numpy 2. Run `pixi run install-k4a`. This will take a long time to build. However, if the environment is deactivated/reactivated, it doesn't work anymore. In that case, run `pip uninstall pyk4a && pip install pyk4a` to get it working with the new env (should probably fix or automate this).

# Commands

### Teleop
`python lerobot/scripts/control_robot.py --robot.type=aloha --robot.cameras='{"cam_kinect": {"type": "opencv", "camera_index": 0, "fps": 30, "width": 1280, "height": 720}}' --control.type=teleoperate --control.display_data=true`

`python lerobot/scripts/control_robot.py --robot.type=aloha --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}}' --control.type=teleoperate --control.display_data=true`

### Record episodes
`python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.single_task="Grasp mug and place it on the table." --control.repo_id=sriramsk/aloha_mug_eef --control.num_episodes=100 --robot.cameras='{"cam_kinect": {"type": "opencv", "camera_index": 0, "fps": 30, "width": 1280, "height": 720}}' --robot.use_eef=true --control.push_to_hub=true --control.fps=60 --control.reset_time_s=5 --control.warmup_time_s=3 --control.resume=true`

`python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.single_task="Grasp mug and place it on the table." --control.repo_id=sriramsk/aloha_mug_eef_depth --control.num_episodes=100 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}}' --robot.use_eef=true --control.push_to_hub=true --control.fps=60 --control.reset_time_s=5 --control.warmup_time_s=3 --control.resume=true --control.num_image_writer_processes=4`

### Visualize 
`python lerobot/scripts/visualize_dataset.py  --repo-id sriramsk/aloha_mug_eef   --episode-index 0`

### Replay
`python lerobot/scripts/control_robot.py --robot.type=aloha --robot.cameras='{"cam_kinect": {"type": "opencv", "camera_index": 0, "fps": 30, "width": 1280, "height": 720}}' --control.type=replay --control.fps=60 --control.repo_id=sriramsk/aloha_mug_eef --control.episode=0`

### Generate goals

`python lerobot/scripts/migrate_dataset.py --source_repo_id sriramsk/aloha_mug_eef_depth_0709 --target_repo_id aloha_mug_eef_depth_0709_with_goal`

### Train
`python lerobot/scripts/train.py --dataset.repo_id=sriramsk/aloha_mug_eef --policy.type=diffusion --output_dir=outputs/train/diffPo_aloha --job_name=diffPo_aloha --policy.device=cuda --wandb.enable=true --policy.n_obs_steps=4 --policy.horizon=128 --policy.n_action_steps 64 --policy.drop_n_last_frames=61 --policy.crop_shape="[540, 960]"`

`python lerobot/scripts/train.py --dataset.repo_id=sriramsk/aloha_mug_eef_depth --policy.type=diffusion --output_dir=outputs/train/diffPo_aloha_eef_rgb --job_name=diffPo_aloha_eef_rgb --policy.device=cuda --wandb.enable=true --policy.n_obs_steps=4 --policy.horizon=128 --policy.n_action_steps 64 --policy.drop_n_last_frames=61 --policy.crop_shape="[600, 600]" --policy.crop_is_random=false`

`python lerobot/scripts/train.py --dataset.repo_id=sriramsk/aloha_mug_eef_depth_0709_with_goal --policy.type=diffusion --output_dir=outputs/train/diffPo_aloha_eef_rgb_0709_gc --job_name=diffPo_aloha_eef_rgb_0709_gc --policy.device=cuda --wandb.enable=true --policy.n_obs_steps=4 --policy.horizon=128 --policy.n_action_steps 64 --policy.drop_n_last_frames=61 --policy.crop_shape="[600, 600]" --policy.crop_is_random=false --policy.use_separate_rgb_encoder_per_camera=true --policy.enable_goal_conditioning=true`

# Rollout
`python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=30 --control.single_task="Grasp mug and place it on the table." --control.repo_id=$USER/eval_aloha_mug --control.num_episodes=10 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_kinect": {"type": "opencv", "camera_index": 0, "fps": 30, "width": 1280, "height": 720}}' --control.push_to_hub=false --control.policy.path=outputs/train/diffPo_aloha_mug/checkpoints/last/pretrained_model/`

`python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=30 --control.single_task="Grasp mug and place it on the table." --control.repo_id=sriramsk/eval_aloha_eef_rgb --control.num_episodes=1 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffPo_aloha_eef_rgb/checkpoints/last/pretrained_model/`


# Misc mapping stuff to align LeRobot Aloha and `robot_descriptions` Aloha

- In configuration.q of the mink configuration object
```py
[
    'left/waist',
    'left/shoulder', 
    'left/elbow', 
    'left/forearm_roll', 
    'left/wrist_angle', 
    'left/wrist_rotate', 
    'left/left_finger', 
    'left/right_finger', 
    'right/waist', 
    'right/shoulder', 
    'right/elbow', 
    'right/forearm_roll', 
    'right/wrist_angle', 
    'right/wrist_rotate', 
    'right/left_finger', 
    'right/right_finger'
]
```

- In `observation['state']`: 
```py
            "left": DynamixelMotorsBusConfig(
                port="/dev/ttyDXL_follower_left",
                motors={
                    # name: (index, model)
                    "waist": [1, "xm540-w270"],
                    "shoulder": [2, "xm540-w270"],
                    "shoulder_shadow": [3, "xm540-w270"],
                    "elbow": [4, "xm540-w270"],
                    "elbow_shadow": [5, "xm540-w270"],
                    "forearm_roll": [6, "xm540-w270"],
                    "wrist_angle": [7, "xm540-w270"],
                    "wrist_rotate": [8, "xm430-w350"],
                    "gripper": [9, "xm430-w350"],
                },
            ),
            "right": DynamixelMotorsBusConfig(
                port="/dev/ttyDXL_follower_right",
                motors={
                    # name: (index, model)
                    "waist": [1, "xm540-w270"],
                    "shoulder": [2, "xm540-w270"],
                    "shoulder_shadow": [3, "xm540-w270"],
                    "elbow": [4, "xm540-w270"],
                    "elbow_shadow": [5, "xm540-w270"],
                    "forearm_roll": [6, "xm540-w270"],
                    "wrist_angle": [7, "xm540-w270"],
                    "wrist_rotate": [8, "xm430-w350"],
                    "gripper": [9, "xm430-w350"],
                },
            ),
```

- Mapping indexes from real to sim
```py
0 - 0
1 - 1
2 - 3
3 - 5
4 - 6
5 - 7

6 - 8
7 - 8

8 - 9
9 - 10
10 - 12
11 - 14
12 - 15
13 - 16

14 - 17
15 - 17
```

- Mapping values from real to sim for IK (*sign + offset). Shoulder joint needed to be handled separately (set idx 1 and 9 sign to 1 and offset to 0.)
```py
signs = [-1, -1, 1, 1, 1, 1, ...]
offsets = [pi/2, pi/2, -pi/2, 0, 0, 0 ...]
```

### Set up sim viewer to compare joint values for real to sim

```py
import mujoco.viewer
import mujoco
from robot_descriptions.loaders.mujoco import load_robot_description

model = load_robot_description("aloha_mj_description")
data = mujoco.MjData(model)
viewer = mujoco.viewer.launch(model, data)
```