# Documentation

Mixed bag of notes, tidbits and info with using LeRobot in our Kinect + Aloha stack:

## Setup

```
git@github.com:r-pad/lerobot.git
pixi shell
pixi run install-pytorch3d
pixi run install-k4a
```

- `pyk4a` needs to be installed from source to build it with numpy 2. Run `pixi run install-k4a`. This will take a long time to build. However, if the environment is deactivated/reactivated, it doesn't work anymore :(. In that case, run `pip uninstall pyk4a && pip install pyk4a` to get it working with the new env (should probably fix or automate this).

## Commands

### Teleop

Base Teleop command:
```
python lerobot/scripts/control_robot.py --robot.type=aloha --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}}' --control.type=teleoperate --control.display_data=true
```

The camera can be configured through the attached config, with options in `lerobot/common/robot_devices/cameras/azure_kinect.py`. 

Other options of note is the setting `--robot.max_relative_target` which is a limit on how much the robot can move in a single step. The default value is 5. With lower values <3, the motion is very laggy and joints move weirdly. The clamping can be disabled with `--robot.max_relative_target=null` allowing for very smooth high-frequency teleop, but I do not recommend as one wrong move can send the Aloha to DynaMixel heaven. The default value works well enough.

Lastly we have `--control.display_data=true` which opens a streaming Rerun window.

### Record episodes

Base record command:

```
python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.single_task="Grasp mug and place it on the table." --control.repo_id=sriramsk/aloha_mug_eef_depth --control.num_episodes=100 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}}' --robot.use_eef=true --control.push_to_hub=true --control.fps=60 --control.reset_time_s=5 --control.warmup_time_s=3 --control.resume=true --control.num_image_writer_processes=4
```

`--control.repo_id` indicates the name with which the dataset is saved and uploaded to Huggingface (if `--control.push_to_hub` is enabled). `--control.resume` allows writing to an existing dataset. While recording, use left/right arrow keys to finish / reset current episode. `--robot.use_eef=true` runs forward kinematics and stores the computed eef pose in the dataset.

### Visualize and replay

```
python lerobot/scripts/visualize_dataset.py  --repo-id sriramsk/aloha_mug_eef   --episode-index 0
```

(Careful when replaying)
```
python lerobot/scripts/control_robot.py --robot.type=aloha --robot.cameras='{"cam_kinect": {"type": "opencv", "camera_index": 0, "fps": 30, "width": 1280, "height": 720}}' --control.type=replay --control.fps=60 --control.repo_id=sriramsk/aloha_mug_eef --control.episode=0
```

### Modify existing dataset

LeRobot doesn't provide a simple way to add new keys / modify existing keys to a dataset. Instead this script takes the blunt approach of simply creating a new dataset, copying required keys and modifying/adding other keys.

```
python upgrade_dataset.py --source_repo_id sriramsk/human_mug_0718 --target_repo_id sriramsk/phantom_mug_0718_heatmapGoal --discard_episodes 2 10 11 13 21 --phantomize --phantom_extradata /data/sriram/lerobot_extradata/sriramsk/human_mug_0718 --push_to_hub --new_features goal_gripper_proj
```

`--source_repo_id` is the id of the existing dataset and `--target_repo_id` is the id of the new dataset being created. `--discard_episodes` skips problematic episodes which may exist in the source data, `--new_features` takes in a list of new features to be added (in this case, a heatmap image).

`--phantomize` and `--phantom_extradata` are extra arguments only required when retargeting a human demonstration dataset following the approach in[Phantom](https://phantom-human-videos.github.io/).

### Train

Too many options to describe in detail, some notes:
- `horizon` and `n_action_steps` are much higher than the original diffusion policy because our data is captured at a high frequency.
- Center crop `[600, 600]` to avoid extra parts of the workspace sneaking in.
- `--policy.use_separate_rgb_encoder_per_camera` because we also have heatmap images and it doesn't make much sense to encode both RGB and heatmaps with the same encoder.
- `enable_goal_conditioning=true` is mostly for inference since this flag can't (?) be set at inference.
- `--policy.use_text_embedding=true` to use siglip features and condition on those as well.
- Bump up `num_workers` if gpu util is low, video decoding benefits well from parallelization.

```
python lerobot/scripts/train.py --dataset.repo_id=sriramsk/aloha_mug_eef_depth_0709_heatmapGoal --policy.type=diffusion --output_dir=outputs/train/diffPo_aloha_eef_rgb_0709_heatmapGoal --job_name=diffPo_aloha_eef_rgb_0709_heatmapGoal --policy.device=cuda --wandb.enable=true --policy.n_obs_steps=4 --policy.horizon=128 --policy.n_action_steps 64 --policy.drop_n_last_frames=61 --policy.crop_shape="[600, 600]" --policy.crop_is_random=false --policy.use_separate_rgb_encoder_per_camera=true --policy.enable_goal_conditioning=true --dataset.image_transforms.enable=true
```

Training with a subsampled dataset:
```
python lerobot/scripts/train.py --dataset.repo_id=sriramsk/aloha_mug_eef_depth_0709_heatmapGoal_subsampled --policy.type=diffusion --output_dir=outputs/train/diffPo_aloha_eef_rgb_0709_heatmapGoal_subsampled --job_name=diffPo_aloha_eef_rgb_0709_heatmapGoal_subsampled --policy.device=cuda --wandb.enable=true --policy.crop_shape="[600, 600]" --policy.crop_is_random=false --policy.use_separate_rgb_encoder_per_camera=true --policy.enable_goal_conditioning=true --dataset.image_transforms.enable=true
```

Pretraining with DROID:
```
python lerobot/scripts/train.py --dataset.repo_id=sriramsk/droid_lerobot --policy.type=diffusion --output_dir=outputs/train/diffPo_droid_lerobot --job_name=diffPo_droid_lerobot --policy.device=cuda --wandb.enable=true --policy.use_separate_rgb_encoder_per_camera=true --policy.enable_goal_conditioning=true --steps=1_000_000 --batch_size=64 --num_workers=48 --prefetch_factor=8 --policy.use_text_embedding=true 
```


### Rollout

```
python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=30 --control.single_task="Grasp mug and place it on the table." --control.repo_id=sriramsk/eval_aloha_eef_rgb_0709_heatmapGoal --control.num_episodes=1 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffPo_aloha_eef_rgb_0709_heatmapGoal/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=120
```


## Misc mapping stuff to align LeRobot Aloha and `robot_descriptions` Aloha

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

## LIBERO

Generating lerobot training data from hdf5 files
Training on one hdf5, testing on one env.
