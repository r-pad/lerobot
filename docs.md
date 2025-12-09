# Documentation

Mixed bag of notes, tidbits and info with using LeRobot in our Kinect + Aloha stack:

## Setup

```
git@github.com:r-pad/lerobot.git
pixi shell
pixi run install-pytorch3d

# only for robot control not for training
pixi run install-pynput
pixi run install-open3d 
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

# With azure kinect
python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.single_task="Grasp mug and place it on the table." --control.repo_id=sriramsk/testing_realsense --control.num_episodes=100 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --robot.use_eef=true --control.push_to_hub=true --control.fps=60 --control.reset_time_s=5 --control.warmup_time_s=3 --control.resume=true --control.num_image_writer_processes=4

python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.single_task="Grasp mug and place it on the table." --control.repo_id=sriramsk/debug --control.num_episodes=2 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --robot.use_eef=true --control.push_to_hub=false --control.fps=60 --control.reset_time_s=5 --control.warmup_time_s=3 --control.num_image_writer_processes=4```

`--control.repo_id` indicates the name with which the dataset is saved and uploaded to Huggingface (if `--control.push_to_hub` is enabled). `--control.resume` allows writing to an existing dataset. While recording, use left/right arrow keys to finish / reset current episode. `--robot.use_eef=true` runs forward kinematics and stores the computed eef pose in the dataset.
```

### Visualize and replay

```
python lerobot/scripts/visualize_dataset.py  --repo-id sriramsk/aloha_mug_eef   --episode-index 0
```

(Careful when replaying)
```
python lerobot/scripts/control_robot.py --robot.type=aloha --robot.cameras='{"cam_kinect": {"type": "opencv", "camera_index": 0, "fps": 30, "width": 1280, "height": 720}}' --control.type=replay --control.fps=60 --control.repo_id=sriramsk/aloha_mug_eef --control.episode=0
```
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

Some config parameters cannot be set at runtime and are instead saved in the config during training. To rollout with different high-level configs, modify `outputs/train/<model-name>/checkpoints/last/pretrained_model/config.json`

Visualize:
```
python lerobot/scripts/visualize_goal_gripper_proj.py --repo-id sriramsk/eval_fold_bottoms_multiTask_gc_dino3dgp_moreHuman
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

- Download source demos from [LIBERO](https://libero-project.github.io/datasets)
- Generate LeRobotDatasets from hdf5 files:

One task:
```
python lerobot/scripts/create_libero_dataset.py --hdf5_list libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5
```

All tasks:
```
python lerobot/scripts/create_libero_dataset.py --suite_names libero_object libero_goal libero_spatial libero_90 libero_10
```

- Training/eval:
```
python lerobot/scripts/train.py --dataset.repo_id=sriramsk/libero_lerobot --policy.type=diffusion --output_dir=outputs/train/diffPo_libero --job_name=diffPo_libero --policy.device=cuda --wandb.enable=true --policy.use_separate_rgb_encoder_per_camera=true --policy.use_text_embedding=true --policy.robot_type=libero_franka --env.type=libero --env.task=libero_object_0 --eval.batch_size=10
```

- Training/eval with goal conditioning:
```
python lerobot/scripts/train.py --dataset.repo_id=sriramsk/libero_lerobot_singleTask_heatmapGoal --policy.type=diffusion --output_dir=outputs/train/diffPo_libero_gc --job_name=diffPo_libero_gc --policy.device=cuda --wandb.enable=true --policy.use_separate_rgb_encoder_per_camera=true --policy.use_text_embedding=true --policy.enable_goal_conditioning=true --policy.robot_type=libero_franka --env.type=libero --env.task=libero_object_0 --eval.batch_size=10 --policy.hl_use_rgb=true --policy.hl_run_id=v8z0lx5h --policy.hl_max_depth=2 --policy.hl_in_channels=7 --policy.hl_intrinsics_txt=lerobot/scripts/libero_franka_calibration/intrinsics.txt --policy.hl_extrinsics_txt=lerobot/scripts/libero_franka_calibration/agentview_cam_to_world.txt
```

- Eval on suites:
```
python lerobot/scripts/eval_suite.py \
    --policy.path=outputs/train/diffPo_libero_90_lerobot/checkpoints/last/pretrained_model \
    --env.type=libero \
    --suite_name=libero_90 \
    --task_ids=0,1,2,3,4,5,6,7,8,9 \
    --eval.batch_size=10 --eval.n_episodes=20
```

## Preparing human data

After collecting human demos with the above commands or in `experiments.md`:

- Process with [wilor](https://github.com/sriramsk1999/wilor):
```
python demo_lerobot_detectron2.py --input_folder "/home/sriram/.cache/huggingface/lerobot/sriramsk/fold_bottoms_20250919_human/videos/chunk-000/" --output_folder "/data/sriram/lerobot_extradata/sriramsk/fold_bottoms_20250919_human/wilor_hand_pose"
```
- Annotate GT events (manually) using `lerobot/scripts/annotate_events.py`
- Process with `upgrade_dataset.py` (`humanize` i.e. keep the human in the video):
```
python upgrade_dataset.py --source_repo_id sriramsk/fold_bottoms_20250919_human --target_repo_id sriramsk/fold_bottoms_20250919_human_heatmapGoal --humanize --new_features goal_gripper_proj gripper_pcds next_event_idx
```
- Alternatively, to use `phantomize` to retarget the human demo, first set up [GSAM-2](https://github.com/sriramsk1999/Grounded-SAM-2) and generate masks:
```
python gsam2_lerobot.py sriramsk/fold_bottoms_multiview_20251031 cam_azure_kinect_front
```
- [E2FGVI](https://github.com/sriramsk1999/Grounded-SAM-2) to generate inpainted videos using the gsam2 masks:
```
# Currently set up on autobot, only works on the 20-class gpus

# Transfer data
rsync -ravz --progress /home/sriram/.cache/huggingface/lerobot/sriramsk/fold_bottoms_multiview_20251031/videos/chunk-000/observation.images.cam_azure_kinect_front.color  sskrishn@autobot.vision.cs.cmu.edu:/project_data/held/sskrishn/E2FGVI/examples/fold_bottoms_multiview_20251031

rsync -ravz --progress /data/sriram/lerobot_extradata/sriramsk/fold_bottoms_multiview_20251031/gsam2_masks  sskrishn@autobot.vision.cs.cmu.edu:/project_data/held/sskrishn/E2FGVI/examples/fold_bottoms_multiview_20251031

# infinite loop because some weird memory issue I haven't had time to fix. 
nohup bash -c 'while true; do python test_lerobot.py --lerobot_dir examples/fold_bottoms_multiview_20251031 --cam_name cam_azure_kinect_front; done' &
```
- Generate Phantom videos from lfd3d:
```
python run_phantom_lerobot.py --calib_json ../../src/lfd3d/datasets/aloha_calibration/multiview_calib.json --lerobot-extradata-path /data/sriram/lerobot_extradata/sriramsk/fold_bottoms_multiview_20251031
```

- And then:
```
python upgrade_dataset.py --source_repo_id sriramsk/fold_bottoms_20250919_human --target_repo_id sriramsk/fold_bottoms_20250919_phantom_heatmapGoal --phantomize --path_to_extradata /data/sriram/lerobot_extradata/ --new_features goal_gripper_proj gripper_pcds next_event_idx --extrinsics_txt /home/sriram/Desktop/lerobot/lerobot/scripts/aloha_calibration/T_world_from_camera_est_v6_0709.txt
```


## Scripts

There are many scripts for manipulating LeRobotDatasets in `lerobot/scripts`. LeRobot doesn't provide a simple way to add new keys / modify existing keys to a dataset. Instead we take the blunt approach of creating a new dataset, copying required keys and modifying/adding other keys.

```
python upgrade_dataset.py --source_repo_id sriramsk/human_mug_0718 --target_repo_id sriramsk/phantom_mug_0718_heatmapGoal --discard_episodes 2 10 11 13 21 --phantomize --path_to_extradata /data/sriram/lerobot_extradata/sriramsk/human_mug_0718 --push_to_hub --new_features goal_gripper_proj
```

`--source_repo_id` is the id of the existing dataset and `--target_repo_id` is the id of the new dataset being created. `--discard_episodes` skips problematic episodes which may exist in the source data, `--new_features` takes in a list of new features to be added (in this case, a heatmap image).

`--phantomize` and `--path_to_extradata` are extra arguments only required when retargeting a human demonstration dataset following the approach in[Phantom](https://phantom-human-videos.github.io/).

**Merge datasets:**
```bash
python lerobot/scripts/merge_datasets.py --datasets DATASET1_ID DATASET2_ID DATASET3_ID --target_repo_id MERGED_DATASET_ID --push_to_hub
```
Merges multiple LeRobot datasets into a single dataset. All input datasets must have compatible features and fps.

**Subsample dataset:**
```bash
python lerobot/scripts/subsample_dataset.py --source_repo_id SOURCE_ID --target_repo_id TARGET_ID --target_fps <fps>
```
Creates a subsampled version of a dataset by forcing a target fps.

# Launching on Orchard

0. (Optional): Make the pixi default cache directory elsewhere.

```
# Add the following line to your .bashrc or .zshrc file.
export PIXI_CACHE_DIR=/project/flame/$USER/.cache/pixi
```

```
# Create the config.toml file with the desired settings, which tells pixi to put environments in flame.
# Only necessary if you have multiple environments, because envs can get quite big, larger than the 20G allowance.
echo 'detached-environments = "/project/flame/$USER/envs"' > ~/.pixi/config.toml
```
1. Clone & install

```
git clone git@github.com:r-pad/lerobot.git
cd lerobot
# Minor complication: currently evdev is giving problems when using the gcc inside pixi env.
CC=/usr/bin/gcc CXX=/usr/bin/g++ pixi install

# Install pytorch3d INSIDE a slurm job. Because we can't pretend.
# This can take up to 20 minutes.
./cluster/launch-slurm.py -J install_pytorch3d --gpus 1 install-pytorch3d
```

2. Launch a job to actually run the training script. This can be made pretty flexible.
```
# MY_TRAINING_SCRIPT is exactly the same python command you write when training locally.
./cluster/launch-slurm.py -J train --gpus 1 --sync-logs $MY_TRAINING_SCRIPT
```

### Calibration Playbook

- In [lfd3d-system](https://github.com/r-pad/lfd3d-system), in separate terminals:
```
pixi run ros2 launch aloha aloha_bringup.launch.py robot:=aloha_stationary use_cameras:=false
pixi run python ros/src/aloha/scripts/teleop.py -r aloha_stationary
pixi run ros2 launch azure_kinect_ros_driver driver.launch.py overwrite_robot_description:=False color_resolution:=720P fps:=30 depth_mode:=NFOV_UNBINNED point_cloud:=True rgb_point_cloud:=True
pixi run ros2 run rviz2 rviz2 -d ros/src/lfd3d_ros/launch/rviz/4arms_pcd.rviz
pixi run ros2 launch lfd3d_ros rgbd_pipeline_launch.py
pixi run ros2 run lfd3d_ros broadcast_transform --transform_file captures/camera_left_v7_1013/T_world_from_camera_est.txt # or the latest calibration file
```

- Can verify quality of current calibration by checking robot pcd / robot urdf overlay quality. If not good enough, run in another terminal:
`pixi run ros2 run lfd3d_ros camera_calibration_collect_ros`

- Collect ~30 images where the aruco marker is detected in the camera by pressing 'c'.

- After this, in `/home/sriram/Desktop/calibration`, run:
```
uv run scripts/calibrate.py --output-dir /home/sriram/Desktop/lfd3d-system/captures/output_20251011_195009 # the latest capture
```
- The aruco marker needs to be mounted on the corresponding link for which we record the pose from `camera_calibration_collect_ros`. Might need to play around with hyperparams a bit but typically I see a loss of around 0.025 with the default hyperparams, and a pretty good overlay in rviz.

- After successful calibration, update `configuration_diffusion.py` and maybe the `config.json` files in the trained checkpoints.
- NOTE: The calibration repo is [over here](https://github.com/r-pad/calibration) but there are new changes on the `sriram/changes` branch which have not been pushed upstream because Sriram does not have access to the repo yet.

#### Clean shut-down after finishing calibration

- First, kill the terminal where `teleop.py` is running. Then, run:
```
pixi run python ros/src/aloha/scripts/teleop.py -r aloha_stationary
```
- After `sleep.py` has finished executing, kill all the other terminals
- Rename the latest capture:
```
mv captures/output_20251011_195009 captures/camera_<cam-name>_<calibration-date>_
```
- Copy the latest `T_world_from_camera_est.txt` into `lerobot/lerobot/scripts/aloha_calibration.txt`
- Update the corresponding `calibration.json`
- And done!
