# Basic Transfer Experiment

Just a small log of commands to keep track of things ...

## Dataset Collection

Teleop script

```py
python lerobot/scripts/control_robot.py --robot.type=aloha --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --control.type=teleoperate --control.display_data=true
```

Tips:
* Maintain start and end pose: gripper in holster, handle vertical and touching base, gripper open.

### Mug on platform (A)

BE CAREFUL WHEN SETTING `--robot.max_relative_target=null`, disables all clamping of extreme motions during teleop.

```py
python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.single_task="Grasp mug and place it on the platform." --control.repo_id=sriramsk/mug_on_platform_20250821 --control.num_episodes=20 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --robot.use_eef=true --control.push_to_hub=true --control.fps=30 --control.reset_time_s=5 --control.warmup_time_s=3 --control.num_image_writer_processes=4 --control.display_data=false --robot.max_relative_target=null
```

### Plate on platform (B)

```py
python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.single_task="Grasp plate and place it on the platform." --control.repo_id=sriramsk/plate_on_platform_20250821 --control.num_episodes=20 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --robot.use_eef=true --control.push_to_hub=true --control.fps=30 --control.reset_time_s=5 --control.warmup_time_s=3 --control.num_image_writer_processes=4 --control.display_data=false --robot.max_relative_target=null
```

### Mug in bin (C)

```py
python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.single_task="Grasp mug and place it in the bin." --control.repo_id=sriramsk/mug_in_bin_20250821 --control.num_episodes=20 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --robot.use_eef=true --control.push_to_hub=true --control.fps=30 --control.reset_time_s=5 --control.warmup_time_s=3 --control.num_image_writer_processes=4 --control.display_data=false --robot.max_relative_target=null
```

### Plate in bin (D)

```py
python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.single_task="Grasp plate and place it in the bin." --control.repo_id=sriramsk/plate_in_bin_20250821 --control.num_episodes=20 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --robot.use_eef=true --control.push_to_hub=true --control.fps=30 --control.reset_time_s=5 --control.warmup_time_s=3 --control.num_image_writer_processes=4 --control.display_data=false --robot.max_relative_target=null
```

## Specialist policies (ignore - these policies were trained without cropping/augmentations, are brittle)

```py
python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Grasp plate and place it in the bin." --control.repo_id=sriramsk/eval_plate_in_bin --control.num_episodes=1 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/plate_in_bin_0821/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=120

python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Grasp mug and place it in the bin." --control.repo_id=sriramsk/eval_mug_in_bin --control.num_episodes=1 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/mug_in_bin_0821/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=120


python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Grasp plate and place it on the platform." --control.repo_id=sriramsk/eval_plate_on_platform --control.num_episodes=1 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/plate_on_platform_0821/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=120


python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Grasp mug and place it on the platform." --control.repo_id=sriramsk/eval_mug_on_platform --control.num_episodes=1 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/mug_on_platform_0821/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=120

```

## Multi-task policies (trained with cropping/augmentations)


Non-goal conditioned train/rollout commands, trained on [plate on platform, plate in bin, mug in bin]

```py
nohup python lerobot/scripts/train.py --dataset.repo_id=sriramsk/Plate_on_platformBin_mugBin_subsampled_noWrist --policy.type=diffusion --output_dir=outputs/train/diffpo_Plate_on_platformBin_mugBin_subsampled_noWrist --job_name=diffPo_Plate_on_platformBin_mugBin_subsampled_noWrist --policy.device=cuda --wandb.enable=true --policy.use_separate_rgb_encoder_per_camera=true --policy.use_text_embedding=true --policy.crop_shape="[600, 600]" --policy.crop_is_random=false --dataset.image_transforms.enable=true > subsample3.out &


# python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Grasp plate and place it on the platform." --control.repo_id=sriramsk/eval_mug_on_platform_debug --control.num_episodes=1 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffpo_Plate_on_platformBin_mugBin_subsampled_noWrist/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=120


# Plate on Platform eval

python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Grasp plate and place it on the platform." --control.repo_id=sriramsk/eval_ll_ACD_plate_on_platform --control.num_episodes=20 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffpo_Plate_on_platformBin_mugBin_subsampled_noWrist/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=120

# Mug on platform eval

python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Grasp mug and place it on the platform." --control.repo_id=sriramsk/eval_ll_ACD_mug_on_platform --control.num_episodes=20 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffpo_Plate_on_platformBin_mugBin_subsampled_noWrist/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=120
```

Goal conditioned train/rollout commands

High-level trained on trained on [plate on platform, plate in bin, mug in bin, mug on platform]
Low-level trained on trained on [plate on platform, plate in bin, mug in bin]

```py
CUDA_VISIBLE_DEVICES=1 nohup python lerobot/scripts/train.py --dataset.repo_id=sriramsk/Plate_platformBin_mugBin_subsmpld_noWrist_GC --policy.type=diffusion --output_dir=outputs/train/diffpo_Plate_platformBin_mugBin_subsmpld_noWrist_heatmap_goal4-high-level --job_name=diffPo_Plate_on_platformBin_mugBin_subsampled_noWrist_heatmapGoal --policy.device=cuda --wandb.enable=true --policy.use_separate_rgb_encoder_per_camera=true --policy.use_text_embedding=true --policy.crop_shape="[600, 600]" --policy.crop_is_random=false --dataset.image_transforms.enable=true --policy.enable_goal_conditioning=true > goal3.out &


python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Grasp mug and place it in the bin." --control.repo_id=sriramsk/eval_mug_on_platform_debug --control.num_episodes=1 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffpo_Plate_platformBin_mugBin_subsmpld_noWrist_heatmap_goal4-high-level/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=120


# Mug on platform eval

python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Grasp mug and place it on the platform." --control.repo_id=sriramsk/eval_hl_ABCD_ll_ACD_mug_on_platform --control.num_episodes=20 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffpo_Plate_platformBin_mugBin_subsmpld_noWrist_heatmap_goal4-high-level/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=120

# Plate on Platform eval

python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Grasp plate and place it on the platform." --control.repo_id=sriramsk/eval_hl_ABCD_ll_ACD_plate_on_platform --control.num_episodes=20 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffpo_Plate_platformBin_mugBin_subsmpld_noWrist_heatmap_goal4-high-level/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=120
```

High-level trained on trained on [plate on platform, plate in bin, mug in bin]
Low-level trained on trained on [plate on platform, plate in bin, mug in bin]

```py
# Mug on platform eval

python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Grasp mug and place it on the platform." --control.repo_id=sriramsk/eval_hl_ABCD_ll_ACD_mug_on_platform_goal3 --control.num_episodes=20 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffpo_Plate_platformBin_mugBin_subsmpld_noWrist_heatmap_goal3-high-level/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=120

# Plate on Platform eval

python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Grasp plate and place it on the platform." --control.repo_id=sriramsk/eval_hl_ABCD_ll_ACD_plate_on_platform_goal3 --control.num_episodes=20 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffpo_Plate_platformBin_mugBin_subsmpld_noWrist_heatmap_goal3-high-level/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=120
```

### Specialist policy with wrist cam

```py
nohup python lerobot/scripts/train.py --dataset.repo_id=sriramsk/plate_in_bin_20250821_subsampled --policy.type=diffusion --output_dir=outputs/train/diffpo_plate_in_bin_subsampled_withWrist --job_name=diffPo_plate_in_bin_subsampled_withWrist --policy.device=cuda --wandb.enable=true --policy.use_separate_rgb_encoder_per_camera=true --policy.use_text_embedding=true --policy.crop_shape="[600, 600]" --policy.crop_is_random=false --dataset.image_transforms.enable=true > plate_in_bin_withWrist.out &
```
