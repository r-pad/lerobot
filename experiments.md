# Experiment / Command Log

Just a small log of commands to keep track of things ...

## Dataset Collection

Teleop script

```py
python lerobot/scripts/control_robot.py --robot.type=aloha --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --control.type=teleoperate --control.display_data=true
```

Multiview:

```py
python lerobot/scripts/control_robot.py --robot.type=aloha --robot.cameras='{"cam_azure_kinect_back": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true, "wired_sync_mode": "master"}, "cam_azure_kinect_front": {"type": "azurekinect", "device_id": 1, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true, "wired_sync_mode": "subordinate", "subordinate_delay_off_master_usec": 200}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --control.type=teleoperate --control.display_data=true
```

Need to be careful when setting up multiview, various things to keep in mind:
- Check which camera is master/subordinate in the real world and set `wired_sync_mode` accordingly
- Usb 10gb usb 3.0 ports and make sure the cameras are spread across different usb controllers (different ports on the outside might correspond to the same internal controller, check `lsusb -t`)
- If teleop crashes for some reason, check all these separately, unplug/replug/reboot as necessary. None of these systems are reliable ....


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


### Non-goal conditioned train/rollout commands, trained on [plate on platform, plate in bin, mug in bin]

```py
nohup python lerobot/scripts/train.py --dataset.repo_id=sriramsk/Plate_on_platformBin_mugBin_subsampled_noWrist --policy.type=diffusion --output_dir=outputs/train/diffpo_Plate_on_platformBin_mugBin_subsampled_noWrist --job_name=diffPo_Plate_on_platformBin_mugBin_subsampled_noWrist --policy.device=cuda --wandb.enable=true --policy.use_separate_rgb_encoder_per_camera=true --policy.use_text_embedding=true --policy.crop_shape="[600, 600]" --policy.crop_is_random=false --dataset.image_transforms.enable=true > subsample3.out &


# python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Grasp plate and place it on the platform." --control.repo_id=sriramsk/eval_mug_on_platform_debug --control.num_episodes=1 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffpo_Plate_on_platformBin_mugBin_subsampled_noWrist/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=120


# Plate on Platform eval

python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Grasp plate and place it on the platform." --control.repo_id=sriramsk/eval_ll_ACD_plate_on_platform --control.num_episodes=20 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffpo_Plate_on_platformBin_mugBin_subsampled_noWrist/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=120

# Mug on platform eval

python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Grasp mug and place it on the platform." --control.repo_id=sriramsk/eval_ll_ACD_mug_on_platform --control.num_episodes=20 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffpo_Plate_on_platformBin_mugBin_subsampled_noWrist/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=120
```

### High-level trained on trained on [plate on platform, plate in bin, mug in bin, mug on platform] || Low-level trained on trained on [plate on platform, plate in bin, mug in bin]

```py
CUDA_VISIBLE_DEVICES=1 nohup python lerobot/scripts/train.py --dataset.repo_id=sriramsk/Plate_platformBin_mugBin_subsmpld_noWrist_GC --policy.type=diffusion --output_dir=outputs/train/diffpo_Plate_platformBin_mugBin_subsmpld_noWrist_heatmap_goal4-high-level --job_name=diffPo_Plate_on_platformBin_mugBin_subsampled_noWrist_heatmapGoal --policy.device=cuda --wandb.enable=true --policy.use_separate_rgb_encoder_per_camera=true --policy.use_text_embedding=true --policy.crop_shape="[600, 600]" --policy.crop_is_random=false --dataset.image_transforms.enable=true --policy.enable_goal_conditioning=true > goal3.out &


# Mug on platform eval

python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Grasp mug and place it on the platform." --control.repo_id=sriramsk/eval_hl_ABCD_ll_ACD_mug_on_platform --control.num_episodes=20 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffpo_Plate_platformBin_mugBin_subsmpld_noWrist_heatmap_goal4-high-level/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=120

# Plate on Platform eval

python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Grasp plate and place it on the platform." --control.repo_id=sriramsk/eval_hl_ABCD_ll_ACD_plate_on_platform --control.num_episodes=20 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffpo_Plate_platformBin_mugBin_subsmpld_noWrist_heatmap_goal4-high-level/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=120
```

### High-level trained on trained on [plate on platform, plate in bin, mug in bin] || Low-level trained on trained on [plate on platform, plate in bin, mug in bin]

```py
# Mug on platform eval

python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Grasp mug and place it on the platform." --control.repo_id=sriramsk/eval_hl_ABCD_ll_ACD_mug_on_platform_goal3 --control.num_episodes=20 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffpo_Plate_platformBin_mugBin_subsmpld_noWrist_heatmap_goal3-high-level/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=120

# Plate on Platform eval

python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Grasp plate and place it on the platform." --control.repo_id=sriramsk/eval_hl_ABCD_ll_ACD_plate_on_platform_goal3 --control.num_episodes=20 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffpo_Plate_platformBin_mugBin_subsmpld_noWrist_heatmap_goal3-high-level/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=120
```

### High-level trained on [plate on platform, plate in bin, mug in bin, mug on platform] with RGB || Low-level trained on [plate on platform, plate in bin, mug in bin]

```py
# Plate on Platform eval

python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Grasp plate and place it on the platform." --control.repo_id=sriramsk/eval_hl_ABCD_ll_ACD_plate_on_platform_goal4_withRgb --control.num_episodes=20 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffpo_Plate_platformBin_mugBin_subsmpld_noWrist_heatmap_goal4-high-level-withRgb/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=120
```

### High-level trained on [plate on platform, plate in bin, mug in bin, mug on platform (and HUMAN)] with RGB || Low-level trained on [plate on platform, plate in bin, mug in bin]

```py
# Mug on Platform eval

python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Grasp mug and place it on the platform." --control.repo_id=sriramsk/eval_hl_ABCD_ll_ACD_mug_on_platform_goal4_withRgbHuman --control.num_episodes=20 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffpo_Plate_platformBin_mugBin_subsmpld_noWrist_heatmap_goal4-high-level-withRgb_human/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=120
```

### High-level trained on [plate on platform, plate in bin, mug in bin, mug on platform (and HUMAN phantomized)] with RGB || Low-level trained on [plate on platform, plate in bin, mug in bin]

```py
# Mug on Platform eval

python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Grasp mug and place it on the platform." --control.repo_id=sriramsk/eval_hl_ABCD_ll_ACD_mug_on_platform_goal4_withRgbPhantom --control.num_episodes=20 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffpo_Plate_platformBin_mugBin_subsmpld_noWrist_heatmap_goal4-high-level-withRgb_phantom/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=120
```

### High-level trained on [plate on platform, plate in bin, mug in bin, mug on platform (HUMAN)] with RGB and two-head arch || Low-level trained on [plate on platform, plate in bin, mug in bin]

```py
# Mug on Platform eval

python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Grasp mug and place it on the platform." --control.repo_id=sriramsk/eval_hl_ABCD_ll_ACD_mug_on_platform_goal4_withRgbHuman0910TwoHead --control.num_episodes=20 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffpo_Plate_platformBin_mugBin_subsmpld_noWrist_heatmap_goal4-high-level-withRgb_human0910_twoHead/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=120
```

### Specialist policy with wrist cam

```py
nohup python lerobot/scripts/train.py --dataset.repo_id=sriramsk/plate_in_bin_20250821_subsampled --policy.type=diffusion --output_dir=outputs/train/diffpo_plate_in_bin_subsampled_withWrist --job_name=diffPo_plate_in_bin_subsampled_withWrist --policy.device=cuda --wandb.enable=true --policy.use_separate_rgb_encoder_per_camera=true --policy.use_text_embedding=true --policy.crop_shape="[600, 600]" --policy.crop_is_random=false --dataset.image_transforms.enable=true > plate_in_bin_withWrist.out &

python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Grasp plate and place it in the bin." --control.repo_id=sriramsk/eval_plate_in_bin_wristCam --control.num_episodes=20 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffpo_plate_in_bin_subsampled_withWrist/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=120

## sanity check with non-gc multitask policy

CUDA_VISIBLE_DEVICES=1 python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Grasp plate and place it in the bin." --control.repo_id=sriramsk/eval_ll_ACD_plate_in_bin --control.num_episodes=20 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": false}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffpo_Plate_on_platformBin_mugBin_subsampled_noWrist/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=120

```

### Fold onesie (A)

BE CAREFUL WHEN SETTING `--robot.max_relative_target=null`, disables all clamping of extreme motions during teleop.

```py
python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.single_task="Fold the onesie." --control.repo_id=sriramsk/fold_onesie_20250831 --control.num_episodes=20 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --robot.use_eef=true --control.push_to_hub=true --control.fps=30 --control.reset_time_s=5 --control.warmup_time_s=3 --control.num_image_writer_processes=4 --control.display_data=true --robot.max_relative_target=null

nohup python lerobot/scripts/train.py --dataset.repo_id=sriramsk/fold_onesie_20250831_subsampled --policy.type=diffusion --output_dir=outputs/train/diffPo_onesie_subsampled --job_name=diffPo_onesie_subsampled --policy.device=cuda --wandb.enable=true --policy.use_separate_rgb_encoder_per_camera=true --policy.use_text_embedding=true --steps=300_000 --policy.crop_shape="[600, 600]" --policy.crop_is_random=false --dataset.image_transforms.enable=true > diffpo_onesie_subsampled.out

nohup python lerobot/scripts/train.py --dataset.repo_id=sriramsk/fold_onesie_20250831_subsampled_heatmapGoal --policy.type=diffusion --output_dir=outputs/train/diffPo_onesie_subsampled_heatmapGoal --job_name=diffPo_onesie_subsampled_heatmapGoal --wandb.enable=true --policy.use_text_embedding=true --steps=300_000 --policy.crop_shape="[600, 600]" --policy.crop_is_random=false --policy.enable_goal_conditioning=true > diffpo_onesie_subsampled_heatmapGoal.out &

CUDA_VISIBLE_DEVICES=1 python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Fold the onesie." --control.repo_id=sriramsk/eval_fold_onesie_specialist --control.num_episodes=1 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffPo_onesie_subsampled/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=120

python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Fold the onesie." --control.repo_id=sriramsk/eval_fold_onesie_specialist_gc --control.num_episodes=10 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffPo_onesie_subsampled_heatmapGoal/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=240
```

### Fold shirt (B)

BE CAREFUL WHEN SETTING `--robot.max_relative_target=null`, disables all clamping of extreme motions during teleop.

```py
python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.single_task="Fold the shirt." --control.repo_id=sriramsk/fold_shirt_20250918 --control.num_episodes=50 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --robot.use_eef=true --control.push_to_hub=true --control.fps=30 --control.reset_time_s=5 --control.warmup_time_s=3 --control.num_image_writer_processes=4 --control.display_data=true --robot.max_relative_target=null
```

### Fold towel (C)

BE CAREFUL WHEN SETTING `--robot.max_relative_target=null`, disables all clamping of extreme motions during teleop.

```py
python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.single_task="Fold the towel." --control.repo_id=sriramsk/fold_towel_20250919 --control.num_episodes=50 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --robot.use_eef=true --control.push_to_hub=true --control.fps=30 --control.reset_time_s=5 --control.warmup_time_s=3 --control.num_image_writer_processes=8 --control.display_data=true --robot.max_relative_target=null
```

### Fold onesie, shirt, towel, pants/skirt (A/B/C/D) - human demo

Human demo, note that its collected at 15fps.

```py

python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.single_task="Fold the onesie." --control.repo_id=sriramsk/fold_onesie_20251015_human --control.num_episodes=10 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --robot.use_eef=true --control.push_to_hub=true --control.fps=15 --control.reset_time_s=5 --control.warmup_time_s=3 --control.num_image_writer_processes=8 --control.display_data=true

python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.single_task="Fold the shirt." --control.repo_id=sriramsk/fold_shirt_20251015_human --control.num_episodes=10 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --robot.use_eef=true --control.push_to_hub=true --control.fps=15 --control.reset_time_s=5 --control.warmup_time_s=3 --control.num_image_writer_processes=8 --control.display_data=true

python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.single_task="Fold the towel." --control.repo_id=sriramsk/fold_towel_20251015_human --control.num_episodes=10 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --robot.use_eef=true --control.push_to_hub=true --control.fps=15 --control.reset_time_s=5 --control.warmup_time_s=3 --control.num_image_writer_processes=8 --control.display_data=true

python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.single_task="Fold the bottoms." --control.repo_id=sriramsk/fold_bottoms_20250919_human --control.num_episodes=50 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --robot.use_eef=true --control.push_to_hub=true --control.fps=15 --control.reset_time_s=5 --control.warmup_time_s=3 --control.num_image_writer_processes=8 --control.display_data=true
```

### Multi-task folders

```sh

nohup python lerobot/scripts/train.py --dataset.repo_id='["sriramsk/fold_onesie_20250831_subsampled", "sriramsk/fold_shirt_20250918_subsampled", "sriramsk/fold_towel_20250919_subsampled"]' --policy.type=diffusion --output_dir=outputs/train/diffPo_multifold_subsampled --job_name=diffPo_multifold_subsampled --wandb.enable=true --policy.use_text_embedding=true --steps=300_000 "--policy.crop_shape=[600, 600]" --policy.crop_is_random=false &

HF_HOME="/scratch/sskrishn/lerobot" nohup python lerobot/scripts/train.py --dataset.repo_id='["sriramsk/fold_onesie_20250831_subsampled_heatmapGoal", "sriramsk/fold_shirt_20250918_subsampled_heatmapGoal", "sriramsk/fold_towel_20250919_subsampled_heatmapGoal"]' --policy.type=diffusion --output_dir=outputs/train/diffPo_multifold_subsampled_heatmapGoal --job_name=diffPo_multifold_subsampled_heatmapGoal --wandb.enable=true --policy.use_text_embedding=false --steps=300_000 "--policy.crop_shape=[600, 600]" --policy.crop_is_random=false --policy.enable_goal_conditioning=true > multifold_gc_diffpo.out &

CUDA_VISIBLE_DEVICES=1 nohup python scripts/train.py model=articubot dataset=rpadLerobot dataset.repo_id="[sriramsk/fold_onesie_20250831_subsampled_heatmapGoal, sriramsk/fold_shirt_20250918_subsampled_heatmapGoal, sriramsk/fold_towel_20250919_subsampled_heatmapGoal, sriramsk/fold_bottoms_20250919_human_heatmapGoal]"  resources.num_workers=16 training.batch_size=8 training.check_val_every_n_epochs=3 dataset.cache_dir=/home/sriram/Desktop/lfd3d/multifold_cache model.use_rgb=True model.in_channels=7 model.use_dual_head=True > multifold.out &

NCCL_P2P_LEVEL=NVL HF_HOME="/scratch/sskrishn/lerobot" nohup torchrun --nproc_per_node=8 scripts/train.py model=articubot dataset=rpadLerobot dataset.repo_id="[sriramsk/fold_onesie_20250831_subsampled_heatmapGoal, sriramsk/fold_shirt_20250918_subsampled_heatmapGoal, sriramsk/fold_towel_20250919_subsampled_heatmapGoal, sriramsk/fold_bottoms_20250919_human_heatmapGoal]" resources.num_workers=16 training.batch_size=4 training.check_val_every_n_epochs=3 dataset.cache_dir=/scratch/sskrishn/multifold_cache model.use_rgb=True model.in_channels=7 resources.gpus=-1 > multifold.out &
```

Eval:

```
python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Fold the onesie." --control.repo_id=sriramsk/eval_fold_onesie_multiTask --control.num_episodes=10 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffPo_multifold_subsampled/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=120

python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Fold the onesie." --control.repo_id=sriramsk/eval_fold_onesie_multiTask_gc --control.num_episodes=10 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffPo_multifold_subsampled_heatmapGoal/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=240

python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Fold the bottoms." --control.repo_id=sriramsk/eval_fold_bottoms_multiTask --control.num_episodes=10 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffPo_multifold_subsampled/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=120

python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Fold the bottoms." --control.repo_id=sriramsk/eval_fold_bottoms_multiTask_gc --control.num_episodes=10 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffPo_multifold_subsampled_heatmapGoal/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=240
```

Single-channel and with DinoHeatmapNetwork:

```sh

HF_HOME="/scratch/sskrishn/lerobot" nohup python lerobot/scripts/train.py --dataset.repo_id='["sriramsk/fold_onesie_20250831_subsampled_heatmapGoal", "sriramsk/fold_shirt_20250918_subsampled_heatmapGoal", "sriramsk/fold_towel_20250919_subsampled_heatmapGoal"]' --policy.type=diffusion --output_dir=outputs/train/diffPo_multifold_subsampled_heatmapGoal_oneChannel --job_name=diffPo_multifold_subsampled_heatmapGoal_oneChannel --wandb.enable=true --policy.use_text_embedding=false --steps=300_000 --policy.crop_shape="[600, 600]" --policy.crop_is_random=false --policy.enable_goal_conditioning=true --policy.hl_model_type="dino_heatmap" --policy.use_single_channel_goal=true > multifold_gc_diffpo_oneChannel.out &


nohup python scripts/train.py model=dino_heatmap dataset=rpadLerobot dataset.repo_id="[sriramsk/fold_onesie_20250831_subsampled_heatmapGoal, sriramsk/fold_shirt_20250918_subsampled_heatmapGoal, sriramsk/fold_towel_20250919_subsampled_heatmapGoal, sriramsk/fold_bottoms_20250919_human_heatmapGoal]"  resources.num_workers=32 dataset.cache_dir=/home/sriram/Desktop/lfd3d/dino_heatmap_multifold_cache training.epochs=500 > dino_heatmap_multifold.out &
```

Eval:

```sh
python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Fold the onesie." --control.repo_id=sriramsk/eval_fold_onesie_multiTask_gc_oneChannel --control.num_episodes=10 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffPo_multifold_subsampled_heatmapGoal_oneChannel/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=240


python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Fold the bottoms." --control.repo_id=sriramsk/eval_fold_bottoms_multiTask_gc_oneChannel --control.num_episodes=10 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffPo_multifold_subsampled_heatmapGoal_oneChannel/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=240
```

Eval with Dino-3DGP:

```sh
python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Fold the onesie." --control.repo_id=sriramsk/eval_fold_onesie_multiTask_gc_dino3dgp --control.num_episodes=10 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffPo_multifold_subsampled_heatmapGoal_dino3dgp/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=240

python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Fold the bottoms." --control.repo_id=sriramsk/eval_fold_bottoms_multiTask_gc_dino3dgp --control.num_episodes=10 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffPo_multifold_subsampled_heatmapGoal_dino3dgp/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=240
```


## Multiview
### Fold onesie (A)

BE CAREFUL WHEN SETTING `--robot.max_relative_target=null`, disables all clamping of extreme motions during teleop.

```py
python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.single_task="Fold the onesie." --control.repo_id=sriramsk/fold_onesie_multiview_20251025 --control.num_episodes=50 --robot.cameras='{"cam_azure_kinect_back": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true, "wired_sync_mode": "master"}, "cam_azure_kinect_front": {"type": "azurekinect", "device_id": 1, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true, "wired_sync_mode": "subordinate", "subordinate_delay_off_master_usec": 200}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --robot.use_eef=true --control.push_to_hub=true --control.fps=30 --control.reset_time_s=5 --control.warmup_time_s=3 --control.num_image_writer_processes=4 --control.display_data=false --robot.max_relative_target=null

nohup python lerobot/scripts/train.py --dataset.repo_id=sriramsk/fold_onesie_multiview_20251025_subsampled --policy.type=diffusion --output_dir=outputs/train/diffPo_multiview_foldOnesie_subsampled --job_name=diffPo_multiview_foldOnesie_subsampled --wandb.enable=true --policy.use_text_embedding=true --steps=300_000 --policy.crop_shape="[700, 700]" --policy.crop_is_random=true &

HF_HOME="/scratch/sskrishn/lerobot" nohup python lerobot/scripts/train.py --dataset.repo_id=sriramsk/fold_onesie_MV_20251025_ss_hg --policy.type=diffusion --output_dir=outputs/train/diffPo_MV_foldOnesie_ss_hg --job_name=diffPo_MV_foldOnesie_ss_hg --wandb.enable=true --policy.use_text_embedding=false --steps=300_000 --policy.crop_shape="[700, 700]" --policy.crop_is_random=true --policy.enable_goal_conditioning=true --batch_size=4 > foldOnesie_multiview_gc_diffPo.out &

NCCL_P2P_LEVEL=NVL HF_HOME="/scratch/sskrishn/lerobot" nohup torchrun --nproc_per_node=8 scripts/train.py model=dino_3dgp dataset=rpadLerobot dataset.repo_id=sriramsk/fold_onesie_MV_20251025_ss_hg resources.num_workers=32 training.check_val_every_n_epochs=3 dataset.cache_dir=/scratch/sskrishn/multiview_cache resources.gpus=-1 > multiview.out &

python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Fold the onesie." --control.repo_id=sriramsk/eval_fold_onesie_multiview --control.num_episodes=10 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect_back": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true, "wired_sync_mode": "master"}, "cam_azure_kinect_front": {"type": "azurekinect", "device_id": 1, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true, "wired_sync_mode": "subordinate", "subordinate_delay_off_master_usec": 200}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffPo_multiview_foldOnesie_subsampled/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=120

python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Fold the onesie." --control.repo_id=sriramsk/eval_fold_onesie_multiview_gc --control.num_episodes=10 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect_back": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true, "wired_sync_mode": "master"}, "cam_azure_kinect_front": {"type": "azurekinect", "device_id": 1, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true, "wired_sync_mode": "subordinate", "subordinate_delay_off_master_usec": 200}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffPo_MV_foldOnesie_ss_hg/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=240
```


### Fold shirt (B)

BE CAREFUL WHEN SETTING `--robot.max_relative_target=null`, disables all clamping of extreme motions during teleop.

```py
python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.single_task="Fold the shirt." --control.repo_id=sriramsk/fold_shirt_multiview_20251030 --control.num_episodes=50 --robot.cameras='{"cam_azure_kinect_back": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true, "wired_sync_mode": "master"}, "cam_azure_kinect_front": {"type": "azurekinect", "device_id": 1, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true, "wired_sync_mode": "subordinate", "subordinate_delay_off_master_usec": 200}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --robot.use_eef=true --control.push_to_hub=true --control.fps=30 --control.reset_time_s=5 --control.warmup_time_s=3 --control.num_image_writer_processes=4 --control.display_data=false --robot.max_relative_target=null
```

### Fold towel (C)

BE CAREFUL WHEN SETTING `--robot.max_relative_target=null`, disables all clamping of extreme motions during teleop.

```py
python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.single_task="Fold the towel." --control.repo_id=sriramsk/fold_towel_multiview_20251030 --control.num_episodes=50 --robot.cameras='{"cam_azure_kinect_back": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true, "wired_sync_mode": "master"}, "cam_azure_kinect_front": {"type": "azurekinect", "device_id": 1, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true, "wired_sync_mode": "subordinate", "subordinate_delay_off_master_usec": 200}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --robot.use_eef=true --control.push_to_hub=true --control.fps=30 --control.reset_time_s=5 --control.warmup_time_s=3 --control.num_image_writer_processes=4 --control.display_data=false --robot.max_relative_target=null
```

### Fold bottoms (D)

```python
python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.single_task="Fold the bottoms." --control.repo_id=sriramsk/fold_bottoms_multiview_20251031 --control.num_episodes=50 --robot.cameras='{"cam_azure_kinect_back": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true, "wired_sync_mode": "master"}, "cam_azure_kinect_front": {"type": "azurekinect", "device_id": 1, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true, "wired_sync_mode": "subordinate", "subordinate_delay_off_master_usec": 200}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --robot.use_eef=true --control.push_to_hub=true --control.fps=15 --control.reset_time_s=5 --control.warmup_time_s=3 --control.num_image_writer_processes=4 --control.display_data=true
```

### Multi-view Multi-task folders

```python
HF_HOME="/scratch/sskrishn/lerobot" nohup python lerobot/scripts/train.py --dataset.repo_id='["sriramsk/fold_onesie_MV_20251025_ss_hg", "sriramsk/fold_towel_MV_20251030_ss_hg", "sriramsk/fold_shirt_MV_20251030_ss_hg"]' --policy.type=diffusion --output_dir=outputs/train/diffPo_MV_multiFold_ss_hg --job_name=diffPo_MV_multiFold_ss_hg --wandb.enable=true --policy.use_text_embedding=false --steps=300_000 --policy.crop_shape="[700, 700]" --policy.crop_is_random=true --policy.enable_goal_conditioning=true --batch_size=6 > multiFold_multiview_gc_diffPo.out &

NCCL_P2P_LEVEL=NVL HF_HOME="/scratch/sskrishn/lerobot" nohup torchrun --nproc_per_node=8 scripts/train.py model=dino_3dgp dataset=rpadLerobot dataset.repo_id='["sriramsk/fold_onesie_MV_20251025_ss_hg", "sriramsk/fold_towel_MV_20251030_ss_hg", "sriramsk/fold_shirt_MV_20251030_ss_hg", "sriramsk/fold_bottoms_MV_20251031_ss_hg"]' resources.num_workers=32 training.check_val_every_n_epochs=3 dataset.cache_dir=/scratch/sskrishn/multiview_cache resources.gpus=-1 > multiview_multifold.out &

python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.fps=15 --control.single_task="Fold the onesie." --control.repo_id=sriramsk/eval_fold_onesie_multiview_multifold_gc --control.num_episodes=10 --control.reset_time_s=5 --control.warmup_time_s=3 --robot.cameras='{"cam_azure_kinect_back": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true, "wired_sync_mode": "master"}, "cam_azure_kinect_front": {"type": "azurekinect", "device_id": 1, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true, "wired_sync_mode": "subordinate", "subordinate_delay_off_master_usec": 200}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --robot.use_eef=true --control.push_to_hub=false --control.policy.path=outputs/train/diffPo_MV_multiFold_ss_hg/checkpoints/last/pretrained_model/ --control.display_data=true --control.episode_time_s=240
```
