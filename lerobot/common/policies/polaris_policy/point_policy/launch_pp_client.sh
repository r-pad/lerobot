#!/bin/bash

python /home/haotian/lerobot/lerobot/scripts/control_robot.py \
    --robot.type=droid \
    --robot.skip_gello_calibration=true \
    --robot.use_eef=true \
    --robot.cameras='{"cam_azure_kinect_left": {"type": "azurekinect", "device_id": 1, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": false, "wired_sync_mode": "master"}, "cam_azure_kinect_front": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": false, "wired_sync_mode": "subordinate", "subordinate_delay_off_master_usec": 200}}' \
    --control.type=record \
    --control.single_task="pick_place_toys." \
    --control.repo_id=xiaochyVera/eval_pp_pick_place_toys_r100h300\
    --control.num_episodes=31 \
    --control.fps=15 \
    --control.episode_time_s=300 \
    --control.reset_time_s=5 \
    --control.warmup_time_s=3 \
    --control.push_to_hub=false \
    --control.display_data=true \
    --control.policy.type=point-policy \
    --control.policy.host=localhost \
    --control.policy.port=8766 \
    --control.policy.cam_image_keys='["observation.images.cam_azure_kinect_front.color", "observation.images.cam_azure_kinect_left.color"]' \
    --control.policy.use_ik=true \
    --control.policy.undo_z_rotation_deg=45.0 \
    --control.policy.vis_tracks=true \
    --control.policy.vis_shape='[720, 2560, 3]' \
    --control.resume=true 