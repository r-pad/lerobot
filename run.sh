### for reset the robot to the initial pose
python lerobot/scripts/control_robot.py --robot.type=aloha --robot.cameras='{"cam_azure_kinect_back": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true, "wired_sync_mode": "master"}, "cam_azure_kinect_front": {"type": "azurekinect", "device_id": 1, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true, "wired_sync_mode": "subordinate", "subordinate_delay_off_master_usec": 200}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --control.type=teleoperate --control.display_data=true


### for running policy
python lerobot/scripts/control_robot.py \
    --robot.type=aloha \
    --control.type=record \
    --control.fps=15 \
    --control.single_task="Fold the onesie." \
    --control.repo_id=sriramsk/eval_fold_ood_onesie_MV_gc_robotDataOnly \
    --control.num_episodes=10 \
    --control.reset_time_s=5 \
    --control.warmup_time_s=3 \
    --robot.cameras='{"cam_azure_kinect_back": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true, "wired_sync_mode": "master"}, "cam_azure_kinect_front": {"type": "azurekinect", "device_id": 1, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true, "wired_sync_mode": "subordinate", "subordinate_delay_off_master_usec": 200}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' \
    --robot.use_eef=true \
    --control.push_to_hub=false \
    --control.display_data=true \
    --control.episode_time_s=240

### for recording teleoperation data 
python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record --control.single_task="Hammer the pink pin." --control.repo_id=sriramsk/hammer_pinkPin_MV_20260119 --control.num_episodes=25 --robot.cameras='{"cam_azure_kinect_back": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true, "wired_sync_mode": "master"}, "cam_azure_kinect_front": {"type": "azurekinect", "device_id": 1, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true, "wired_sync_mode": "subordinate", "subordinate_delay_off_master_usec": 200}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' --robot.use_eef=true --control.push_to_hub=true --control.fps=30 --control.reset_time_s=5 --control.warmup_time_s=3 --control.num_image_writer_processes=4 --control.display_data=false --robot.max_relative_target=null
NOTE that --robot.max_relative_target=null will make robot move very fast
left arrow re-record the current episode
right arrow early exit and go to resetting the environment
don't close the robot gripper finger too fast


### then subsample the data
~/Desktop/lerobot/lerobot/scripts/subsample_dataset.py

### then add goal to the data 
python upgrade_dataset.py --source_repo_id pap_three_subsampled_0520_v2 --target_repo_id pap_three_subsampled_0520_v2_with_goal --visualize_goal_indices --new_features next_event_idx --close_threshold 30 52 52 --open_threshold 70 70 70

### then convert the data to articubot format
see scripts/convert.sh

### then convert the data to pi0 format
see external/openpi-mimicgen/examples/aloha_ours/convert_aloha_data_to_pi_lerobot_joint.py