# Preparing Human Demonstrations (GHOST)

Pipeline for turning human (non-robot) demonstration videos into training data, used in [GHOST](https://ghost-human-demo.github.io/). Human demos are processed either to keep the human in the frame (`--humanize`) or to retarget the hand to a robot end-effector via [Phantom](https://phantom-human-videos.github.io/) (`--phantomize`).

## 1. Collect human demos

Record human demos the same way as robot demos, but **at 15 fps** (the human moves faster than teleop):

```bash
python lerobot/scripts/control_robot.py --robot.type=aloha --control.type=record \
    --control.single_task="Fold the onesie." \
    --control.repo_id=<your_hf_user>/fold_onesie_human \
    --control.num_episodes=25 \
    --robot.cameras='{"cam_azure_kinect_back": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true, "wired_sync_mode": "master"}, "cam_azure_kinect_front": {"type": "azurekinect", "device_id": 1, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true, "wired_sync_mode": "subordinate", "subordinate_delay_off_master_usec": 200}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' \
    --robot.use_eef=true --control.push_to_hub=true \
    --control.fps=15 --control.reset_time_s=5 --control.warmup_time_s=3 \
    --control.num_image_writer_processes=4 --control.display_data=true
```

## 2. Hand pose estimation

Process the recorded videos with [WiLoR](https://github.com/sriramsk1999/wilor):

```bash
python demo_lerobot_detectron2.py \
    --input_folder "/home/sriram/.cache/huggingface/lerobot/<repo_id>/videos/chunk-000/" \
    --output_folder "/data/sriram/lerobot_extradata/<repo_id>/wilor_hand_pose"
```

## 3. Annotate events

Manually annotate ground-truth events with `annotate_events.py` (see [dataset_utilities.md](dataset_utilities.md)):

```bash
python lerobot/scripts/annotate_events.py <video_dir> "fold the onesie"
```

## 4. Upgrade dataset

Run `upgrade_dataset.py` with either `--humanize` or `--phantomize`.

### Humanize (keep human in video)

```bash
python lerobot/scripts/upgrade_dataset.py \
    --source_repo_id <source_id> \
    --target_repo_id <target_id> \
    --humanize \
    --new_features goal_gripper_proj gripper_pcds next_event_idx
```

### Phantomize (retarget human to robot)

This needs three external steps before the upgrade:

a. Generate masks with [GSAM-2](https://github.com/sriramsk1999/Grounded-SAM-2):

```bash
python gsam2_lerobot.py <repo_id> <cam_name>
```

b. Inpaint with [E2FGVI](https://github.com/MCG-NKU/E2FGVI) using the GSAM-2 masks.

c. Generate Phantom videos (from `ghost`):

```bash
python run_phantom_lerobot.py \
    --calib_json <calibration_json> \
    --lerobot-extradata-path /data/sriram/lerobot_extradata/<repo_id>
```

d. Create the phantomized dataset:

```bash
python lerobot/scripts/upgrade_dataset.py \
    --source_repo_id <source_id> \
    --target_repo_id <target_id> \
    --phantomize \
    --path_to_extradata /data/sriram/lerobot_extradata/ \
    --new_features goal_gripper_proj gripper_pcds next_event_idx \
    --extrinsics_txt <path_to_extrinsics>
```

## Training on phantomized data

When training a policy that consumes phantomized images, pass `--policy.phantomize=true`. During evaluation, set `phantomize=False` for in-distribution evals and `phantomize=True` for out-of-distribution (human-task) evals. See [training.md](training.md) for full commands.
