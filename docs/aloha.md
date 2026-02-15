# Aloha Setup & Usage

Aloha-specific notes for the r-pad LeRobot fork with Azure Kinect + Aloha hardware.

## Teleoperation

Base teleop command:

```bash
python lerobot/scripts/control_robot.py \
    --robot.type=aloha \
    --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}}' \
    --control.type=teleoperate \
    --control.display_data=true
```

Camera options are configured via the JSON config above. See `lerobot/common/robot_devices/cameras/azure_kinect.py` for all available options.

**`--robot.max_relative_target`**: Limits how much the robot can move per step (default: 5). Lower values (<3) cause laggy, jerky motion. Setting to `null` disables clamping for smooth high-frequency teleop, but is risky — one wrong move can damage the hardware. The default works well enough.

**`--control.display_data=true`**: Opens a streaming Rerun visualization window.

## Recording Episodes

Base record command:

```bash
python lerobot/scripts/control_robot.py \
    --robot.type=aloha \
    --control.type=record \
    --control.single_task="Grasp mug and place it on the table." \
    --control.repo_id=sriramsk/aloha_mug_eef_depth \
    --control.num_episodes=100 \
    --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}}' \
    --robot.use_eef=true \
    --control.push_to_hub=true \
    --control.fps=60 \
    --control.reset_time_s=5 \
    --control.warmup_time_s=3 \
    --control.resume=true \
    --control.num_image_writer_processes=4
```

With Azure Kinect + Intel RealSense wrist camera:

```bash
python lerobot/scripts/control_robot.py \
    --robot.type=aloha \
    --control.type=record \
    --control.single_task="Grasp mug and place it on the table." \
    --control.repo_id=sriramsk/testing_realsense \
    --control.num_episodes=100 \
    --robot.cameras='{"cam_azure_kinect": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true}, "cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' \
    --robot.use_eef=true \
    --control.push_to_hub=true \
    --control.fps=60 \
    --control.reset_time_s=5 \
    --control.warmup_time_s=3 \
    --control.resume=true \
    --control.num_image_writer_processes=4
```

Key flags:
- `--control.repo_id` — dataset name and HuggingFace upload repo (if `--control.push_to_hub` is enabled)
- `--control.resume` — write to an existing dataset
- `--robot.use_eef=true` — runs forward kinematics and stores computed EEF pose
- Left/right arrow keys: finish / reset current episode

## Visualize and Replay

Visualize:

```bash
python lerobot/scripts/visualize_dataset.py \
    --repo-id sriramsk/aloha_mug_eef \
    --episode-index 0
```

Replay (be careful!):

```bash
python lerobot/scripts/control_robot.py \
    --robot.type=aloha \
    --robot.cameras='{"cam_kinect": {"type": "opencv", "camera_index": 0, "fps": 30, "width": 1280, "height": 720}}' \
    --control.type=replay \
    --control.fps=60 \
    --control.repo_id=sriramsk/aloha_mug_eef \
    --control.episode=0
```

## Misc: Real-to-Sim Joint Mapping

Mapping between LeRobot Aloha joint indices and `robot_descriptions` Aloha (`mink` configuration):

**`mink` configuration.q joint order:**
```
left/waist, left/shoulder, left/elbow, left/forearm_roll, left/wrist_angle, left/wrist_rotate, left/left_finger, left/right_finger,
right/waist, right/shoulder, right/elbow, right/forearm_roll, right/wrist_angle, right/wrist_rotate, right/left_finger, right/right_finger
```

**Real-to-sim index mapping:**
```
Real → Sim
0  → 0     8  → 9
1  → 1     9  → 10
2  → 3     10 → 12
3  → 5     11 → 14
4  → 6     12 → 15
5  → 7     13 → 16
6  → 8     14 → 17
7  → 8     15 → 17
```

**Real-to-sim value mapping** (sign and offset, for IK):
```python
signs   = [-1, -1, 1, 1, 1, 1, ...]
offsets = [pi/2, pi/2, -pi/2, 0, 0, 0, ...]
```

Shoulder joint (indices 1 and 9) must be handled separately: set sign to 1 and offset to 0.

### Sim viewer for joint comparison

```python
import mujoco.viewer
import mujoco
from robot_descriptions.loaders.mujoco import load_robot_description

model = load_robot_description("aloha_mj_description")
data = mujoco.MjData(model)
viewer = mujoco.viewer.launch(model, data)
```

## Preparing Human Demonstrations (Detailed)

After collecting human demos:

1. **Hand pose estimation** with [WiLoR](https://github.com/sriramsk1999/wilor):
   ```bash
   python demo_lerobot_detectron2.py \
       --input_folder "/home/sriram/.cache/huggingface/lerobot/<repo_id>/videos/chunk-000/" \
       --output_folder "/data/sriram/lerobot_extradata/<repo_id>/wilor_hand_pose"
   ```

2. **Annotate events** (manually) using `lerobot/scripts/annotate_events.py`

3. **Upgrade dataset** — either humanize or phantomize:

   **Humanize** (keep human in video):
   ```bash
   python upgrade_dataset.py \
       --source_repo_id <source_id> \
       --target_repo_id <target_id> \
       --humanize \
       --new_features goal_gripper_proj gripper_pcds next_event_idx
   ```

   **Phantomize** (retarget human to robot):

   a. Generate masks with [GSAM-2](https://github.com/sriramsk1999/Grounded-SAM-2):
   ```bash
   python gsam2_lerobot.py <repo_id> <cam_name>
   ```

   b. Inpaint with [E2FGVI](https://github.com/MCG-NKU/E2FGVI) using the GSAM-2 masks

   c. Generate Phantom videos from lfd3d:
   ```bash
   python run_phantom_lerobot.py \
       --calib_json <calibration_json> \
       --lerobot-extradata-path /data/sriram/lerobot_extradata/<repo_id>
   ```

   d. Create the phantomized dataset:
   ```bash
   python upgrade_dataset.py \
       --source_repo_id <source_id> \
       --target_repo_id <target_id> \
       --phantomize \
       --path_to_extradata /data/sriram/lerobot_extradata/ \
       --new_features goal_gripper_proj gripper_pcds next_event_idx \
       --extrinsics_txt <path_to_extrinsics>
   ```

## Calibration Playbook

### Prerequisites

Set up [lfd3d-system](https://github.com/r-pad/lfd3d-system). In separate terminals:

```bash
pixi run ros2 launch aloha aloha_bringup.launch.py robot:=aloha_stationary use_cameras:=false
pixi run python ros/src/aloha/scripts/teleop.py -r aloha_stationary
pixi run ros2 launch azure_kinect_ros_driver driver.launch.py overwrite_robot_description:=False color_resolution:=720P fps:=30 depth_mode:=NFOV_UNBINNED point_cloud:=True rgb_point_cloud:=True
pixi run ros2 run rviz2 rviz2 -d ros/src/lfd3d_ros/launch/rviz/4arms_pcd.rviz
pixi run ros2 launch lfd3d_ros rgbd_pipeline_launch.py
pixi run ros2 run lfd3d_ros broadcast_transform --transform_file captures/<latest_calibration>/T_world_from_camera_est.txt
```

### Running Calibration

1. Verify the current calibration by checking robot point cloud / URDF overlay quality in rviz
2. If not good enough, run:
   ```bash
   pixi run ros2 run lfd3d_ros camera_calibration_collect_ros
   ```
3. Collect ~30 images where the ArUco marker is detected by pressing 'c'
4. In the [calibration repo](https://github.com/r-pad/calibration), run:
   ```bash
   uv run scripts/calibrate.py --output-dir <latest_capture_dir>
   ```
5. Typical good calibration: loss around 0.025 with the default hyperparams and a good overlay in rviz

### After Calibration

- Update `configuration_diffusion.py` and potentially `config.json` in trained checkpoints
- The ArUco marker must be mounted on the corresponding link for which the pose is recorded

### Clean Shutdown

1. Kill the terminal running `teleop.py`
2. Run:
   ```bash
   pixi run python ros/src/aloha/scripts/sleep.py -r aloha_stationary
   ```
3. After `sleep.py` finishes, kill all other terminals
4. Rename the latest capture:
   ```bash
   mv captures/<output_dir> captures/camera_<cam-name>_<date>
   ```
5. Copy the latest `T_world_from_camera_est.txt` into `lerobot/lerobot/scripts/aloha_calibration/`
6. Update the corresponding `calibration.json`
