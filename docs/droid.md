# Droid Setup Guide

Droid is a Franka Panda + Robotiq robot with a wrist-mounted ZED camera. In this repository, teleop is done using the `GELLO` and control with `deoxys`.

## Architecture

- **Franka Panda** (follower arm): controlled via deoxys C++ controller running on a NUC
- **GELLO** (leader arm): 7-DOF Dynamixel arm read over USB serial
- **Robotiq gripper**: controlled via Modbus RTU over USB serial
- **Communication**: Python on workstation sends joint targets over ZMQ to the NUC, which runs the real-time control loop at 1000 Hz via libfranka

## Installation

### 1. Install GELLO

Installing deoxys long time as it builds the C++ franka-interface from source.

```bash
pixi run install-gello
pixi run install-deoxys
pixi run install-zed
```

## NUC Setup

The NUC runs the deoxys C++ real-time controller that communicates directly with the Franka via libfranka. In the deoxys repository on the NUC, the deoxys C++ binary reads safety torque limits from `config/control_config.yml` on the NUC. The default ships with:

```yaml
CONTROL:
        SAFETY:
                MAX_TORQUE: 5
                MIN_TORQUE: -5
```

**This 5 Nm limit is far too low for any manipulation task.** The arm will be unable to lift objects and will feel sluggish regardless of impedance gain settings. For teleoperation and manipulation, increase to:

```yaml
CONTROL:
        SAFETY:
                MAX_TORQUE: 50
                MIN_TORQUE: -50
                MAX_TRANS_SPEED: 0.1
                MIN_TRANS_SPEED: -0.1
                MAX_ROT_SPEED: 0.1
                MIN_ROT_SPEED: -0.1
```

Restart `auto_arm.sh` after changing this file.

## GELLO Calibration

### Finding joint offsets

The GELLO Dynamixel servos report raw encoder positions that need to be mapped to Franka joint angles via offsets and signs. To find the correct offsets:

1. Physically place the GELLO arm in the same pose as the Franka (e.g., both at the home position)
2. Run the upstream offset detection script:

```bash
python third_party/gello/scripts/gello_get_offset.py --port /dev/serial/by-id/YOUR_DEVICE --start-joints 0 0 0 0 0 0 0 --joint-signs 1 1 1 1 1 -1 1 --gripper
```

3. The script outputs offsets as multiples of pi/2. Update `gello_joint_offsets` and `gello_joint_signs` in `DroidRobotConfig` in `lerobot/common/robot_devices/robots/configs.py`.

### Verifying calibration

Use the joint reading script to verify that GELLO and Franka joint values match:

```bash
python docs/read_joints.py
```

This prints both GELLO and Franka joint positions in a loop along with the difference. When the arms are in the same physical pose, the diff should be close to zero across all joints. If a joint has a large offset, adjust `gello_joint_offsets` for that joint. If a joint moves in the wrong direction, flip the sign in `gello_joint_signs`.

## Running Teleoperation

```bash
python -m lerobot.scripts.control_robot \
    --robot.type=droid \
    --control.type=teleoperate
```

On startup, the robot will:
1. Connect to the Franka via deoxys
2. Connect to the GELLO
3. Auto-detect and activate the Robotiq gripper
4. Run calibration (move Franka to match GELLO pose)
5. Begin teleoperation

## Recording Episodes

Record with two fixed Azure Kinects (hardware-synced master/subordinate) and the wrist-mounted ZED:

```bash
python lerobot/scripts/control_robot.py --robot.type=droid --control.type=record \
    --control.single_task="Move the red mug." \
    --control.repo_id=<your_hf_user>/move_red_mug_droid \
    --control.num_episodes=20 \
    --robot.cameras='{"cam_azure_kinect_left": {"type": "azurekinect", "device_id": 1, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true, "wired_sync_mode": "master"}, "cam_azure_kinect_front": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, "height": 720, "use_transformed_depth": true, "wired_sync_mode": "subordinate", "subordinate_delay_off_master_usec": 200}, "cam_wrist": {"type": "zed", "serial_number": "10296178", "fps": 30, "width": 1280, "height": 720, "use_depth": false}}' \
    --robot.use_eef=true --control.push_to_hub=true \
    --control.fps=30 --control.reset_time_s=5 --control.warmup_time_s=3 \
    --control.num_image_writer_processes=4 --control.display_data=true
```

Pass the same `--robot.cameras` to `--control.type=teleoperate` to preview the camera streams.

## Training & Evaluation

Train and roll out as usual, but set `--policy.robot_type=droid`. See [training.md](training.md) for the full set of training/eval commands.
