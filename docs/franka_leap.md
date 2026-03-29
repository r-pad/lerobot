# Franka + LEAP Hand Setup Guide

Franka + LEAP hand is a Franka Panda arm with a LEAP dexterous hand (16-DOF). Arm teleop is done with a GELLO, and hand teleop uses a Manus glove with a [GeoRT](https://github.com/YingYuan0414/GeoRT/) retargeting model. Control is through deoxys.

## Architecture

- **Franka Panda** (follower arm): controlled via deoxys C++ controller running on a NUC
- **GELLO** (leader arm): 7-DOF Dynamixel arm read over USB serial
- **LEAP hand** (16-DOF dexterous hand): controlled via bundled LeapNode (modified from [LEAP Hand API](https://github.com/leap-hand/LEAP_Hand_API))
- **Manus glove**: streams hand pose data over ZMQ
- **GeoRT model**: retargets Manus glove data to LEAP hand joint angles (allegro convention). Inference code vendored from [GeoRT](https://github.com/YingYuan0414/GeoRT/); training happens in the upstream repo
- **Communication**: Python on workstation sends joint targets over ZMQ to the NUC, which runs the real-time control loop at 1000 Hz via libfranka

## Installation

```bash
pixi run install-gello
pixi run install-deoxys
```

The LEAP hand control code and GeoRT inference code are bundled in this repo — no additional packages to install for those. You do need `dynamixel-sdk` (for the LEAP hand motors) and `pyzmq` (for Manus glove), both of which should already be available.

## Network Configuration (required for each new robot)

The deoxys config file specifies the IP addresses of the workstation (PC), NUC, and Franka robot. **You must update this for your specific network setup.**

Edit `lerobot/common/robot_devices/robots/franka_configs/charmander_leap.yml`:

```yaml
PC:
  NAME: "your-workstation-hostname"
  IP: <workstation IP on the robot network>

NUC:
  NAME: "your-nuc-hostname"
  IP: <NUC IP on the robot network>
  PUB_PORT: 5556
  SUB_PORT: 5555
  GRIPPER_PUB_PORT: 5558
  GRIPPER_SUB_PORT: 5557

ROBOT:
  IP: <Franka robot IP>
```

- **PC**: the workstation running the Python teleop/policy code
- **NUC**: the Intel NUC running the deoxys C++ real-time controller
- **ROBOT**: the Franka Panda's network interface

All three must be on the same subnet. If you are setting up a new robot or moving to a different network, update these IPs before running anything.

You can also point to a different config file entirely by setting `deoxys_general_cfg_file` in `FrankaLeapRobotConfig`.

## NUC Setup

The NUC runs the deoxys C++ real-time controller that communicates directly with the Franka via libfranka. In the deoxys repository on the NUC, the deoxys C++ binary reads safety torque limits from `config/control_config.yml` on the NUC. The default ships with:

```yaml
CONTROL:
        SAFETY:
                MAX_TORQUE: 5
                MIN_TORQUE: -5
```

**This 5 Nm limit is far too low for any manipulation task.** For teleoperation and manipulation, increase to:

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

3. The script outputs offsets as multiples of pi/2. Update `gello_joint_offsets` and `gello_joint_signs` in `FrankaLeapRobotConfig` in `lerobot/common/robot_devices/robots/configs.py`.

### Verifying calibration

Use the joint reading script to verify that GELLO and Franka joint values match:

```bash
python docs/read_joints.py --config lerobot/common/robot_devices/robots/franka_configs/charmander_leap.yml
```

## GeoRT Model Setup

The [GeoRT](https://github.com/YingYuan0414/GeoRT/) retargeting model maps Manus glove finger poses to LEAP hand joint angles. The inference code is vendored in this repo; training happens in the upstream GeoRT repository.

Configure the path to your trained checkpoints in `FrankaLeapRobotConfig`:

- `geort_checkpoint_root`: Path to the GeoRT `checkpoint/` directory (default: `/home/leap/Desktop/GeoRT/checkpoint`)
- `geort_ckpt_tag`: Substring to match against checkpoint folder names (default: `"sriram_1"`)

The checkpoint directory should contain a folder matching the tag, with `last.pth` and `config.json` inside it.

Make sure the Manus glove ZMQ server is running before starting teleop.

## Running Teleoperation

```bash
python -m lerobot.scripts.control_robot \
    --robot.type=franka_leap \
    --control.type=teleoperate
```

On startup, the robot will:
1. Connect to the Franka via deoxys
2. Connect to the GELLO
3. Connect to the LEAP hand (via Dynamixel on `/dev/ttyUSB0` or `/dev/ttyUSB1`)
4. Load the GeoRT model and connect to the Manus glove (ZMQ on `localhost:8000`)
5. Run calibration (move Franka to match GELLO pose)
6. Begin teleoperation (arm via GELLO, hand via Manus glove)

## State and Action Space

| Component | DOF | Details |
|-----------|-----|---------|
| Franka arm | 7 | Joint positions (rad) |
| LEAP hand | 16 | Allegro convention (4 joints x 4 fingers: index, middle, ring, thumb) |
| **Total** | **23** | `observation.state` and `action` |

When `use_eef` is enabled (default), the following are also recorded:

| Feature | Dim | Contents |
|---------|-----|----------|
| `observation.right_eef_pose` | 25 | 6D rotation + 3D translation + 16 hand joints |
| `action.right_eef_pose` | 25 | Same layout as observation |

## Vendored Dependencies

The following are bundled in this repo to keep it self-contained:

- **LeapNode** (`lerobot/common/robot_devices/robots/leap_hand/`): LEAP hand Dynamixel motor control. Modified from [LEAP Hand API](https://github.com/leap-hand/LEAP_Hand_API).
- **GeoRT inference** (`lerobot/common/robot_devices/robots/geort/`): IK retargeting model + Manus glove ZMQ client. Vendored from [GeoRT (forked)](https://github.com/YingYuan0414/GeoRT/) (Meta Platforms, Inc.).