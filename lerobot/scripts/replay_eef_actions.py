"""
Replay one or more pre-recorded sequences of absolute EEF actions on the real Franka Panda.

Each .npy file is a (T, 10) action sequence in polaris convention:
    [trans(3) | rot6d(6) | gripper(1)]
      - trans: xyz in robot base frame (metres)
      - rot6d: first two columns of the rotation matrix (6D representation)
      - gripper: 1.0 = closed, 0.0 = open  (polaris binary convention)

Per demo, the script:
  1. Loads the action sequence (shape (T, 10)).
  2. Opens the gripper, computes IK for the first action, prints max joint delta,
     and prompts before smoothly interpolating the Franka to the start pose.
  3. Streams actions at --fps Hz: polaris -> lerobot conversion -> IK -> send_action.
  4. Optionally records observations + actions into a LeRobot dataset (--repo_id),
     one episode per demo file.

By default, between consecutive demos the script just smooth-moves the Franka
directly to the next demo's first action (after the per-demo Enter prompt).
Pass --gello_reset to instead run gello_reset_loop() between demos: aligns the
Franka to the GELLO's current pose and lets you teleoperate manually until Enter
is pressed.

Optional rerun visualization (--display_data) opens a viewer with live camera feeds,
per-joint action scalars, and the commanded EEF coordinate frame (red/green/blue = X/Y/Z).
Auto-disabled in headless environments.

Usage (single file):
    pixi run python lerobot/scripts/replay_eef_actions.py \\
        --actions path/to/demo_0.npy \\
        --fps 30 --repo_id xiaochyVera/my_dataset

Usage (whole directory, demos replayed in numerical order):
    pixi run python lerobot/scripts/replay_eef_actions.py \\
        --actions_dir path/to/pick_red_mug/ \\
        --fps 30 --repo_id xiaochyVera/my_dataset \\
        [--start_demo 5]    # resume from demo_5 if interrupted
        [--display_data]    # open rerun viewer
        [--push_to_hub]     # push the dataset to HF Hub when done
        [--undo_z_rotation_deg 45.0]  # undo training-time -45 deg Z augmentation
"""

import argparse
import pathlib
import re
import threading
import time

import numpy as np
import pytorch3d.transforms as transforms
import rerun as rr
import torch

from lerobot.common.policies.robot_adapters import DroidAdapter
from lerobot.common.robot_devices.cameras.configs import AzureKinectCameraConfig, ZedCameraConfig
from lerobot.common.robot_devices.control_utils import is_headless
from lerobot.common.robot_devices.robots.configs import DroidRobotConfig
from lerobot.common.robot_devices.robots.droid import DroidRobot

CAMERAS = {
    "cam_azure_kinect_left": AzureKinectCameraConfig(
        device_id=1, fps=30, width=1280, height=720,
        use_transformed_depth=False, wired_sync_mode="master",
    ),
    "cam_azure_kinect_front": AzureKinectCameraConfig(
        device_id=0, fps=30, width=1280, height=720,
        use_transformed_depth=False, wired_sync_mode="subordinate",
        subordinate_delay_off_master_usec=200,
    ),
    "cam_wrist": ZedCameraConfig(
        serial_number=10296178, fps=30, width=1280, height=720, use_depth=False,
    ),
}


def _demo_index(path: pathlib.Path) -> int:
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else 0


def sorted_demos(directory: str) -> list[pathlib.Path]:
    """List all .npy files in `directory`, sorted by the first integer in the filename.

    Files without a number get index 0 (sort first / tied alphabetically by stem).
    """
    files = list(pathlib.Path(directory).glob("*.npy"))
    return sorted(files, key=lambda p: (_demo_index(p), p.stem))


def polaris_to_lerobot_eef(action_np: np.ndarray, undo_z_rotation_deg: float = 0.0) -> torch.Tensor:
    import pytorch3d.transforms as p3d
    t = torch.from_numpy(action_np).float()
    trans   = t[0:3]
    rot6d   = t[3:9]
    gripper = t[9:10]

    if undo_z_rotation_deg != 0.0:
        a = np.deg2rad(undo_z_rotation_deg)
        R_z = torch.tensor(
            [[np.cos(a), -np.sin(a), 0.0],
             [np.sin(a),  np.cos(a), 0.0],
             [0.0,        0.0,       1.0]], dtype=torch.float32
        )
        R_pred = p3d.rotation_6d_to_matrix(rot6d.unsqueeze(0)).squeeze(0)
        rot6d = p3d.matrix_to_rotation_6d((R_z @ R_pred).unsqueeze(0)).squeeze(0)

    gripper_lerobot = 1.0 - gripper
    return torch.cat([rot6d, trans, gripper_lerobot])


def _build_dataset(repo_id: str, fps: float, robot: DroidRobot, task: str):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    obs = robot.capture_observation()
    features = {}
    for key, val in obs.items():
        v = val if isinstance(val, torch.Tensor) else torch.from_numpy(val)
        if key.startswith("observation.images."):
            h, w, c = v.shape
            features[key] = {"dtype": "video", "shape": (h, w, c), "names": ["height", "width", "channel"]}
        else:
            features[key] = {"dtype": "float32", "shape": tuple(v.shape), "names": None}
    features["action"] = features["observation.state"].copy()
    return LeRobotDataset.create(repo_id=repo_id, fps=int(fps), features=features)


def log_rerun_frame(observation: dict, joint_action: torch.Tensor, eef_lerobot: torch.Tensor):
    """Mirror control_loop's rerun logging: scalars + images + EEF coordinate frame."""
    # Per-joint scalar plots
    for i, val in enumerate(joint_action.numpy()):
        rr.log(f"sent_action_{i}", rr.Scalar(float(val)))

    # Camera images
    for key, val in observation.items():
        if "image" in key:
            img = val.numpy() if isinstance(val, torch.Tensor) else val
            rr.log(key, rr.Image(img), static=True)

    # EEF as 3D coordinate frame: rotated basis vectors are the COLUMNS of R,
    # so we transpose to put each axis on a row for rerun's Arrows3D.
    rot6d = eef_lerobot[0:6]
    eef_trans = eef_lerobot[6:9].numpy()
    R = transforms.rotation_6d_to_matrix(rot6d).numpy()
    axes = (R * 0.1).T  # (3, 3): row i = i-th EEF axis in world frame, scaled
    rr.log("eef_frame", rr.Arrows3D(
        origins=[eef_trans] * 3,
        vectors=axes,
        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],  # X=red, Y=green, Z=blue
    ))


def gello_reset_loop(robot: DroidRobot, fps: float = 30.0):
    """Interpolate to current GELLO position, then run GELLO teleoperation until Enter.

    First smoothly moves the robot to match where GELLO currently is (so there's
    no sudden jump when teleoperation begins), then hands control to GELLO.
    Pressing Enter stops the loop so the next demo can begin.
    """
    gello_joints = np.array(robot.gello.get_joint_state())
    gello_target = gello_joints[:7]
    franka_current = robot._get_franka_joints()
    max_delta = float(np.max(np.abs(gello_target - franka_current)))
    print(f"  [GELLO] Interpolating to GELLO pose (max_delta={max_delta:.4f} rad)...")
    robot._smooth_move_to(gello_target)
    print("  [GELLO] Use GELLO to position the robot. Press Enter when ready...")
    ready = threading.Event()

    def _wait_enter():
        input()
        ready.set()

    t = threading.Thread(target=_wait_enter, daemon=True)
    t.start()

    dt = 1.0 / fps
    while not ready.is_set():
        t_start = time.perf_counter()
        robot.teleop_step(record_data=False)
        elapsed = time.perf_counter() - t_start
        remaining = dt - elapsed
        if remaining > 0:
            time.sleep(remaining)


def replay_episode(
    actions_np: np.ndarray,
    robot: DroidRobot,
    adapter: DroidAdapter,
    dataset,
    dt: float,
    undo_z_rotation_deg: float,
    task: str,
    demo_name: str = "",
    display_data: bool = False,
):
    """Replay one episode. Prompts before moving, then streams actions at dt Hz."""
    T = len(actions_np)

    robot.open_gripper()
    time.sleep(0.5)

    # IK the first action and show delta so user can check safety
    first_eef = polaris_to_lerobot_eef(actions_np[0], undo_z_rotation_deg)
    obs = robot.capture_observation()
    current_state = obs["observation.state"]
    if not isinstance(current_state, torch.Tensor):
        current_state = torch.from_numpy(current_state).float()
    first_joints = adapter._eef_to_joints(first_eef, current_state).numpy()
    current_joints = robot._get_franka_joints()
    max_delta = float(np.max(np.abs(first_joints[:7] - current_joints)))
    print(f"  Target joints: {first_joints.round(3)}  max_delta={max_delta:.4f} rad")
    choice = input("  Press Enter to move to start pose, 's' to skip this demo, 'q' to quit: ").strip().lower()
    if choice == "s":
        print(f"  Skipped {demo_name}.")
        return
    if choice == "q":
        print("  Quit requested.")
        raise KeyboardInterrupt

    robot._smooth_move_to(first_joints[:7])
    time.sleep(0.3)

    print(f"  Replaying {T} steps at {1/dt:.0f} Hz...")
    for step_idx, action_np in enumerate(actions_np):
        t_start = time.perf_counter()



        eef_lerobot = polaris_to_lerobot_eef(action_np, undo_z_rotation_deg)

        obs = robot.capture_observation()
        current_state = obs["observation.state"]
        if not isinstance(current_state, torch.Tensor):
            current_state = torch.from_numpy(current_state).float()

        joint_action = adapter._eef_to_joints(eef_lerobot, current_state)

        if step_idx % 30 == 0:
            trans   = eef_lerobot[6:9].numpy().round(4)
            gripper = eef_lerobot[9].item()
            print(f"  [{step_idx:04d}/{T}] trans={trans}  gripper={gripper:.2f}")

        robot.send_action(joint_action)

        if display_data:
            log_rerun_frame(obs, joint_action, eef_lerobot)

        if dataset is not None:
            dataset.add_frame({**obs, "action": joint_action, "task": task})

        elapsed = time.perf_counter() - t_start
        remaining = dt - elapsed
        if remaining > 0:
            time.sleep(remaining)

    if dataset is not None:
        dataset.save_episode()
        print(f"  Episode saved ({demo_name})")


def main():
    parser = argparse.ArgumentParser(description="Replay EEF action sequence(s) on Franka Panda")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--actions", type=str,
                       help="Single .npy file (T, 10) in polaris format, or a directory "
                            "of .npy files (auto-detected; same as --actions_dir)")
    group.add_argument("--actions_dir", type=str,
                       help="Directory of *.npy action files — replayed in numerical order "
                            "(sorted by the first integer in each filename)")
    parser.add_argument("--start_demo", type=int, default=0,
                        help="Skip demos with index < start_demo (for resuming; default: 0)")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--dry_run", action="store_true",
                        help="Print file list and first action without connecting to robot")
    parser.add_argument("--skip_gello_calibration", action="store_true")
    parser.add_argument("--repo_id", type=str, default=None)
    parser.add_argument("--task", type=str, default="replay")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--undo_z_rotation_deg", type=float, default=0.0)
    parser.add_argument("--display_data", action="store_true",
                        help="Open rerun viewer with camera feeds, action scalars, and EEF frame")
    parser.add_argument("--no_wrist_cam", action="store_true",
                        help="Skip the ZED wrist camera (use when it's unplugged or not detected)")
    parser.add_argument("--gello_reset", action="store_true",
                        help="Between demos, run GELLO teleop reset loop. "
                             "Default: skip it and smooth-move directly to the next demo's start pose.")
    args = parser.parse_args()

    # Build ordered file list. --actions accepts either a single .npy file or a
    # directory of .npy files (auto-detected); --actions_dir is the explicit
    # directory form.
    actions_dir = args.actions_dir
    if actions_dir is None and args.actions and pathlib.Path(args.actions).is_dir():
        actions_dir = args.actions

    if actions_dir:
        demo_files = sorted_demos(actions_dir)
        if not demo_files:
            raise FileNotFoundError(f"No .npy files found in {actions_dir}")
        demo_files = [f for f in demo_files if _demo_index(f) >= args.start_demo]
        print(f"Found {len(demo_files)} demos in {actions_dir} "
              f"(starting from demo_{args.start_demo})")
    else:
        demo_files = [pathlib.Path(args.actions)]

    if args.dry_run:
        for f in demo_files[:5]:
            arr = np.load(f)
            print(f"  {f.name}: shape={arr.shape}  first={arr[0].round(4)}")
        if len(demo_files) > 5:
            print(f"  ... and {len(demo_files) - 5} more")
        return

    # Connect once for all demos
    cameras = {k: v for k, v in CAMERAS.items() if not (args.no_wrist_cam and k == "cam_wrist")}
    robot_cfg = DroidRobotConfig(
        use_eef=True,
        skip_gello_calibration=args.skip_gello_calibration,
        cameras=cameras,
    )
    robot = DroidRobot(robot_cfg)
    robot.connect()
    adapter = DroidAdapter(action_space="right_eef")
    dt = 1.0 / args.fps

    display_data = args.display_data and not is_headless()
    if display_data:
        rr.init("replay_eef_actions", spawn=True)

    dataset = None
    if args.repo_id is not None:
        print(f"Creating dataset: {args.repo_id}")
        dataset = _build_dataset(args.repo_id, args.fps, robot, args.task)
        print(f"Dataset features: {list(dataset.features.keys())}")
        dataset.start_image_writer(num_processes=0, num_threads=4 * len(robot.cameras))

    try:
        for ep_num, demo_path in enumerate(demo_files):
            actions_np = np.load(demo_path)
            assert actions_np.ndim == 2 and actions_np.shape[1] == 10, (
                f"Expected (T, 10), got {actions_np.shape} in {demo_path}"
            )
            print(f"\n[{ep_num + 1}/{len(demo_files)}] {demo_path.name}  ({len(actions_np)} steps)")

            if ep_num > 0 and args.gello_reset:
                gello_reset_loop(robot, fps=args.fps)

            replay_episode(
                actions_np=actions_np,
                robot=robot,
                adapter=adapter,
                dataset=dataset,
                dt=dt,
                undo_z_rotation_deg=args.undo_z_rotation_deg,
                task=args.task,
                demo_name=demo_path.name,
                display_data=display_data,
            )

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if dataset is not None and args.push_to_hub:
            dataset.push_to_hub(repo_id=args.repo_id)
            print(f"Pushed to HuggingFace: {args.repo_id}")
        print("Done.")
        robot.disconnect()


if __name__ == "__main__":
    main()
