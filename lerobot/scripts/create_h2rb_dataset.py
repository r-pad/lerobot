"""
Convert collected_data episodes into a LeRobotDataset.

Dataset structure expected:
    collected_data/
        episode_000000/
            trajectory.npz       # keys: states_ee, states_joint, action_ee,
                                 #        action_joint, gripper_pcd,
                                 #        goal_gripper_pcd, gripper_width,
                                 #        delta_action
            cam0.mp4             # front RGB   -> observation.images.cam_azure_kinect_front.color
            cam1.mp4             # left  RGB   -> observation.images.cam_azure_kinect_back.color
            wrist_cam.mp4        # wrist RGB   -> observation.images.cam_wrist.color
            cam0_depth.mkv       # front depth -> observation.images.cam_azure_kinect_front.transformed_depth
            cam1_depth.mkv       # left  depth -> observation.images.cam_azure_kinect_back.transformed_depth
            wrist_cam_depth.mkv  # wrist depth -> observation.images.cam_wrist.transformed_depth

Usage:
    python create_polaris_dataset.py \
        --data_dir /home/haotian/polaris/collected_data \
        --repo_id  haotian/polaris_lerobot \
        --task     "pick up the red cup and place it in the target location"
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from pathlib import Path

import json
import av
from PIL import Image
import imageio.v3 as iio
import numpy as np
from tqdm import tqdm
from lerobot.scripts.dataset_utils import generate_heatmap_from_points, project_points_to_image

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


# ── helpers ───────────────────────────────────────────────────────────────────

def load_video_frames(video_path: str) -> np.ndarray:
    """Read all RGB frames → (T, H, W, 3) uint8."""
    frames = iio.imread(video_path, plugin="pyav")
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise RuntimeError(f"Unexpected frame shape from {video_path}: {frames.shape}")
    return frames


def read_depth_video(video_path: str) -> np.ndarray:
    """
    Read 16-bit depth .mkv → (T, H, W) float32 in metres.
    Mirrors reference script: gray16le / 1000.0
    """
    container = av.open(video_path)
    video_stream = container.streams.video[0]
    loaded_frames = []
    for frame in container.decode(video_stream):
        if frame.format.name in ["gray16le", "gray16be"]:
            frame_array = frame.to_ndarray(format="gray16le") / 1000.0   # mm → m
        else:
            raise NotImplementedError(f"Unsupported depth format: {frame.format.name}")
        loaded_frames.append(frame_array)
    container.close()
    return np.stack(loaded_frames)   # (T, H, W) float32


def get_video_fps(video_path: str) -> float:
    meta = iio.immeta(video_path, plugin="pyav")
    fps = meta.get("fps", 30.0)
    return float(fps) if fps else 30.0


def resize_frames(frames: np.ndarray, H: int, W: int) -> np.ndarray:
    from PIL import Image
    return np.stack([
        np.array(Image.fromarray(f).resize((W, H), Image.BILINEAR))
        for f in frames
    ])


def resize_depth_frames(frames: np.ndarray, H: int, W: int) -> np.ndarray:
    """Resize (T, H, W) depth frames with nearest-neighbour."""
    from PIL import Image
    return np.stack([
        np.array(Image.fromarray(f).resize((W, H), Image.NEAREST))
        for f in frames
    ])


def load_calibrations(calibration_config_path: str):
    """Load intrinsics and extrinsics from a calibration JSON config.

    JSON format:
    {
        "cam_azure_kinect_front": {
            "intrinsics": "path/to/intrinsics.txt",   # 3x3
            "extrinsics": "path/to/extrinsics.txt"    # 4x4 T_world_from_camera
        },
        ...
    }
    Returns dict: {cam_name: {"K": (3,3), "T_world_cam": (4,4), "world_to_cam": (4,4)}}
    """
    with open(calibration_config_path, "r") as f:
        config = json.load(f)

    calibrations = {}
    for cam_name, cam_cfg in config.items():
        K          = np.loadtxt(cam_cfg["intrinsics"])           # (3, 3)
        T_world_cam = np.loadtxt(cam_cfg["extrinsics"])          # (4, 4)
        world_to_cam = np.linalg.inv(T_world_cam)
        calibrations[cam_name] = {
            "K":           K,
            "T_world_cam": T_world_cam,
            "world_to_cam": world_to_cam,
        }
    print(f"Loaded calibrations for: {list(calibrations.keys())}")
    return calibrations


def generate_goal_gripper_proj(gripper_pcd_4x3: np.ndarray, K: np.ndarray,
                                world_to_cam: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Project a (4, 3) gripper PCD goal into a heatmap image (H, W, 3) uint8.
    Mirrors the DROID reference script using generate_heatmap_from_points
    and project_points_to_image from lerobot.scripts.dataset_utils.
    """
    # Transform from world to camera frame
    pcd_hom = np.concatenate([gripper_pcd_4x3,
                               np.ones((gripper_pcd_4x3.shape[0], 1))], axis=-1)  # (4, 4)
    pcd_cam = (world_to_cam @ pcd_hom.T).T[:, :3]                                 # (4, 3)

    urdf_proj = project_points_to_image(pcd_cam, K)                               # (4, 2)
    heatmap   = generate_heatmap_from_points(urdf_proj, (H, W))                   # (H, W, 3)
    return heatmap


# ── main ──────────────────────────────────────────────────────────────────────

def gen_h2rd_dataset(
    data_dir: str,
    repo_id: str,
    task: str,
    calibrations: dict,
    img_shape: tuple = (720, 1280),   # (H, W)
    num_episodes: str = "all",
    target_fps: int = None,
) -> LeRobotDataset:

    episode_dirs = sorted(glob(os.path.join(data_dir, "episode_*")))
    if not episode_dirs:
        raise ValueError(f"No episode_* folders found in {data_dir}")

    # ── infer shapes from first episode ──────────────────────────────────
    sample_dir = episode_dirs[0]
    sample_npz = np.load(os.path.join(sample_dir, "trajectory.npz"))

    states_ee_dim         = sample_npz["states_ee"].shape[-1]
    states_joint_dim      = sample_npz["states_joint"].shape[-1]
    action_ee_dim         = sample_npz["action_ee"].shape[-1]
    action_joint_dim      = sample_npz["action_joint"].shape[-1]
    gripper_pcd_dim         = sample_npz["gripper_pcd"].shape[1]
    goal_gripper_pcd_dim    = sample_npz["goal_gripper_pcd"].shape[1]

    sample_fps = get_video_fps(os.path.join(sample_dir, "cam0.mp4"))
    sample_vid = load_video_frames(os.path.join(sample_dir, "cam0.mp4"))
    orig_h, orig_w = sample_vid.shape[1], sample_vid.shape[2]
    H, W = img_shape if img_shape is not None else (orig_h, orig_w)

    if target_fps is None:
        target_fps = int(sample_fps)
    subsample_factor = int(sample_fps) // target_fps
    if subsample_factor < 1:
        subsample_factor = 1

    print(f"states_ee={states_ee_dim}  states_joint={states_joint_dim}")
    print(f"action_ee={action_ee_dim}  action_joint={action_joint_dim}")
    print(f"gripper_pcd_dim={gripper_pcd_dim}  goal_gripper_pcd_dim={goal_gripper_pcd_dim}")
    print(f"image=({H},{W})  source_fps={sample_fps}  target_fps={target_fps}  subsample_factor={subsample_factor}")
    print(f"Found {len(episode_dirs)} episodes in {data_dir}")

    # ── feature schema ────────────────────────────────────────────────────
    # Naming convention mirrors reference script:
    #   observation.images.<cam_name>.color
    #   observation.images.<cam_name>.transformed_depth
    features = {
        # ── proprioception ──
        "observation.state": {
            "dtype": "float32",
            "shape": (states_joint_dim,),
            "names": [f"states_joint_{i}" for i in range(states_joint_dim)],
        },
        "observation.right_eef_pose": {
            "dtype": "float32",
            "shape": (states_ee_dim,),
            "names": [f"states_ee_{i}" for i in range(states_ee_dim)],
        }, 
        "observation.points.gripper_pcds": {
            "dtype": "pcd",
            "shape": (gripper_pcd_dim, 3),
            "names": ['N', 'channels'],
        },
        "observation.goal_gripper_pcd": {
            "dtype": "pcd",
            "shape": (goal_gripper_pcd_dim, 3),
            "names": ['N', 'channels'],
        },
        # ── actions ──
        "action": {
            "dtype": "float32",
            "shape": (action_joint_dim,),
            "names": [f"action_joint_{i}" for i in range(action_joint_dim)],
        },
        "action.right_eef_pose": {
            "dtype": "float32",
            "shape": (action_ee_dim,),
            "names": [f"action_ee_{i}" for i in range(action_ee_dim)],
        },
        # ── RGB cameras ──
        "observation.images.cam_azure_kinect_front.color": {
            "dtype": "video",
            "shape": (H, W, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.cam_azure_kinect_left.color": {
            "dtype": "video",
            "shape": (H, W, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.cam_wrist": {
            "dtype": "video",
            "shape": (H, W, 3),
            "names": ["height", "width", "channels"],
        },
        # ── depth — stored as uint16 mm, shape (H, W, 1) ──
        "observation.images.cam_azure_kinect_front.transformed_depth": {
            "dtype": "video",
            "shape": (H, W, 1),
            "names": ["height", "width", "channels"],
        },
        "observation.images.cam_azure_kinect_left.transformed_depth": {
            "dtype": "video",
            "shape": (H, W, 1),
            "names": ["height", "width", "channels"],
        },
        # "observation.images.cam_wrist.transformed_depth": {
        #     "dtype": "video",
        #     "shape": (H, W, 1),
        #     "names": ["height", "width", "channels"],
        # },
        # ── goal gripper projection heatmaps ──
        "observation.images.cam_azure_kinect_front.goal_gripper_proj": {
            "dtype": "video",
            "shape": (H, W, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.cam_azure_kinect_left.goal_gripper_proj": {
            "dtype": "video",
            "shape": (H, W, 3),
            "names": ["height", "width", "channels"],
        },
        "next_event_idx": {
            "dtype": "int32",
            "shape": (1,),
            "names": ["idx"],
        },
        "embodiment": {
            "dtype": "string",
            "shape": (1,),
            "names": ["embodiment"],
            "info": "Name of embodiment",
        },
    }

    # ── create dataset ────────────────────────────────────────────────────
    print(f"Creating LeRobotDataset: {repo_id}")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=target_fps,
        features=features,
    )

    # ── process episodes ──────────────────────────────────────────────────
    skipped = 0
    for ep_dir in tqdm(episode_dirs, desc="Episodes"):
        ep_idx = int(Path(ep_dir).name.split("_")[-1])
        if num_episodes != "all" and ep_idx >= int(num_episodes):
            break

        npz_path         = os.path.join(ep_dir, "trajectory.npz")
        cam0_path        = os.path.join(ep_dir, "cam0.mp4")
        cam1_path        = os.path.join(ep_dir, "cam1.mp4")
        wrist_path       = os.path.join(ep_dir, "wrist_cam.mp4")
        cam0_depth_path  = os.path.join(ep_dir, "cam0_depth.mkv")
        cam1_depth_path  = os.path.join(ep_dir, "cam1_depth.mkv")
        wrist_depth_path = os.path.join(ep_dir, "wrist_cam_depth.mkv")

        missing = [p for p in [npz_path, cam0_path, cam1_path, wrist_path,
                                cam0_depth_path, cam1_depth_path, wrist_depth_path]
                   if not os.path.exists(p)]
        if missing:
            print(f"  Skipping {ep_dir}: missing {missing}")
            skipped += 1
            continue

        try:
            npz = np.load(npz_path)
            states_ee        = npz["states_ee"].astype(np.float32)
            states_joint     = npz["states_joint"].astype(np.float32)
            action_ee        = npz["action_ee"].astype(np.float32)
            action_joint     = npz["action_joint"].astype(np.float32)
            gripper_pcd      = npz["gripper_pcd"].astype(np.float32)         # (T, 4, 3)
            goal_gripper_pcd = npz["goal_gripper_pcd"].astype(np.float32)
            gripper_width    = npz["gripper_width"].astype(np.float32)
            delta_action     = npz["delta_action"].astype(np.float32)

            T = action_ee.shape[0]

            # ── fix missing final subgoal ─────────────────────────────────
            last_event_idx = 0
            for i in range(T):
                if np.allclose(goal_gripper_pcd[i], gripper_pcd[i], atol=1e-6):
                    last_event_idx = i
            for i in range(last_event_idx + 1, T):
                goal_gripper_pcd[i] = gripper_pcd[-1]

            # ── load all 6 videos in parallel ─────────────────────────────
            with ThreadPoolExecutor(max_workers=6) as pool:
                f_cam0   = pool.submit(load_video_frames, cam0_path)
                f_cam1   = pool.submit(load_video_frames, cam1_path)
                f_wrist  = pool.submit(load_video_frames, wrist_path)
                f_d0     = pool.submit(read_depth_video,  cam0_depth_path)
                f_d1     = pool.submit(read_depth_video,  cam1_depth_path)
                f_dw     = pool.submit(read_depth_video,  wrist_depth_path)
                cam0_frames  = f_cam0.result()
                cam1_frames  = f_cam1.result()
                wrist_frames = f_wrist.result()
                cam0_depth   = f_d0.result()
                cam1_depth   = f_d1.result()
                wrist_depth  = f_dw.result()

            # ── align lengths ─────────────────────────────────────────────
            T_min = min(T,
                        len(cam0_frames), len(cam1_frames), len(wrist_frames),
                        len(cam0_depth),  len(cam1_depth),  len(wrist_depth))

            if T_min < T:
                print(f"  Warning: traj T={T} > video frames={T_min}, truncating")

            states_ee        = states_ee[:T_min]
            states_joint     = states_joint[:T_min]
            action_ee        = action_ee[:T_min]
            action_joint     = action_joint[:T_min]
            gripper_pcd      = gripper_pcd[:T_min]
            goal_gripper_pcd = goal_gripper_pcd[:T_min]
            gripper_width    = gripper_width[:T_min]
            delta_action     = delta_action[:T_min]
            cam0_frames      = cam0_frames[:T_min]
            cam1_frames      = cam1_frames[:T_min]
            wrist_frames     = wrist_frames[:T_min]
            cam0_depth       = cam0_depth[:T_min]
            cam1_depth       = cam1_depth[:T_min]
            wrist_depth      = wrist_depth[:T_min]

            # ── subsample to target_fps ───────────────────────────────────
            if subsample_factor > 1:
                idx              = list(range(0, T_min, subsample_factor))
                states_ee        = states_ee[idx]
                states_joint     = states_joint[idx]
                action_ee        = action_ee[idx]
                action_joint     = action_joint[idx]
                gripper_pcd      = gripper_pcd[idx]
                goal_gripper_pcd = goal_gripper_pcd[idx]
                gripper_width    = gripper_width[idx]
                delta_action     = delta_action[idx]
                cam0_frames      = cam0_frames[idx]
                cam1_frames      = cam1_frames[idx]
                wrist_frames     = wrist_frames[idx]
                cam0_depth       = cam0_depth[idx]
                cam1_depth       = cam1_depth[idx]
                wrist_depth      = wrist_depth[idx]
                T_min            = len(idx)

            # ── compute next_event_idx (on final subsampled data) ─────────
            subgoal_frames = [
                i - 1 for i in range(1, T_min)
                if not np.allclose(goal_gripper_pcd[i], goal_gripper_pcd[i - 1], atol=1e-6)
            ]
            if subgoal_frames:
                subgoal_frames.append(T_min - 1)
            else:
                subgoal_frames = [T_min - 1]

            next_event_idx = np.zeros(T_min, dtype=np.int32)
            sg_ptr = 0
            for i in range(T_min):
                while sg_ptr < len(subgoal_frames) - 1 and i > subgoal_frames[sg_ptr]:
                    sg_ptr += 1
                next_event_idx[i] = subgoal_frames[sg_ptr]

            # swap gripper_pcd: top, left, right, middle → left, right, top, middle
            gripper_pcd = gripper_pcd[:, [1, 2, 0, 3]]
            goal_gripper_pcd = goal_gripper_pcd[:, [1, 2, 0, 3]]

            # ── resize if needed ──────────────────────────────────────────
            if (orig_h, orig_w) != (H, W):
                cam0_frames  = resize_frames(cam0_frames,  H, W)
                cam1_frames  = resize_frames(cam1_frames,  H, W)
                wrist_frames = resize_frames(wrist_frames, H, W)
                cam0_depth   = resize_depth_frames(cam0_depth,  H, W)
                cam1_depth   = resize_depth_frames(cam1_depth,  H, W)
                wrist_depth  = resize_depth_frames(wrist_depth, H, W)

            # ── add frames (goal_gripper_proj as placeholder zeros) ───────
            # Mirrors reference script: add dummy values first, then fill
            # goal_gripper_proj via episode_buffer after all frames are added.
            # This means we only compute num_subgoals heatmaps, not T heatmaps.
            cam0_depth_mm  = (cam0_depth  * 1000).astype(np.uint16)[:, :, :, None]
            cam1_depth_mm  = (cam1_depth  * 1000).astype(np.uint16)[:, :, :, None]
            wrist_depth_mm = (wrist_depth * 1000).astype(np.uint16)[:, :, :, None]

            zeros_heatmap = np.zeros((H, W, 3), dtype=np.uint8)
            for t in range(T_min):
                d0 = cam0_depth_mm[t]
                d1 = cam1_depth_mm[t]
                dw = wrist_depth_mm[t]

                dataset.add_frame({
                    "task":                          task,
                    # proprioception
                    "observation.state":             states_joint[t],
                    "observation.right_eef_pose":    states_ee[t],
                    "observation.points.gripper_pcds":       gripper_pcd[t],
                    "observation.goal_gripper_pcd":          goal_gripper_pcd[t],
                    # actions
                    "action":                        action_joint[t],  # joint action
                    "action.right_eef_pose":         action_ee[t],
                    # RGB (uint8, H×W×3)
                    "observation.images.cam_azure_kinect_front.color": cam0_frames[t],
                    "observation.images.cam_azure_kinect_left.color":  cam1_frames[t],
                    "observation.images.cam_wrist":              wrist_frames[t],
                    # depth (uint16 mm, H×W×1)
                    "observation.images.cam_azure_kinect_front.transformed_depth": d0,
                    "observation.images.cam_azure_kinect_left.transformed_depth":  d1,
                    # "observation.images.cam_wrist.transformed_depth":              dw,
                    "next_event_idx": np.array([next_event_idx[t]], dtype=np.int32),
                    "embodiment": "droid",
                    # placeholder — filled below via episode_buffer
                    "observation.images.cam_azure_kinect_front.goal_gripper_proj": zeros_heatmap,
                    "observation.images.cam_azure_kinect_left.goal_gripper_proj":  zeros_heatmap,
                })

            # ── fill goal_gripper_proj via episode_buffer ─────────────────
            # Mirrors _process_episode_goals in reference script.
            # Compute one heatmap per subgoal (not per frame) — fast.
            cam_names = ["cam_azure_kinect_front", "cam_azure_kinect_left"]
            for cam in cam_names:
                goal_key = f"observation.images.{cam}.goal_gripper_proj"
                K          = calibrations[cam]["K"]
                world_to_cam = calibrations[cam]["world_to_cam"]

                # One heatmap per subgoal frame
                goal_images = []
                for sg_idx in subgoal_frames:
                    heatmap = generate_goal_gripper_proj(
                        goal_gripper_pcd[sg_idx], K, world_to_cam, H, W)
                    goal_images.append(Image.fromarray(heatmap).convert("RGB"))

                # Assign to frames: each frame gets goal of its segment
                current_sg = 0
                for i in range(T_min):
                    if current_sg < len(subgoal_frames) - 1 and i > subgoal_frames[current_sg]:
                        current_sg += 1
                    goal_images[current_sg].save(dataset.episode_buffer[goal_key][i])

            dataset.save_episode()

        except Exception as e:
            import traceback
            print(f"  ERROR processing {ep_dir}: {e}")
            traceback.print_exc()
            skipped += 1
            continue

    print(f"\nDone! {len(episode_dirs) - skipped}/{len(episode_dirs)} episodes saved.")
    print(f"Dataset root: {dataset.root}")
    return dataset


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert polaris collected_data to LeRobotDataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_dir",     type=str, required=True,
                        help="Path to folder containing episode_* subdirs")
    parser.add_argument("--repo_id",      type=str, required=True,
                        help="HuggingFace repo id for the output dataset")
    parser.add_argument("--task",         type=str,
                        default="pick up the red cup and place it in the target location",
                        help="Natural language task description")
    parser.add_argument("--img_height",   type=int, default=720,
                        help="Resize image height (0 = keep original)")
    parser.add_argument("--img_width",    type=int, default=1280,
                        help="Resize image width  (0 = keep original)")
    parser.add_argument("--num_episodes", type=str, default="all",
                        help="Max episodes to process ('all' for all)")
    parser.add_argument("--target_fps",   type=int, default=None,
                        help="Target FPS after subsampling (default: keep source FPS)")
    parser.add_argument("--calibration_config", type=str, required=False,
                        default="droid_calibration/calibration_multiview.json",
                        help="Path to calibration JSON config file")
    args = parser.parse_args()

    img_shape = (args.img_height, args.img_width) if args.img_height > 0 else None
    calibrations = load_calibrations(args.calibration_config)

    gen_h2rd_dataset(
        data_dir=args.data_dir,
        repo_id=args.repo_id,
        task=args.task,
        calibrations=calibrations,
        img_shape=img_shape,
        num_episodes=args.num_episodes,
        target_fps=args.target_fps,
    )