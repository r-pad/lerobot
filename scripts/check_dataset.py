#!/usr/bin/env python3
"""
Visualize per-timestep NPZ episodes (output of ``convert_dataset.py``).

Example::

    python3 visualize_datasets.py /data/robogen/droid_process/droid_data_articubot/AUTOLab+2023-07-07+Fri_Jul__7_09:42:23_2023+24400334

In the Open3D window: **N** = next frame, **P** = previous frame, **Space** = next.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

VIEWER_BACKGROUND = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)


def _sorted_npz_paths(traj_dir: Path) -> list[Path]:
    paths = sorted(traj_dir.glob("*.npz"), key=lambda x: int(str(x).split(".")[0].split("/")[-1]))
    return [p for p in paths if p.is_file()]


def load_npz_geometries(npz_path: Path) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    data = np.load(npz_path, allow_pickle=False)

    obj_pts = np.asarray(data["point_cloud"], dtype=np.float64).reshape(-1, 3)
    grip_pts = np.asarray(data["gripper_pcd"], dtype=np.float64).reshape(-1, 3)
    goal_pts = np.asarray(data["goal_gripper_pcd"], dtype=np.float64).reshape(-1, 3)

    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(obj_pts)
    obj_pcd.paint_uniform_color([0.0, 0.0, 1.0])
    if "rgb_values" in data.files:
        rgb = np.asarray(data["rgb_values"], dtype=np.float64).reshape(-1, 3)
        if rgb.shape[0] == obj_pts.shape[0]:
            obj_pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb, 0.0, 1.0))

    grip_pcd = o3d.geometry.PointCloud()
    grip_pcd.points = o3d.utility.Vector3dVector(grip_pts)
    grip_pcd.paint_uniform_color([1.0, 0.0, 0.0])

    goal_pcd = o3d.geometry.PointCloud()
    goal_pcd.points = o3d.utility.Vector3dVector(goal_pts)
    goal_pcd.paint_uniform_color([0.0, 1.0, 0.0])

    return obj_pcd, grip_pcd, goal_pcd


def visualize_npz_trajectory(
    traj_dir: Path,
    *,
    start_index: int = 0,
    point_size: float = 2.0,
) -> None:
    npz_paths = _sorted_npz_paths(traj_dir)
    if not npz_paths:
        raise FileNotFoundError(f"No .npz files under {traj_dir}")

    idx = max(0, min(int(start_index), len(npz_paths) - 1))

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name="NPZ trajectory (N=next, P=prev, Space=next)",
        width=1280,
        height=720,
    )

    obj_pcd, grip_pcd, goal_pcd = load_npz_geometries(npz_paths[idx])
    vis.add_geometry(obj_pcd)
    vis.add_geometry(grip_pcd)
    vis.add_geometry(goal_pcd)

    def _frame_label() -> str:
        return f"NPZ trajectory [{idx + 1}/{len(npz_paths)}] {npz_paths[idx].name}"

    def _update_title() -> None:
        label = _frame_label()
        setter = getattr(vis, "set_window_title", None)
        if callable(setter):
            setter(label)
        else:
            print(label, flush=True)

    _update_title()

    def _swap_frame(new_idx: int) -> None:
        nonlocal idx, obj_pcd, grip_pcd, goal_pcd
        vis.remove_geometry(obj_pcd, reset_bounding_box=False)
        vis.remove_geometry(grip_pcd, reset_bounding_box=False)
        vis.remove_geometry(goal_pcd, reset_bounding_box=False)
        idx = new_idx
        obj_pcd, grip_pcd, goal_pcd = load_npz_geometries(npz_paths[idx])
        vis.add_geometry(obj_pcd, reset_bounding_box=False)
        vis.add_geometry(grip_pcd, reset_bounding_box=False)
        vis.add_geometry(goal_pcd, reset_bounding_box=False)
        vis.update_renderer()
        _update_title()

    def next_frame(vis_) -> bool:
        nonlocal idx
        if idx >= len(npz_paths) - 1:
            return False
        _swap_frame(idx + 1)
        return False

    def prev_frame(vis_) -> bool:
        nonlocal idx
        if idx <= 0:
            return False
        _swap_frame(idx - 1)
        return False

    vis.register_key_callback(ord("N"), next_frame)
    vis.register_key_callback(ord("P"), prev_frame)
    vis.register_key_callback(ord(" "), next_frame)

    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    opt.background_color = VIEWER_BACKGROUND.copy()

    vis.run()
    vis.destroy_window()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "trajectory_dir",
        type=Path,
        help="Directory containing 000000.npz, 000001.npz, ... (articubot / convert_dataset output)",
    )
    p.add_argument(
        "--start",
        type=int,
        default=0,
        help="Initial frame index (0-based, clamped to valid range).",
    )
    p.add_argument(
        "--point-size",
        type=float,
        default=2.0,
        help="Open3D point size for rendering.",
    )
    args = p.parse_args(argv)

    traj_dir = args.trajectory_dir.expanduser().resolve()
    if not traj_dir.is_dir():
        print(f"Not a directory: {traj_dir}", file=sys.stderr)
        return 2

    try:
        visualize_npz_trajectory(traj_dir, start_index=args.start, point_size=args.point_size)
    except FileNotFoundError as ex:
        print(ex, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
