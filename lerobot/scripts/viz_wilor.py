#!/usr/bin/env python3
"""
Visualize WiLoR 3D hand mesh from a .npy file, saved as a video.

The npy file contains the full MANO mesh: (T, 778, 3).

Usage:
    python viz_wilor.py <npy_file> [options]

Examples:
    python viz_wilor.py episode_000000.mp4.npy
    python viz_wilor.py episode_000000.mp4.npy --out hand.mp4 --plot_pinch --step 2
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation
from pathlib import Path


# MANO mesh fingertip vertex indices (standard, 778-vertex mesh)
THUMB_TIP_IDX = 763
INDEX_TIP_IDX = 343


def compute_pinch(hand_pcd: np.ndarray) -> np.ndarray:
    """Thumb-index tip distance per frame. hand_pcd: (T, 778, 3)"""
    return np.linalg.norm(
        hand_pcd[:, THUMB_TIP_IDX] - hand_pcd[:, INDEX_TIP_IDX], axis=-1
    )


def draw_hand(ax, mesh, frame_idx=None, pinch=None, subsample=6):
    """Draw one MANO mesh frame onto a 3D axes."""
    ax.cla()

    # Subsample mesh vertices for speed
    pts = mesh[::subsample]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c="#3498db", s=2, alpha=0.5, depthshade=True)

    # Highlight thumb and index tips
    for idx, label, color in [
        (THUMB_TIP_IDX, "thumb", "#e74c3c"),
        (INDEX_TIP_IDX, "index", "#2ecc71"),
    ]:
        ax.scatter(*mesh[idx], color=color, s=80, zorder=6, label=label)

    # Pinch line
    ax.plot(
        [mesh[THUMB_TIP_IDX, 0], mesh[INDEX_TIP_IDX, 0]],
        [mesh[THUMB_TIP_IDX, 1], mesh[INDEX_TIP_IDX, 1]],
        [mesh[THUMB_TIP_IDX, 2], mesh[INDEX_TIP_IDX, 2]],
        color="yellow", linewidth=2, linestyle="--",
    )

    title = f"Frame {frame_idx}" if frame_idx is not None else ""
    if pinch is not None:
        title += f"  |  pinch={pinch:.4f}"
    ax.set_title(title, color="white")
    ax.set_xlabel("X", color="gray")
    ax.set_ylabel("Y", color="gray")
    ax.set_zlabel("Z", color="gray")
    ax.tick_params(colors="gray")
    ax.set_facecolor("#1a1a2e")


def set_equal_axes(ax, mesh_all):
    mins = mesh_all.reshape(-1, 3).min(axis=0)
    maxs = mesh_all.reshape(-1, 3).max(axis=0)
    center = (mins + maxs) / 2
    half = (maxs - mins).max() / 2 * 0.6
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def save_video(hand_pcd, out_path: Path, plot_pinch: bool, step: int, fps: int):
    T = hand_pcd.shape[0]
    pinch_series = compute_pinch(hand_pcd)

    if plot_pinch:
        fig = plt.figure(figsize=(14, 5), facecolor="#0f0f1a")
        ax3d   = fig.add_subplot(121, projection="3d")
        ax_pin = fig.add_subplot(122)
        ax_pin.set_facecolor("#1a1a2e")
        ax_pin.plot(pinch_series, color="#2ecc71", linewidth=1.5)
        vline = ax_pin.axvline(0, color="orange", linestyle="--")
        ax_pin.set_xlabel("frame", color="gray")
        ax_pin.set_ylabel("pinch distance", color="gray")
        ax_pin.set_title("Thumb-Index Pinch", color="white")
        ax_pin.tick_params(colors="gray")
    else:
        fig = plt.figure(figsize=(7, 6), facecolor="#0f0f1a")
        ax3d  = fig.add_subplot(111, projection="3d")
        vline = None

    ax3d.set_facecolor("#1a1a2e")
    frames = list(range(0, T, step))

    def update(frame_idx):
        draw_hand(ax3d, hand_pcd[frame_idx], frame_idx=frame_idx, pinch=pinch_series[frame_idx])
        set_equal_axes(ax3d, hand_pcd)
        if vline is not None:
            vline.set_xdata([frame_idx, frame_idx])
        return []

    ani = FuncAnimation(fig, update, frames=frames, interval=1000 // fps, blit=False)
    plt.tight_layout()
    if out_path is not None:
        print(f"Rendering {len(frames)} frames -> {out_path} ...")
        ani.save(str(out_path), writer="ffmpeg", fps=fps, dpi=120)
        plt.close(fig)
        print("Done.")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize WiLoR MANO hand mesh.")
    parser.add_argument("npy_file",     type=Path, help="Path to episode .npy file  (T, 778, 3)")
    parser.add_argument("--animate",    action="store_true",
                        help="Show interactive animation instead of saving to file")
    parser.add_argument("--out",        type=Path, default=None,
                        help="Output video path (default: <npy_file>.mp4, ignored with --animate)")
    parser.add_argument("--step",       type=int,  default=1,
                        help="Frame step (default: 1 = every frame)")
    parser.add_argument("--fps",        type=int,  default=20,
                        help="Output video FPS / animation speed (default: 20)")
    parser.add_argument("--plot_pinch", action="store_true",
                        help="Show thumb-index pinch distance plot alongside 3D view")
    args = parser.parse_args()

    if not args.npy_file.exists():
        print(f"ERROR: file not found: {args.npy_file}")
        raise SystemExit(1)

    out_path = None if args.animate else (args.out or args.npy_file.with_suffix(".mp4"))

    hand_pcd = np.load(args.npy_file).astype(np.float32)
    T, N, _ = hand_pcd.shape
    print(f"Loaded {args.npy_file.name}  shape={hand_pcd.shape}  ({'mesh' if N == 778 else f'{N} pts'})")

    pinch = compute_pinch(hand_pcd)
    print(f"Pinch  min={pinch.min():.4f}  max={pinch.max():.4f}  mean={pinch.mean():.4f}")

    save_video(hand_pcd, out_path=out_path, plot_pinch=args.plot_pinch,
               step=args.step, fps=args.fps)


if __name__ == "__main__":
    main()