#!/usr/bin/env python3
"""
Auto-annotate human demo episodes using WiLoR hand pose openness as keyframe signal.

Hand openness is estimated from the thumb-index pinch distance.
Open/close transitions are used as goal events, mirroring how
extract_events_with_gripper_pos works for robot data.

A task spec must be provided via --task so the script can validate that the
number of auto-detected keyframes matches the expected number of subgoals.
The last frame is always forced as the final keyframe.

Usage:
    python annotate_wilor.py <extradata_dir> --task <task_name> [options]

The script reads from:
    {extradata_dir}/wilor_hand_pose/episode_XXXXXX.mp4.npy

Annotation JSONs are written to:
    {extradata_dir}/events/episode_XXXXXX.mp4.json
"""

import json
import sys
import argparse
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Task spec — mirrors annotate_event.py / classify_utils.py
# Each list defines the ordered subgoal labels for a task.
# The number of detected keyframes (including the mandatory last frame) must
# equal len(subgoals).
# ---------------------------------------------------------------------------

TASK_SPEC = {
    "pick and place red mug": ["grasp red mug", "place red mug on table"],
    "stack bowls": ["grasp green bowl", "stack on blue bowl"],
    "insert donut": ["pick up donut", "drop donut"],
    "pick toys": ["grasp blue toy", "drop blue toy","grasp orange toy", "drop orange toy","grasp yellow toy", "drop yellow toy"],
}


# ---------------------------------------------------------------------------
# Hand openness estimation — thumb-index pinch distance
# ---------------------------------------------------------------------------

# WiLoR outputs the full MANO mesh (778 vertices). Fingertip vertex indices:
#   0        = wrist
#   1-4      = thumb  (CMC, MCP, IP, tip)
#   5-8      = index  (MCP, PIP, DIP, tip)
#   9-12     = middle
#   13-16    = ring
#   17-20    = pinky
THUMB_TIP_IDX = 745  # MANO mesh vertex index for thumb tip
INDEX_TIP_IDX = 317   # MANO mesh vertex index for index finger tip


def compute_hand_openness(hand_pcd: np.ndarray) -> np.ndarray:
    """
    Compute thumb-index pinch distance for each frame.

    Small value  -> fingers pinched together (grasping / closed).
    Large value  -> fingers spread apart (open / releasing).

    Args:
        hand_pcd: (T, N, 3) array of MANO keypoints in world coordinates.

    Returns:
        openness: (T,) float array of Euclidean thumb-index tip distances.
    """
    thumb = hand_pcd[:, THUMB_TIP_IDX, :]  # (T, 3)
    index = hand_pcd[:, INDEX_TIP_IDX, :]  # (T, 3)
    return np.linalg.norm(thumb - index, axis=-1)  # (T,)


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------

def extract_events_from_hand_openness(
    openness: np.ndarray,
    close_thresh: float,
    open_thresh: float,
    min_hold_frames: int,
    n_transitions: int,
    thumb_z: np.ndarray | None = None,
    index_z: np.ndarray | None = None,
    grasp_z_max: float | None = None,
    drop_z_min: float | None = None,
    min_pinch: float = 0.02,
) -> list[int]:
    """
    Extract keyframe indices from the thumb-index pinch signal.

    Collects exactly n_transitions open/close transition frames, then always
    appends T-1 as the final keyframe. So the returned list has length
    n_transitions + 1, matching len(task_spec) where the last subgoal
    corresponds to the end of the episode.

    A CLOSE event fires when openness drops below close_thresh (grasp).
    An OPEN  event fires when openness rises above open_thresh  (release).

    Frames where openness < min_pinch are skipped entirely (likely tracker
    noise or an occluded hand), so they do not influence state or hold counts.

    Optional z-axis gating (e.g. for pick_place_toys):
        grasp_z_max: close events only fire when both thumb_z and index_z < this value.
        drop_z_min:  open  events only fire when both thumb_z and index_z > this value.

    Args:
        openness:        (T,) pinch distance per frame.
        close_thresh:    distance below which hand is considered closed.
        open_thresh:     distance above which hand is considered open.
        min_hold_frames: debounce — transition only fires after this many
                         consecutive frames in the new state.
        n_transitions:   number of transition events to collect (= len(task_spec)).
        thumb_z:         (T,) z-coordinate of thumb tip (world frame).
        index_z:         (T,) z-coordinate of index tip (world frame).
        grasp_z_max:     if set, close events require thumb_z and index_z both < this.
        drop_z_min:      if set, open  events require thumb_z and index_z both > this.
        min_pinch:       frames with openness below this value are ignored (default: 0.02).

    Returns:
        List of length n_transitions + 1, with T-1 always appended as an extra
        final entry regardless of whether it was already detected.
    """
    T = len(openness)
    goal_indices = []
    is_closed = openness[0] < close_thresh
    hold_counter = 0
    transition_start = None  # first frame of the current candidate transition

    for i in range(T):
        if len(goal_indices) >= n_transitions:
            break
        val = openness[i]
        if val < min_pinch:
            continue
        if not is_closed:
            if val < close_thresh:
                if hold_counter == 0:
                    transition_start = i
                hold_counter += 1
                if hold_counter >= min_hold_frames:
                    z_ok = (
                        grasp_z_max is None
                        or (thumb_z[transition_start] < grasp_z_max and index_z[transition_start] < grasp_z_max)
                    )
                    if z_ok:
                        is_closed = True
                        goal_indices.append(transition_start)
                    hold_counter = 0
                    transition_start = None
            else:
                hold_counter = 0
                transition_start = None
        else:
            if val > open_thresh:
                if hold_counter == 0:
                    transition_start = i
                hold_counter += 1
                if hold_counter >= min_hold_frames:
                    z_ok = (
                        drop_z_min is None
                        or (thumb_z[transition_start] > drop_z_min and index_z[transition_start] > drop_z_min)
                    )
                    if z_ok:
                        is_closed = False
                        goal_indices.append(transition_start)
                    hold_counter = 0
                    transition_start = None
            else:
                hold_counter = 0
                transition_start = None

    # Always force the last frame as the final keyframe
    goal_indices.append(T - 1)

    return goal_indices


# ---------------------------------------------------------------------------
# Per-episode processing
# ---------------------------------------------------------------------------

def process_episode(
    episode_idx: int,
    wilor_dir: Path,
    output_dir: Path,
    task_spec: list[str],
    close_thresh: float,
    open_thresh: float,
    min_hold_frames: int,
    interactive: bool,
    overwrite: bool,
    plot: bool,
    task_name: str = "",
    min_pinch: float = 0.02,
) -> bool:
    """
    Annotate one episode. Returns True if annotation was written, False if skipped.
    """
    episode_name = f"episode_{episode_idx:06d}.mp4"
    json_file    = output_dir / f"{episode_name}.json"
    npy_file     = wilor_dir  / f"episode_{episode_idx:06d}.mp4.npy"

    if json_file.exists() and not overwrite:
        print(f"  [{episode_idx:04d}] already annotated, skipping  (use --overwrite to redo)")
        return False

    if not npy_file.exists():
        print(f"  [{episode_idx:04d}] WiLoR file not found: {npy_file}, skipping")
        return False

    hand_pcd = np.load(npy_file).astype(np.float32)  # (T, N, 3)
    T = hand_pcd.shape[0]

    openness   = compute_hand_openness(hand_pcd)
    expected_n = len(task_spec)

    thumb_z = index_z = grasp_z_max = drop_z_min = None
    if task_name == "pick toys":
        thumb_z    = hand_pcd[:, THUMB_TIP_IDX, 2]
        index_z    = hand_pcd[:, INDEX_TIP_IDX, 2]
        grasp_z_max = 0.1
        drop_z_min  = 0.15

    event_idxs = extract_events_from_hand_openness(
        openness,
        close_thresh=close_thresh,
        open_thresh=open_thresh,
        min_hold_frames=min_hold_frames,
        n_transitions=expected_n,  # T-1 is appended on top, giving expected_n + 1 total
        thumb_z=thumb_z,
        index_z=index_z,
        grasp_z_max=grasp_z_max,
        drop_z_min=drop_z_min,
        min_pinch=min_pinch,
    )
    ok = len(event_idxs) == expected_n + 1
    status = "OK" if ok else f"MISMATCH (expected {expected_n + 1}, got {len(event_idxs)})"
    print(f"\n  [{episode_idx:04d}]  frames={T}  "
          f"pinch min={openness.min():.4f} max={openness.max():.4f}  "
          f"p10={np.percentile(openness, 10):.4f} p90={np.percentile(openness, 90):.4f}  "
          f"{status}  keyframes: {event_idxs}")

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(openness, label="pinch distance")
        ax.axhline(close_thresh, color="red",   linestyle="--", label=f"close_thresh={close_thresh}")
        ax.axhline(open_thresh,  color="green", linestyle="--", label=f"open_thresh={open_thresh}")
        for idx in event_idxs:
            ax.axvline(idx, color="orange", alpha=0.7, label=f"keyframe {idx}")
        ax.set_title(f"Episode {episode_idx:04d}  —  {status}")
        ax.set_xlabel("frame")
        ax.set_ylabel("thumb-index distance")
        ax.legend(loc="upper right", fontsize=7)
        plt.tight_layout()
        plt.show()

    if not ok:
        print(f"    subgoals: {task_spec}")
        if not interactive:
            print(f"    Skipping — re-run with --interactive to fix, or adjust --close_thresh / --open_thresh")
            return False

    if interactive:
        ans = input(f"    Accept {len(event_idxs)} keyframes (incl. forced last frame)? [Y/n/e(dit)]: ").strip().lower()
        if ans == "n":
            print("    Skipped.")
            return False
        if ans == "e":
            raw = input(f"    Enter {expected_n + 1} indices, last must be final frame "
                        f"(T-1={T-1}), comma-separated: ").strip()
            try:
                event_idxs = [int(x.strip()) for x in raw.split(",")]
                if event_idxs[-1] != T - 1:
                    event_idxs[-1] = T - 1
                    print(f"    Forced last index to {T - 1}")
            except ValueError:
                print("    Parse error — keeping auto-detected indices.")
        if len(event_idxs) != expected_n + 1:
            print(f"    ERROR: have {len(event_idxs)} indices but need {expected_n + 1}. Skipping.")
            return False

    data = {"event_idxs": event_idxs, "events": task_spec}
    with open(json_file, "w") as f:
        json.dump(data, f)
    print(f"    wrote {json_file.name}  ->  {list(zip(task_spec, event_idxs))}")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Auto-annotate human demos using WiLoR thumb-index pinch as keyframe signal."
    )
    parser.add_argument(
        "extradata_dir", type=Path,
        help="Root extradata dir. Reads from {dir}/wilor_hand_pose/, writes to {dir}/events/",
    )
    parser.add_argument(
        "--task", type=str, required=True, choices=list(TASK_SPEC.keys()),
        help="Task name — must match a key in TASK_SPEC",
    )
    parser.add_argument("--episodes",     type=int, nargs="*", default=None,
                        help="Specific episode indices to process (default: all in wilor dir)")
    parser.add_argument("--close_thresh", type=float, default=0.042,
                        help="Pinch distance below which hand is closed (default: 0.042)")
    parser.add_argument("--open_thresh",  type=float, default=0.055,
                        help="Pinch distance above which hand is open (default: 0.055)")
    parser.add_argument("--min_hold",     type=int,   default=3,
                        help="Frames to hold new state before firing event (default: 3)")
    parser.add_argument("--min_pinch",    type=float, default=0.02,
                        help="Ignore frames with pinch distance below this value (default: 0.02)")
    parser.add_argument("--interactive",  action="store_true",
                        help="Pause after each episode to confirm or manually edit indices")
    parser.add_argument("--overwrite",    action="store_true",
                        help="Re-annotate episodes that already have a JSON file")
    parser.add_argument("--plot",         action="store_true",
                        help="Plot the pinch signal with detected keyframes for each episode")

    args = parser.parse_args()

    wilor_dir  = args.extradata_dir / "wilor_hand_pose"
    output_dir = args.extradata_dir / "events"
    task_spec  = TASK_SPEC[args.task]

    if not wilor_dir.exists():
        print(f"ERROR: WiLoR directory not found: {wilor_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover episodes
    if args.episodes is not None:
        episode_indices = args.episodes
    else:
        npy_files = sorted(wilor_dir.glob("episode_*.mp4.npy"))
        if not npy_files:
            print(f"No WiLoR npy files found in {wilor_dir}")
            sys.exit(1)
        episode_indices = [int(p.name.split("_")[1].split(".")[0]) for p in npy_files]

    print(f"Task     : {args.task}")
    print(f"Subgoals : {task_spec}  ({len(task_spec)} keyframes expected per episode)")
    print(f"Pinch thresholds: close={args.close_thresh}  open={args.open_thresh}  min_hold={args.min_hold}  min_pinch={args.min_pinch}")
    print(f"Processing {len(episode_indices)} episode(s)...\n")

    written = 0
    for ep_idx in episode_indices:
        ok = process_episode(
            ep_idx,
            wilor_dir=wilor_dir,
            output_dir=output_dir,
            task_spec=task_spec,
            close_thresh=args.close_thresh,
            open_thresh=args.open_thresh,
            min_hold_frames=args.min_hold,
            interactive=args.interactive,
            overwrite=args.overwrite,
            plot=args.plot,
            task_name=args.task,
            min_pinch=args.min_pinch,
        )
        if ok:
            written += 1

    print(f"\nDone. Wrote {written}/{len(episode_indices)} annotation file(s) to {output_dir}")


if __name__ == "__main__":
    main()