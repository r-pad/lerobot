"""
Rename episode files in extradata directories (events, wilor_hand_pose) to be
consecutively numbered starting from 0.

Usage:
    python rename_extradata.py --root /path/to/extradata [--dry_run]
"""

import argparse
import re
import shutil
from pathlib import Path


EPISODE_PATTERN = re.compile(r"^episode_(\d+)(.*)$")


def collect_episodes(directory: Path) -> list[tuple[int, Path]]:
    """Return sorted list of (episode_idx, path) for all episode files in directory."""
    entries = []
    for p in directory.iterdir():
        m = EPISODE_PATTERN.match(p.name)
        if m:
            entries.append((int(m.group(1)), p))
    return sorted(entries, key=lambda x: x[0])


def rename_dir(directory: Path, dry_run: bool) -> None:
    if not directory.exists():
        print(f"  Skipping (not found): {directory}")
        return

    episodes = collect_episodes(directory)
    if not episodes:
        print(f"  No episode files in {directory}")
        return

    old_indices = [idx for idx, _ in episodes]
    already_consecutive = old_indices == list(range(len(old_indices)))
    if already_consecutive:
        print(f"  {directory.name}: already consecutive ({len(episodes)} files), nothing to do.")
        return

    print(f"  {directory.name}: {old_indices} → 0…{len(episodes)-1}")

    if dry_run:
        for new_idx, (old_idx, p) in enumerate(episodes):
            if old_idx != new_idx:
                m = EPISODE_PATTERN.match(p.name)
                new_name = f"episode_{new_idx:06d}{m.group(2)}"
                print(f"    [DRY] {p.name} → {new_name}")
        return

    # Two-pass rename to avoid collisions: old → .tmp, then .tmp → new
    tmp_paths = []
    for new_idx, (old_idx, p) in enumerate(episodes):
        if old_idx == new_idx:
            tmp_paths.append(None)
            continue
        tmp = p.with_name(p.name + ".rename_tmp")
        shutil.move(str(p), str(tmp))
        tmp_paths.append(tmp)

    for new_idx, ((_old_idx, orig_p), tmp) in enumerate(zip(episodes, tmp_paths)):
        if tmp is None:
            continue
        m = EPISODE_PATTERN.match(orig_p.name)
        new_name = f"episode_{new_idx:06d}{m.group(2)}"
        dst = orig_p.parent / new_name
        shutil.move(str(tmp), str(dst))
        print(f"    {orig_p.name} → {new_name}")

    print(f"    Done. {len(episodes)} files renumbered.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Renumber extradata episode files consecutively.")
    parser.add_argument("--root", type=str, required=True,
                        help="Root extradata directory containing events/ and wilor_hand_pose/")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would change without writing anything")
    args = parser.parse_args()

    root = Path(args.root)
    for subdir in ["events", "wilor_hand_pose"]:
        rename_dir(root / subdir, dry_run=args.dry_run)
