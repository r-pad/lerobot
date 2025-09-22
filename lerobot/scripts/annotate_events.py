#!/usr/bin/env python3
import json
import sys
import os
from pathlib import Path

TASK_SPEC = {
    "grasp mug and place mug on table": ["grasp mug", "place mug on table"],
    "fold the bottoms": ["grasp 1", "fold 1", "grasp 2", "fold 2", "grasp 3", "fold 3"],
}

def main():
    if len(sys.argv) != 3:
        print("usage: annotate.py <output_dir> <task_spec>")
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    task_spec = sys.argv[2]

    if task_spec not in TASK_SPEC:
        print(f"unknown task spec: {task_spec}")
        print(f"available: {list(TASK_SPEC.keys())}")
        sys.exit(1)

    events = TASK_SPEC[task_spec]
    output_dir.mkdir(exist_ok=True)

    episode_num = 0
    while True:
        episode_name = f"episode_{episode_num:06d}.mp4"
        json_file = output_dir / f"{episode_name}.json"

        if json_file.exists():
            episode_num += 1
            continue

        print(f"\n=== {episode_name} ===")
        print(f"events: {events}")

        # get indices
        indices_str = input(f"event indices for {len(events)} events (comma-separated, empty to quit): ").strip()
        if not indices_str:
            break

        indices = [int(x.strip()) for x in indices_str.split(",")]

        if len(indices) != len(events):
            print(f"error: expected {len(events)} indices, got {len(indices)}")
            continue

        # write json
        data = {"event_idxs": indices, "events": events}
        with open(json_file, "w") as f:
            json.dump(data, f)

        print(f"wrote {json_file.name}")
        episode_num += 1

if __name__ == "__main__":
    main()
