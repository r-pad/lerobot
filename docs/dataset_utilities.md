# Dataset Utilities

Scripts for inspecting and transforming `LeRobotDataset`s.

## Visualizing Datasets

```bash
python lerobot/scripts/visualize_dataset.py \
    --repo-id <repo_id> \
    --episode-index 0
```

Opens a [Rerun](https://rerun.io/) viewer for the episode.

## `upgrade_dataset.py`

Create a new dataset from an existing one, adding/modifying features:

```bash
python lerobot/scripts/upgrade_dataset.py \
    --source_repo_id <source_id> \
    --target_repo_id <target_id> \
    --new_features goal_gripper_proj gripper_pcds next_event_idx \
    --push_to_hub
```

Supports `--phantomize` (retarget human demo to robot via [Phantom](https://phantom-human-videos.github.io/)) and `--humanize` (keep human in video). Use `--discard_episodes` to skip problematic episodes. See [ghost.md](ghost.md) for the full human-demonstration pipeline.

## `merge_datasets.py`

```bash
python lerobot/scripts/merge_datasets.py \
    --datasets DATASET1_ID DATASET2_ID DATASET3_ID \
    --target_repo_id MERGED_DATASET_ID \
    --push_to_hub
```

All input datasets must have compatible features and fps.

## `subsample_dataset.py`

Downsample a dataset to a target fps:

```bash
python lerobot/scripts/subsample_dataset.py \
    --source_repo_id SOURCE_ID \
    --target_repo_id TARGET_ID \
    --target_fps <fps>
```

## `annotate_events.py`

Manually annotate ground-truth events (used to define subgoals for goal-conditioned policies). Takes a directory of episode videos and a known task spec, then prompts for the frame index of each event per episode:

```bash
python lerobot/scripts/annotate_events.py <video_dir> "fold the towel"
```

Run without arguments to see the list of available task specs.
