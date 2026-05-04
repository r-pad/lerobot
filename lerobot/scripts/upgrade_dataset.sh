#!/usr/bin/env bash
set -euo pipefail

MAX_JOBS=4
EXTRA_DATA_ROOT="/home/haotian/Desktop/h2rdataset"

COMMON_ARGS=(
  --humanize
  --new_features goal_gripper_proj gripper_pcds next_event_idx
)

# Format:
# "SOURCE_REPO_ID TARGET_REPO_ID EXTRA_DATA_FOLDER_UNDER_h2rdataset"
DATASETS=(
  "xiaochyVera/pick_red_mug_human_ss_map Kovavavvavava/pick_red_mug_human_ss_map_hg"
  "xiaochyVera/pick_red_mug_human_1_ss_map Kovavavvavava/pick_red_mug_human_1_ss_map_hg"
  "xiaochyVera/pick_red_mug_human_2_ss_map Kovavavvavava/pick_red_mug_human_2_ss_map_hg"
  "xiaochyVera/pick_red_mug_human_3_ss_map Kovavavvavava/pick_red_mug_human_3_ss_map_hg"
)

run_upgrade() {
  local source_repo_id="$1"
  local target_repo_id="$2"
  local extra_data_folder="$3"

  local path_to_extradata="${EXTRA_DATA_ROOT}/${extra_data_folder}"

  echo "Starting: $source_repo_id -> $target_repo_id"
  echo "Extra data: $path_to_extradata"

  python upgrade_dataset.py \
    --source_repo_id "$source_repo_id" \
    --target_repo_id "$target_repo_id" \
    "${COMMON_ARGS[@]}" \
    --path_to_extradata "$path_to_extradata"

  echo "Finished: $source_repo_id -> $target_repo_id"
}

for row in "${DATASETS[@]}"; do
  read -r source_repo_id target_repo_id extra_data_folder <<< "$row"

  run_upgrade "$source_repo_id" "$target_repo_id" "$extra_data_folder" &

  while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
    wait -n
  done
done

wait
echo "All dataset upgrades completed."