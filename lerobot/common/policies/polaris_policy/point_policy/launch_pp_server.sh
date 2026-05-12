#!/bin/bash

data_dir_1="/home/haotian/lerobot/polaris/src/polaris/policy/Point-Policy/data/pick_place_red_mug_realrobot_40_human_100/processed_data_pkl/expert_demos"
bc_weight_1="/home/haotian/lerobot/polaris/src/polaris/policy/Point-Policy/point_policy/exp_local/2026.05.04/pick_place_red_mug_realrobot_40_human_100/deterministic/024229_hidden_dim_256/snapshot/100000.pt"


TASK_NAME_1="pick_place_red_mug"

echo "Starting server 1 on port 8766..."
CUDA_VISIBLE_DEVICES=0 python /home/haotian/lerobot/lerobot/common/policies/polaris_policy/point_policy/point_policy_server.py \
    --bc_weight "$bc_weight_1" \
    --port 8766 \
    --overrides "agent=point_policy" "suite=point_policy" "dataloader=point_policy" \
    "suite.use_robot_points=true" "suite.use_object_points=true" \
    "experiment=eval_point_policy" "suite/task/franka_env=${TASK_NAME_1}" \
    "agent.max_episode_len=450" "data_dir=${data_dir_1}" &
PID2=$!

echo "Press Ctrl+C to stop all servers"

trap "echo 'Shutting down...'; kill $PID1 $PID2 2>/dev/null; exit" SIGINT SIGTERM

wait
echo "All servers shut down."