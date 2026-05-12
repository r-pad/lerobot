#!/usr/bin/env bash
# Launch ghost_server.py in the mapanything conda env.
#
# Usage:
#   bash launch_ghost_server.sh                 # default port 8766
#   PORT=8767 bash launch_ghost_server.sh       # override port
set -euo pipefail

CALIB="${CALIB:-/home/haotian/lerobot/lerobot/scripts/droid_calibration}"
PORT="${PORT:-8766}"
MAPANY_MODEL="${MAPANY_MODEL:-facebook/map-anything}"
BC_WEIGHT="${BC_WEIGHT:-}"
DEVICE="${DEVICE:-cuda}"
CUDA_DEV="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Launching ghost_server: port=${PORT}  cuda=${CUDA_DEV}  calib=${CALIB}"
CUDA_VISIBLE_DEVICES="${CUDA_DEV}" \
    conda run -n mapanything --no-capture-output \
    python "${SCRIPT_DIR}/ghost_server.py" \
        --port "${PORT}" \
        --calib "${CALIB}" \
        --mapanything_model "${MAPANY_MODEL}" \
        --device "${DEVICE}" \
        ${BC_WEIGHT:+--bc_weight "${BC_WEIGHT}"}
