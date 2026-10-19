#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_ID="${1:?Supply a fresh diagnostic run ID}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
python3 -m panama_uav train --condition clean --profile smoke --run-id "${RUN_ID}" --device cpu
python3 -m panama_uav evaluate --run-id "${RUN_ID}" --diagnostic --cases 1 --agents 2 --conditions controlled_cloud_task_5 physical_cloud_combined --device cpu
python3 -m panama_uav report --kind inference --path "output/communication/${RUN_ID}/inference"
python3 -m panama_uav replay --run-id "${RUN_ID}" --diagnostic --device cpu
