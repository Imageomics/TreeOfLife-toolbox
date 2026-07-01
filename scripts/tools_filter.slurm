#!/bin/bash
#SBATCH --job-name tool_filter
#SBATCH --partition=cpu
#SBATCH --mem=200G
#SBATCH --time=2:00:00

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 tool_name"
    exit 1
fi

tool_name=$1

logs_dir="${OUTPUT_TOOLS_LOGS_FOLDER:-${SCRATCH:-$PWD}/tools_logs}"
mkdir -p "$logs_dir"

driver_memory="110G"
executor_memory="64G"

module load spark/3.5.1
module load python/3.12
if [ -z "$REPO_ROOT" ]; then
    echo "REPO_ROOT is not set. Please export REPO_ROOT to the repository path." >&2
    exit 1
fi
source "${REPO_ROOT}/.venv/bin/activate"

spark-submit \
    --driver-memory $driver_memory \
    --executor-memory $executor_memory \
    "${TOOLBOX_PATH}/main/filter.py" \
    "${tool_name}" \
    > "${logs_dir}/tool_filter.log"
