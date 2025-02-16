#!/bin/bash
#SBATCH --job-name tool_filter
#SBATCH --mem=0

logs_dir="${OUTPUT_TOOLS_LOGS_FOLDER}"
mkdir -p "$logs_dir"

driver_memory="110G"
executor_memory="64G"

module load spark/3.4.1
module load miniconda3/23.3.1-py310
source "${REPO_ROOT}/.venv/bin/activate"
export PYTHONPATH=${PYTHONPATH}:"${REPO_ROOT}/src":"${REPO_ROOT}/distributed-downloader"

pbs-spark-submit \
    --driver-memory $driver_memory \
    --executor-memory $executor_memory \
    "/users/PAS2119/andreykopanev/distributed_downloader_test/column_name_change/filter.py" \
    > "${logs_dir}/tool_filter.log"
