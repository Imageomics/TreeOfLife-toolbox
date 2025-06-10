#!/bin/bash
#SBATCH --job-name t2w_filter
#SBATCH --mem=0
#SBATCH --time=02:00:00

logs_dir="${OUTPUT_T2W_LOGS_FOLDER}"
mkdir -p "$logs_dir"

driver_memory="110G"
executor_memory="64G"

module load spark/3.4.1
module load miniconda3
source "${REPO_ROOT}/.venv/gbif_venv/bin/activate"

pbs-spark-submit \
    --driver-memory $driver_memory \
    --executor-memory $executor_memory \
    "${REPO_ROOT}/src/tol2webdataset/filter.py" \
    > "${logs_dir}/t2w_filter.log"
