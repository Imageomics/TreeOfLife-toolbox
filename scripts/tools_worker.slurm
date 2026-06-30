#!/bin/bash
#SBATCH --job-name tool_worker
#SBATCH --partition=cpu
#SBATCH --time=03:00:00

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 tool_name"
    exit 1
fi

tool_name=$1

logs_dir="${OUTPUT_TOOLS_LOGS_FOLDER}"
mkdir -p "$logs_dir"

module load intel/2021.10.0
module load intel-oneapi-mpi/2021.10.0
source "${REPO_ROOT}/.venv/bin/activate"
export PYARROW_IGNORE_TIMEZONE=1

srun \
  --mpi=pmi2 \
  --nodes="$TOOLS_MAX_NODES" \
  --ntasks-per-node="$TOOLS_WORKERS_PER_NODE" \
  --cpus-per-task="$TOOLS_CPU_PER_WORKER" \
  --output="${logs_dir}/tool_worker-%2t.log" \
  python "${TOOLBOX_PATH}/main/runner.py" "${tool_name}"
