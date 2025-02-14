#!/bin/bash
#SBATCH --job-name tool_worker
#SBATCH --mem=0
#SBATCH --time=03:00:00

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 tool_name"
    exit 1
fi

tool_name=$1

logs_dir="${OUTPUT_TOOLS_LOGS_FOLDER}"
mkdir -p "$logs_dir"

module load intel/2021.10.0
module load intelmpi/2021.10
module load miniconda3/23.3.1-py310
source "${REPO_ROOT}/.venv/bin/activate"
export PYARROW_IGNORE_TIMEZONE=1
export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=0
export PYTHONPATH=${PYTHONPATH}:"${REPO_ROOT}/src":"${REPO_ROOT}/distributed-downloader"

srun \
  --mpi=pmi2 \
  --nodes="$TOOLS_MAX_NODES" \
  --ntasks-per-node="$TOOLS_WORKERS_PER_NODE" \
  --cpus-per-task="$TOOLS_CPU_PER_WORKER" \
  --mem=0 \
  --output="${logs_dir}/tool_worker-%2t.log" \
  python "${REPO_ROOT}/src/distributed_downloader/tools/runner.py" "${tool_name}"
