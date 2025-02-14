#!/bin/bash
#SBATCH --job-name tool_verifier
#SBATCH --mem=0
#SBATCH --time=00:05:00

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
  --nodes=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=1 \
  --mem=0 \
  --output="${logs_dir}/tool_verifier.log" \
  python "${REPO_ROOT}/src/distributed_downloader/tools/verification.py" "${tool_name}"
