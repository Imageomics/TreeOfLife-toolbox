#!/bin/bash
#SBATCH --job-name tool_verifier
#SBATCH --partition=cpu
#SBATCH --time=00:05:00

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
  --nodes=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=1 \
  --output="${logs_dir}/tool_verifier.log" \
  python "${TOOLBOX_PATH}/main/verification.py" "${tool_name}"
