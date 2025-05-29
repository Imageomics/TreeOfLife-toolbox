#!/bin/bash
#SBATCH --job-name t2w_worker
#SBATCH --mem=0
#SBATCH --time=03:00:00

# Check if any arguments were passed
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 job_index"
    exit 1
fi

job_index=$1

logs_dir="${OUTPUT_T2W_LOGS_FOLDER}/${job_index}"
mkdir -p "$logs_dir"

module load intel/2021.10.0
module load intelmpi/2021.10
module load miniconda3
source "${REPO_ROOT}/.venv/gbif_venv/bin/activate"
export PYARROW_IGNORE_TIMEZONE=1
export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=0

srun \
  --mpi=pmi2 \
  --nodes="$T2W_MAX_NODES_PER_RUNNER" \
  --ntasks-per-node="$T2W_WORKERS_PER_NODE" \
  --cpus-per-task="$T2W_CPU_PER_WORKER" \
  --mem=0 \
  --output="${logs_dir}/t2w_worker-%2t.log" \
  python "${REPO_ROOT}/src/tol2webdataset/runner.py"
