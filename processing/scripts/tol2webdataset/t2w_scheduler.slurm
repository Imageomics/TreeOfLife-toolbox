#!/bin/bash
#SBATCH --job-name t2w_scheduler
#SBATCH --mem=0
#SBATCH --time=00:10:00

logs_dir="${OUTPUT_T2W_LOGS_FOLDER}"
mkdir -p "$logs_dir"

module load intel/2021.10.0
module load intelmpi/2021.10
module load miniconda3
source "${REPO_ROOT}/.venv/gbif_venv/bin/activate"
export PYARROW_IGNORE_TIMEZONE=1
export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=0

srun \
  --mpi=pmi2 \
  --nodes=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=1 \
  --mem=0 \
  --output="${logs_dir}/t2w_scheduler.log" \
  python "${REPO_ROOT}/src/tol2webdataset/scheduler.py"
