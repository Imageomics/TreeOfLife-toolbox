#!/bin/bash
# =============================================================================
# TreeOfLife HDF5 batch embedding - SLURM ARRAY task (1 GPU per task), for
# sources with many files of highly variable size (gbif: ~24.6K files, 1-20K
# images each).
#
# Prerequisite (one CPU job / login-safe for small sources):
#   python count_images.py <DATA_ROOT>/source=gbif file_counts.parquet
#   NTASKS=$(python build_chunks.py file_counts.parquet <EMB_OUT>/model=<m>/source=gbif 7000000)
#
# Then submit (cap concurrency to be a good citizen on a shared account):
#   export RUN_CONFIG=configs/embed_bioclip_2_5.yaml
#   export RUN_TARGET_DIR=<DATA_ROOT>/source=gbif
#   export RUN_OUT_ROOT=<EMB_OUT>/model=bioclip_2_5/source=gbif
#   sbatch --array=0-$((NTASKS-1))%6 --time=06:00:00 embed_array.slurm
#
# Each task N reads its precomputed image-count-balanced chunk
# (task_<N>/filelist.txt) and writes task-scoped output under task_<N>/ so
# tasks never collide. Budget --time for the WORST chunk: chunks with many
# small files run slower per image (per-file HDF5 open overhead).
# =============================================================================
#SBATCH --job-name=tol_embed_array
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1            # 1 task == world_size 1 == rank 0
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=234G
#SBATCH --partition=gpu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --mail-type=FAIL
#SBATCH --output=embed_array_%A_%a.out
#SBATCH --error=embed_array_%A_%a.err

set -euo pipefail

: "${RUN_CONFIG:?set RUN_CONFIG}"
: "${RUN_TARGET_DIR:?set RUN_TARGET_DIR}"
: "${RUN_OUT_ROOT:?set RUN_OUT_ROOT}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as a job array}"

module load cuda
source /path/to/venv/bin/activate

export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/path/to/scratch/cache/torchinductor}"
export HF_HOME="${HF_HOME:-/path/to/scratch/cache/huggingface}"
export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${HF_HOME}"

N="${SLURM_ARRAY_TASK_ID}"
TASK_DIR="${RUN_OUT_ROOT}/task_${N}"
CHUNK_LIST="${TASK_DIR}/filelist.txt"
[[ -f "${CHUNK_LIST}" ]] || { echo "ERROR: no chunk list at ${CHUNK_LIST} (run build_chunks.py first)"; exit 1; }

echo "Array task ${N}: $(wc -l < "${CHUNK_LIST}") files -> ${TASK_DIR}"

srun python -m hpc_inference.inference.embed.open_clip_embed \
    "${RUN_TARGET_DIR}" \
    "${TASK_DIR}" \
    --input_type hdf5 \
    --config "${RUN_CONFIG}" \
    --file_list "${CHUNK_LIST}"

echo "Task ${N} done -> ${TASK_DIR}"
exit 0
