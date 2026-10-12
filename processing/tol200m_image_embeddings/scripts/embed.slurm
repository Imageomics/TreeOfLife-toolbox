#!/bin/bash
# =============================================================================
# TreeOfLife HDF5 batch embedding - multi-rank GPU job (one task = one GPU rank).
# For sources with a moderate file count (bioscan, eol, fathomnet). For gbif-scale
# sources use count_images.py + build_chunks.py + embed_array.slurm instead.
#
# Submit with the RUN_* variables exported, e.g.:
#   export RUN_CONFIG=configs/embed_bioclip_2_5.yaml
#   export RUN_TARGET_DIR=<DATA_ROOT>/source=bioscan
#   export RUN_OUTPUT_DIR=<EMB_OUT>/model=bioclip_2_5/source=bioscan
#   sbatch --nodes=4 --time=02:00:00 embed.slurm     # 4 nodes x 1 GPU = 4 ranks
#
# Rank sizing: ranks split whole files, and each rank's dataloader workers also
# split whole files, so keep  files_per_rank >= num_workers  (else workers idle)
# and world_size <= num_files (else a rank gets nothing).
# =============================================================================
#SBATCH --job-name=tol_embed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1            # 1 task == 1 rank == 1 GPU
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=26             # match the cluster's per-GPU CPU share
#SBATCH --mem=234G                     # match the cluster's per-GPU memory share
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --mail-type=END,FAIL
#SBATCH --output=embed_%x_%j.out
#SBATCH --error=embed_%x_%j.err

set -euo pipefail

: "${RUN_CONFIG:?set RUN_CONFIG (path to embed_*.yaml)}"
: "${RUN_TARGET_DIR:?set RUN_TARGET_DIR (source=... dir with .h5 files)}"
: "${RUN_OUTPUT_DIR:?set RUN_OUTPUT_DIR (embedding output dir)}"
RUN_FILE_LIST="${RUN_FILE_LIST:-}"     # optional: restrict to a subset of files

module load cuda                       # match your venv's torch CUDA major
source /path/to/venv/bin/activate

# Cache compile artifacts + HF weights on scratch (avoids recompiling /
# re-downloading the model every job).
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/path/to/scratch/cache/torchinductor}"
export HF_HOME="${HF_HOME:-/path/to/scratch/cache/huggingface}"
export OMP_NUM_THREADS=1               # keep dataloader workers from oversubscribing cores
export HDF5_USE_FILE_LOCKING=FALSE     # read-only access: safe, skips flock on GPFS
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"   # Slurm 23.11+ srun guard
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${HF_HOME}" "${RUN_OUTPUT_DIR}"

echo "Job ${SLURM_JOB_ID}: ntasks=${SLURM_NTASKS} config=${RUN_CONFIG}"
echo "  target=${RUN_TARGET_DIR}"
echo "  output=${RUN_OUTPUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

EXTRA_ARGS=()
[[ -n "${RUN_FILE_LIST}" ]] && EXTRA_ARGS+=(--file_list "${RUN_FILE_LIST}")

srun python -m hpc_inference.inference.embed.open_clip_embed \
    "${RUN_TARGET_DIR}" \
    "${RUN_OUTPUT_DIR}" \
    --input_type hdf5 \
    --config "${RUN_CONFIG}" \
    "${EXTRA_ARGS[@]}"

echo "Done. Embeddings + profile_results under: ${RUN_OUTPUT_DIR}"
exit 0
