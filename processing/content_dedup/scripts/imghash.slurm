#!/bin/bash
#SBATCH --job-name=pdq-fathomnet
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --output=logs/%j_log.out


source /apps/lmod/lmod/init/bash

module load spark/3.5.1

BASE_DIR="" # Set your base path for data

PYSPARK_PYTHON=$BASE_DIR/content_dedup/.venv/bin/python slurm-spark-submit \
  --driver-memory 16G \
  --executor-memory 96G \
  --executor-cores 12 \
  scripts/imghash.py \
    --inputs '${BASE_DIR}/TreeOfLife/data/source=fathomnet/*/*.parquet' \
    --write-to $BASE_DIR/content_dedup/hashes/parquet/ \
    --max-partition-bytes 128MB
