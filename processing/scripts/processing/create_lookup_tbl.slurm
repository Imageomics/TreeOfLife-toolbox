#!/bin/bash
#SBATCH --job-name create_lookup_tbl
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=96
#SBATCH --time=00:30:00
#SBATCH --account=YOUR_ACCOUNT

# Cluster: Cardinal
module load spark/3.5.1 

# Define Spark configurations as strings
SPARK_DRIVER_MEMORY="64G"
SPARK_EXECUTOR_MEMORY="75G"

GROUP="yale_plants"
BASE_DIR=""  # Set your base path for data
OUTPUT_PATH="$BASE_DIR/gbif/lookup_tables/2024-05-01/lookup_${GROUP}"
N_MAX_FILES_PER_BATCH=100

# Submit Spark job
slurm-spark-submit \
    --driver-memory "$SPARK_DRIVER_MEMORY" \
    --executor-memory "$SPARK_EXECUTOR_MEMORY" \
    "${REPO_ROOT}/src/processing/create_lookup_tbl.py" \
    "${GROUP}" \
    "${OUTPUT_PATH}" \
    --n_max_files_per_batch "${N_MAX_FILES_PER_BATCH}" \
    > "${REPO_ROOT}/logs/create_lookup_tbl.log"