#!/bin/bash
#SBATCH --job-name extract_images
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=96
#SBATCH --time=00:30:00
#SBATCH --account=YOUR_ACCOUNT

module load spark/3.5.1 

# Define Spark configurations as strings
SPARK_DRIVER_MEMORY="64G"
SPARK_EXECUTOR_MEMORY="75G"

BASE_DIR=""  # Set your base path for data
LOOKUP_TBL_PATH="$BASE_DIR/gbif/lookup_tables/2024-05-01/lookup_multi_images_camera_trap_15"
OUTPUT_PATH="$BASE_DIR/gbif/image_lookup/multi_images_camera_trap_15"


# Submit Spark job
slurm-spark-submit \
    --driver-memory "$SPARK_DRIVER_MEMORY" \
    --executor-memory "$SPARK_EXECUTOR_MEMORY" \
    "${REPO_ROOT}/src/processing/extract_images.py" \
    "${LOOKUP_TBL_PATH}" \
    "${OUTPUT_PATH}" \
    > "${REPO_ROOT}/logs/extract_images.log"