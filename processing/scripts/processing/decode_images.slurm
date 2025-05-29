#!/bin/bash
#SBATCH --job-name decode_images
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=96
#SBATCH --time=01:00:00
#SBATCH --account=YOUR_ACCOUNT

# Cluster: Cardinal
module load spark/3.5.1 

# Activate Python virtual environment
source PATH/TO/ENVIRONMENT/dm/bin/activate

# Define Spark configurations as strings
SPARK_DRIVER_MEMORY="64G"
SPARK_EXECUTOR_MEMORY="75G"

BASE_DIR=""  # Set your base path for data
INPUT_PATH="$BASE_DIR/gbif/image_lookup/test_bioclip"
OUTPUT_PATH="$BASE_DIR/gbif/image_lookup/test_bioclip_decoded"
# INPUT_PATH="$BASE_DIR/gbif/image_lookup/multi_images_camera_trap_15"
# OUTPUT_PATH="$BASE_DIR/gbif/image_lookup/multi_images_camera_trap_15_decoded"

PARTITION_SIZE=100
FORMAT="jpeg"
COMPRESS_FLAG="--compress"

# Submit Spark job
slurm-spark-submit \
    --driver-memory "$SPARK_DRIVER_MEMORY" \
    --executor-memory "$SPARK_EXECUTOR_MEMORY" \
    "${REPO_ROOT}/src/processing/decode_images.py" \
    "${INPUT_PATH}" \
    "${OUTPUT_PATH}" \
    --format "${FORMAT}" \
    --partition_size "${PARTITION_SIZE}" \
    > "${REPO_ROOT}/logs/decode_images.log"