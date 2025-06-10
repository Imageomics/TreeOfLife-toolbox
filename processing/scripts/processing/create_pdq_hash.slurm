#!/bin/bash
#SBATCH --job-name create_pdq_hash
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=96
#SBATCH --time=50:00:00
#SBATCH --account=YOUR_ACCOUNT

module load spark/3.5.1 
module load intel

source PATH/TO/ENVIRONMENT/dml/bin/activate
which python
echo "Python virtual environment activated"

# Define Spark configurations as strings
SPARK_DRIVER_MEMORY="64G"
SPARK_EXECUTOR_MEMORY="75G"

BASE_DIR=""  # Set your base path for data
TARGET_DIR="$BASE_DIR/TreeOfLife/data/source=gbif"
OUTPUT_DIR="$BASE_DIR/TreeOfLife/pdq_hash/source=gbif"


# Submit Spark job
slurm-spark-submit \
    --driver-memory "$SPARK_DRIVER_MEMORY" \
    --executor-memory "$SPARK_EXECUTOR_MEMORY" \
    "${REPO_ROOT}/src/processing/create_pdq_hash.py" \
    --target_dir "${TARGET_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --processed_files_log_path "${OUTPUT_DIR}/processed_files.log" \
    --batch_size "30" \
    > "${REPO_ROOT}/logs/create_pdq_hash.log"