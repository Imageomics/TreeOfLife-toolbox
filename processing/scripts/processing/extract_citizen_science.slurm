#!/bin/bash
#SBATCH --job-name extract_citizen_science
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=96
#SBATCH --time=08:00:00
#SBATCH --account=YOUR_ACCOUNT

# Cluster: Cardinal
module load spark/3.5.1 

# Define Spark configurations as strings
SPARK_DRIVER_MEMORY="64G"
SPARK_EXECUTOR_MEMORY="75G"
SPARK_EXECUTOR_CORES="12"  # Ensure this is a string
SPARK_EXECUTOR_INSTANCES="80"

# Submit Spark job
slurm-spark-submit \
    --driver-memory "$SPARK_DRIVER_MEMORY" \
    --executor-memory "$SPARK_EXECUTOR_MEMORY" \
    "${REPO_ROOT}/src/processing/extract_citizen_science.py" \
    > "${REPO_ROOT}/logs/extract_citizen_science.log"