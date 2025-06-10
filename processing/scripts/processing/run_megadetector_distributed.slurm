#!/bin/bash
#SBATCH --job-name=run_megadetector_distributed
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2            # 2 tasks per node
#SBATCH --gpus-per-task=1              # 1 GPU per task
#SBATCH --mem=470G                     
#SBATCH --partition=nextgen
#SBATCH --time=4:00:00
#SBATCH --account=YOUR_ACCOUNT


# Load modules and activate Python env
module load cuda/12.4.1
source PATH/TO/ENVIRONMENT/CameraTraps_ascend/bin/activate
which python
echo "Python virtual environment activated"

BASE_DIR=""     # Set your base path for data
TARGET_DIR="$BASE_DIR/TreeOfLife/image_lookup/2024-05-01/gbif/multi_images_camera_trap"
OUTPUT_DIR="$BASE_DIR/TreeOfLife/image_lookup/2024-05-01/gbif/multi_images_camera_trap_detection_result_second_attempt"

srun python "${REPO_ROOT}/src/processing/run_megadetector_distributed.py" \
    --target_dir "${TARGET_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    > "${REPO_ROOT}/logs/run_megadetector_distributed.log"

