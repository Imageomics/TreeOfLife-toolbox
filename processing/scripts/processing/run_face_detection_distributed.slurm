#!/bin/bash
#SBATCH --job-name=run_face_detection_distributed
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2            # 2 tasks per node
#SBATCH --gpus-per-task=1              # 1 GPU per task
#SBATCH --mem=470G                     
#SBATCH --partition=nextgen
#SBATCH --time=12:00:00
#SBATCH --account=YOUR_ACCOUNT

# Load modules and activate your Python env
module load cuda/12.4.1
source PATH/TO/ENVIRONMENT/CLIP_ascend/bin/activate
which python
echo "Python virtual environment activated"

BASE_DIR=""     # Set your base path for data
TARGET_DIR="$BASE_DIR/TreeOfLife/data/source=gbif"
OUTPUT_DIR="$BASE_DIR/TreeOfLife/face_detection_result/source=gbif/batch_left_second_attempt"

# Launch script once per task using srun
srun python "${REPO_ROOT}/src/processing/run_face_detection_distributed.py" \
    --target_dir "${TARGET_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    > "${REPO_ROOT}/logs/run_face_detection_distributed.log"
