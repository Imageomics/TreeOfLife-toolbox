#!/bin/bash
#SBATCH --job-name=run_clip_distributed
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
TARGET_DIR="$BASE_DIR/TreeOfLife/image_lookup/2024-05-01/gbif/museum_specimen_all_repartitioned/category=Plant"
OUTPUT_DIR="$BASE_DIR/TreeOfLife/image_lookup/2024-05-01/gbif/museum_specimen_all_classified/category=Plant"
CLASS_EMBS_DICT_PATH="$BASE_DIR/TreeOfLife/image_lookup/2024-05-01/gbif/support_set/museum_specimen_plants/support_set_clustered_embeddings.pt"


# Launch script once per task using srun
srun python "${REPO_ROOT}/src/processing/run_clip_distributed.py" \
    --target_dir "${TARGET_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --class_embeddings_dict_path "${CLASS_EMBS_DICT_PATH}" \
    > "${REPO_ROOT}/logs/run_clip_distributed.log"
