#!/bin/bash
#SBATCH --job-name create_support_set
#SBATCH --nodes=1
#SBATCH --mem=920G
#SBATCH --gpus-per-node=1
#SBATCH --partition=quad
#SBATCH --time=00:30:00
#SBATCH --account=YOUR_ACCOUNT

# Activate Python virtual environment
source /PATH/TO/ENVIRONMENT/CLIP_ascend/bin/activate

BASE_DIR=""  # Set your base path for data
SUPPORT_SET_URLS_PATH="$BASE_DIR/TreeOfLife-dev/data/gbif/support_set/museum_specimen_plant/support_set_urls.json"

python \
    "${REPO_ROOT}/src/processing/create_support_set.py" \
    "${SUPPORT_SET_URLS_PATH}" \
    > "${REPO_ROOT}/logs/create_support_set.log"