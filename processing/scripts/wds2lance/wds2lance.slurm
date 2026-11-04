#!/bin/bash
#SBATCH --job-name wds2lance
#SBATCH --mem=96G
#SBATCH --time=01:00:00

# Converts each split's webdataset shards into <OUT_DIR>/<split>.lance (delete it first to convert again).

# Activate a Python environment with processing/requirements_wds2lance.txt installed
source PATH/TO/ENVIRONMENT/bin/activate

BASE_DIR=""  # Set your base path for data
WDS_DIR="$BASE_DIR/path/to/webdataset"  # contains <split>/*.tar
OUT_DIR="$BASE_DIR/path/to/lance"
SPLITS="train val"
CONVERTER="${REPO_ROOT}/src/wds2lance/wds_to_lance.py"

set -euo pipefail
for split in $SPLITS; do
    python "$CONVERTER" "$WDS_DIR/$split" "$OUT_DIR/$split.lance"
done
