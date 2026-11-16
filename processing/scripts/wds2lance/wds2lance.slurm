#!/bin/bash
#SBATCH --job-name wds2lance
#SBATCH --mem=96G
#SBATCH --time=01:00:00

# Converts each split's webdataset shards into <OUT_DIR>/<split>.lance.
# A split that is already converted is skipped, so the job can be submitted again after a failure
# (delete the failed split's <split>.lance first).

# Activate a Python environment with processing/requirements_wds2lance.txt installed
source PATH/TO/ENVIRONMENT/bin/activate

BASE_DIR=""  # Set your base path for data
WDS_DIR="$BASE_DIR/path/to/webdataset"  # contains <split>/*.tar
OUT_DIR="$BASE_DIR/path/to/lance"
SPLITS="train val"
CONVERTER="${REPO_ROOT}/src/wds2lance/wds_to_lance.py"

set -euo pipefail
for split in $SPLITS; do
    # the manifest is written last, so only a finished split has one
    if [ -d "$OUT_DIR/$split.lance" ] && [ -f "$OUT_DIR/$split.lance.manifest.json" ]; then
        echo "$split.lance is already converted, skipping"
        continue
    fi
    python "$CONVERTER" "$WDS_DIR/$split" "$OUT_DIR/$split.lance"
done
