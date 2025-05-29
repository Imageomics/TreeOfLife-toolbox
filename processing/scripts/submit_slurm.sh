#!/bin/bash

SCRIPTS_DIR=$(dirname "$(realpath "$0")")
REPO_ROOT=$(dirname "$(realpath "${SCRIPTS_DIR}")")
export REPO_ROOT
LOGS_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOGS_DIR}"

# Check if any arguments were passed
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 script1 [script2 script3 ...]"
    exit 1
fi

# Loop over all arguments
for script in "$@"; do
    # Check if the file exists
    if [ ! -f "$script" ]; then
        echo "Error: File '$script' not found"
        continue
    fi
    filename=$(basename "$script")
    ext="${filename##*.}"
    base_filename=$(basename "${filename}" ."${ext}")

    # Submit the script to Slurm
    sbatch --output="${REPO_ROOT}/logs/${base_filename}.out" --error="${REPO_ROOT}/logs/${base_filename}.err" "${script}"
done
