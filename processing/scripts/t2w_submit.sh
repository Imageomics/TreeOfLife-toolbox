#!/bin/bash

set -e

SCRIPTS_DIR=$(dirname "$(realpath "$0")")
REPO_ROOT=$(dirname "$(realpath "${SCRIPTS_DIR}")")
export REPO_ROOT
logs_dir="${OUTPUT_T2W_LOGS_FOLDER}"
mkdir -p "${logs_dir}"

# Check if any arguments were passed
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 script1 job_index [dependency] [--spark]"
    exit 1
fi

script=$1
if [ ! -f "$script" ]; then
  echo "Error: File '$script' not found"
  exit 1
fi

filename=$(basename "$script")
ext="${filename##*.}"
base_filename=$(basename "${filename}" ."${ext}")
job_index=$2
dependency=""
spark_flag=""

if [ "$3" == "--spark" ]; then
  spark_flag="--spark"
  dependency="$4"
else
  dependency="$3"
  if [ "$4" == "--spark" ]; then
    spark_flag="--spark"
  fi
fi

sbatch_cmd="sbatch --output=\"${logs_dir}/${base_filename}.out\" --error=\"${logs_dir}/${base_filename}.err\" --nodes=${T2W_MAX_NODES_PER_RUNNER}"

if [ -n "$dependency" ]; then
  sbatch_cmd+=" --dependency=afterany:${dependency}"
fi

if [ -z "$spark_flag" ]; then
  sbatch_cmd+=" --ntasks-per-node=${T2W_WORKERS_PER_NODE} --cpus-per-task=${T2W_CPU_PER_WORKER}"
fi

sbatch_cmd+=" --account=${ACCOUNT} ${script} ${job_index}"

eval "$sbatch_cmd"
