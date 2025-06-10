#!/bin/bash
#SBATCH --job-name extract_museum_specimen
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=96
#SBATCH --time=06:00:00
#SBATCH --account=YOUR_ACCOUNT

module load spark/3.5.1 
FLAG_REPARTITION="--repartition"

slurm-spark-submit \
    "${REPO_ROOT}/src/processing/extract_museum_specimen.py" \
    >"${REPO_ROOT}/logs/extract_museum_specimen.log"