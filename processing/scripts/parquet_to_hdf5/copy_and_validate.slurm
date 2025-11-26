#!/bin/bash
#SBATCH --job-name copy_val_tol_hdf5
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# TreeOfLife Copy and Validate HDF5 Hybrid Data Slurm Wrapper
# Submits copy_and_validate.sh as a Slurm job with parameterized arguments

# Function to display usage
usage() {
    cat << EOF
Usage: sbatch [SLURM_OPTIONS] $0 --source-root SOURCE_ROOT --source-group SOURCE_GROUP --dest-root DEST_ROOT [--email EMAIL]

Required arguments:
    --source-root    Source root directory (e.g., /path/to/data)
    --source-group   Source group name (e.g., gbif_group_20)
    --dest-root      Destination root directory (e.g., /path/to/data)

Optional arguments:
    --email          Email address for success notification

SLURM options (use before script name):
    --account ACCT   SLURM account
    --output PATH    Output log file (e.g., /path/to/job-%j.out)
    --error PATH     Error log file (e.g., /path/to/job-%j.err)

EOF
}

# Parse command line arguments
SOURCE_ROOT=""
SOURCE_GROUP=""
DEST_ROOT=""
EMAIL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --source-root)
            SOURCE_ROOT="$2"
            shift 2
            ;;
        --source-group)
            SOURCE_GROUP="$2"
            shift 2
            ;;
        --dest-root)
            DEST_ROOT="$2"
            shift 2
            ;;
        --email)
            EMAIL="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: Unknown argument $1"
            usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$SOURCE_ROOT" || -z "$SOURCE_GROUP" || -z "$DEST_ROOT" ]]; then
    echo "Error: Missing required arguments"
    echo ""
    usage
    exit 1
fi

# Validate that source directory exists
SOURCE_BASE="$SOURCE_ROOT/source=$SOURCE_GROUP"
if [[ ! -d "$SOURCE_BASE" ]]; then
    echo "Error: Source directory does not exist: $SOURCE_BASE"
    exit 1
fi

# Get the script directory - use SLURM_SUBMIT_DIR which is where sbatch was called from
# Since sbatch is called with: processing/scripts/parquet_to_hdf5/copy_and_validate.slurm
# The script directory is: SLURM_SUBMIT_DIR/processing/scripts/parquet_to_hdf5/
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/processing/scripts/parquet_to_hdf5"

echo "============================================="
echo "SLURM TreeOfLife Copy and Validate Job"
echo "============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Source Root: $SOURCE_ROOT"
echo "Source Group: $SOURCE_GROUP"
echo "Dest Root: $DEST_ROOT"
echo "Email: ${EMAIL:-"(none)"}"
echo "============================================="

# Set up environment - load any required modules
# (Add module loads here if needed for file operations)

# Execute the copy and validate script with parameters
bash "$SCRIPT_DIR/copy_and_validate.sh" \
    --source-root "$SOURCE_ROOT" \
    --source-group "$SOURCE_GROUP" \
    --dest-root "$DEST_ROOT" \
    ${EMAIL:+--email "$EMAIL"}

# Capture the exit status of the copy_and_validate.sh script
EXIT_STATUS=$?

echo ""
echo "============================================="
if [[ $EXIT_STATUS -eq 0 ]]; then
    echo "SUCCESS: Copy and validation completed successfully"
    echo "Finished: $(date)"
else
    echo "FAILURE: Copy and validation failed with exit code $EXIT_STATUS"
    echo "Finished: $(date)"
fi
echo "============================================="

exit $EXIT_STATUS