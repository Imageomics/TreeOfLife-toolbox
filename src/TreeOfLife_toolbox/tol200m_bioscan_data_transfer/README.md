# BIOSCAN Data Transfer Tool

## Purpose

This tool takes data from BIOSCAN format and converts it into the Tree of Life (TOL) format, transferring it to the "
final" location at the same time. The tool processes image data from BIOSCAN, resizes images if necessary, calculates
hashsums, and organizes them according to the TOL data structure.

## How It Works

The tool operates in three main phases:

1. **Filtering Phase** (`ToLBioscanDataTransferFilter`):
    - Loads input data from BIOSCAN TSV file
    - Joins it with provenance data to enrich metadata
    - Partitions the data into manageable batches for parallel processing
    - Saves partitioned data as CSV files

2. **Scheduling Phase** (`ToLBioscanDataTransferScheduleCreation`):
    - Creates a schedule CSV file mapping source partition paths to destination paths
    - Assigns UUIDs to each partition for unique identification in the TOL structure

3. **Transfer Phase** (`ToLBioscanDataTransferRunner`):
    - Uses MPI (Message Passing Interface) for parallel processing
    - For each scheduled partition:
        - Reads the CSV files in the partition
        - Processes the associated images (resizing if needed)
        - Calculates original and resized hashsums
        - Converts data into the TOL format
        - Saves data as compressed Parquet files at the destination paths
    - Maintains a verification file to track progress and enable resuming interrupted transfers

## Configuration Parameters

### New Parameters Required

- `provenance_path`: Path to a provenance Parquet file containing source metadata
- `path_to_tol_folder`: Path to where you save TOL data folder (a.k.a. `<output_dir>/data`)
- `bioscan_image_folder`: Path to where you unzipped BIOSCAN images (up to `original_full` folder)

### Existing Parameters

- `path_to_input`: Path to the TSV file from BIOSCAN
- `urls_folder`: Folder where the partitioned BIOSCAN TSV files will be temporarily stored
- `batch_size` (optional): Number of records per batch (default: 10,000)
- `image_size` (optional): Maximum dimension for resized images (default: 720)

## Required new scripts

### tools_worker.slurm

**Purpose**: Executes the actual tool processing using MPI parallelism.

**Key Components**:

- Runs on the configured number of nodes with specified worker distribution
- Calls `main/runner.py` with the specified tool name
- Reads the schedule created by the scheduler and processes assigned partitions
- Uses all allocated nodes for maximum parallelism
- Configures memory settings for optimal performance
- Typical run time is 3 hours
- Creates output files specific to the tool (e.g., resized images)
- **Important**: The script should be run through `mpi4py.futures` to ensure proper parallel execution.

```bash
mpirun -n <num_workers> python -m mpi4py.futures tools_worker.slurm
```

## Pre-conditions

- You have downloaded the BIOSCAN dataset and unzipped it to an accessible location
- You have a TSV file from BIOSCAN with required metadata
- You have a provenance Parquet file with matching source IDs
- All paths specified in the configuration are valid and accessible

## Post-conditions

- Partitioned data will be saved in `path_to_tol_folder` in the same structure as the original Tree Of Life dataset
- Each image will have both original and resized hashsums calculated
- Images exceeding the maximum size will be resized while preserving aspect ratio
- All data will be saved in Parquet format with zstd compression for efficient storage
- A verification file will track successfully processed partitions
