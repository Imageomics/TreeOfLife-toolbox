# Data Transfer Module

## Summary Description

The Data Transfer module provides functionality for efficiently transferring parquet files from source locations to
destination locations with integrity verification. It operates in three phases:

1. **Filtering**: The `DataTransferFilter` class identifies parquet files that need to be transferred and generates
   unique destination paths.
2. **Scheduling**: The `DataTransferScheduleCreation` class creates a comprehensive schedule of all file transfers by
   combining filter tables.
3. **Execution**: The `DataTransferRunner` class performs the actual file transfers with parallel processing using MPI,
   including MD5 hash verification.

The module handles two types of files:

- **Success files** (`successes.parquet`): Contains data about successfully downloaded images
- **Error files** (`errors.parquet`): Contains data about download errors

Each file is transferred to a structured destination path that includes the server name and a randomly generated UUID to
ensure uniqueness.

## Configuration Requirements

Add these values to your config file:

- `dst_image_folder`: The destination folder where the image (successes) files will be copied
- `dst_error_folder`: The destination folder where the error files will be copied

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

## Expected Input Structure

The source directory structure should follow this pattern:

```
<path_to_output_folder>/<images_folder>/
├── server_name=<server1>
│   ├── partition_id=<id>
│   │   ├── successes.parquet
│   │   ├── errors.parquet
│   │   └── completed
│   └── partition_id=<id2>
│       ├── ...
└── server_name=<server2>
    └── ...
```

Important components:

- Source folders are partitioned by `server_name=*` and then `partition_id=*`
- Each partition contains:
    - `successes.parquet`: Successfully downloaded images
    - `errors.parquet`: Errors produced during download
    - `completed`: Empty marker file indicating the download is complete for this partition

## Output Structure

Files will be transferred to a simplified structure:

```
<dst_image_folder>/
├── server=<server1>
│   ├── data_<uuid1>.parquet
│   ├── data_<uuid2>.parquet
│   └── ...
└── server=<server2>
    └── ...

<dst_error_folder>/
├── server=<server1>
│   ├── errors_<uuid1>.parquet
│   ├── errors_<uuid2>.parquet
│   └── ...
└── server=<server2>
    └── ...
```

Key aspects of the output structure:

- Files are organized by server name (with special characters like ':' replaced with '_')
- Each file has a unique UUID to prevent collisions
- Success files are prefixed with `data_` and error files with `errors_`
- The parquet file format and internal structure from the source is preserved
