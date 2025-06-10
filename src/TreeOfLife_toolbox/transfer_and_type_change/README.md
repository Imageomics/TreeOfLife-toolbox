# Transfer and Type Change Tool

## Summary

This tool was specifically created for transferring data from one place to another on research storage and changing the column type in certain chunks. It performs two main operations:

1. Transfers parquet files from a source location to a destination location
2. Converts the 'source_id' column from its original type to string type

The tool operates in three phases:

- **Filtering**: Processes the division CSV to identify which files need to be processed
- **Scheduling**: Creates a schedule for parallel processing by assigning files to worker ranks
- **Execution**: Runs multiple workers in parallel to read, convert, and write files

## Configuration Parameters

Required configuration fields:

- `src_path` - Path to the source data (absolute path)
- `dst_path` - Path to the destination data (absolute path)

## Division File

Before running the tool, you need to create a `divisions.csv` file that has the following columns:

- `division` - ID of the division (e.g. `0`, `1`, `2`, etc.)
- `source` - Source name of the data (is part of the path)
- `server` - Server name of the data (is part of the path)
- `file_name` - File name of the data (is part of the path)

The goal of this file is to divide the data into manageable chunks (smaller than 10TB) for efficient processing.

## Usage

```bash
python -m TreeOfLife_toolbox.transfer_and_type_change.main <config_path> transfer_and_type_change <divisions_path>
```

## Pre-conditions

- Source data should be in the following folder structure:

```
<src_path>/source=<source_name>/data/server=<server_name>/<file_name>
```

- The source files must be in parquet format
- Each file must have a 'source_id' column
- The division CSV file must be properly formatted

## Post-conditions

- Destination data will be in the following folder structure:

```
<dst_path>/source=<source_name>/server=<server_name>/<file_name>
```

- The 'source_id' column in all files will be converted to string type
- Original files will be removed after successful transfer and conversion
- Verification records will be created to track which files have been processed

## Special Submission Scripts

## tools_submit.sh

### Purpose

`tools_submit.sh` is designed for submitting tool-related jobs to Slurm with specific resource requirements for tooling
operations. It supports both regular and Spark-based tool submissions and handles job dependencies.

### Usage

```bash
./tools_submit.sh script tool_name seq_id [dependency] [--spark]
```

### Arguments

1. `script`: The script file to submit to Slurm
2. `tool_name`: The name of the tool to be run
3. `seq_id`: The sequence ID for the job's division in the `divisions.csv` file
4. `dependency`: (Optional) The job ID that this job depends on
5. `--spark`: (Optional) Flag indicating this is a Spark-based job

### Features

- Sets up the repository root environment variable (`REPO_ROOT`)
- Creates the logs directory automatically
- Handles job dependencies (if provided)
- Special handling for Spark jobs, which have different resource requirements
- For non-Spark jobs, applies tool-specific resource configurations

### Environment Variables Used

- `OUTPUT_TOOLS_LOGS_FOLDER`: Directory to store tool log files
- `TOOLS_MAX_NODES`: Maximum number of nodes for tools
- `TOOLS_WORKERS_PER_NODE`: Number of tool workers per node
- `TOOLS_CPU_PER_WORKER`: Number of CPUs per tool worker
- `ACCOUNT`: Slurm account to charge the job to

## Tools Slurm Script Architecture

### tools_scheduler.slurm

**Purpose**: Creates execution schedules for the tool workers based on filtered data.

**Key Components**:

- Runs on a single node
- Calls `transfer_and_type_change/scheduler.py` with the specified tool name and sequence ID
- Processes the CSV files produced by the filter step
- Assigns images to different worker processes to balance the load
- Typical run time is 5 minutes
- Creates schedule files that map partitions to worker ranks

**Example**:
For a size-based filter tool, the scheduler might group images by server name and partition ID (which corresponds to a
single parquet file) and assign these groups to different MPI ranks (e.g., worker 1 processes partitions 1,2,3,4).

### tools_worker.slurm

**Purpose**: Executes the actual tool processing using MPI parallelism.

**Key Components**:

- Runs on the configured number of nodes with specified worker distribution
- Calls `transfer_and_type_change/runner.py` with the specified tool name and sequence ID
- Reads the schedule created by the scheduler and processes assigned partitions
- Uses all allocated nodes for maximum parallelism
- Configures memory settings for optimal performance
- Typical run time is 3 hours
- Creates output files specific to the tool (e.g., resized images)

**Example**:
For an image resizing tool, each worker would load the images assigned to it from the schedule, resize them to the
specified dimensions, and save the results to the output location.

### tools_verifier.slurm

**Purpose**: Verifies the completion of the tool processing and updates status flags.

**Key Components**:

- Runs on a single node
- Calls `transfer_and_type_change/verification.py` with the specified tool name and sequence ID
- Checks if all scheduled tasks have been completed
- Updates the completion status in the tool's checkpoint file
- Typical run time is 5 minutes
- Sets the "completed" flag when all processing is done

**Example**:
For any tool, the verifier checks if all scheduled tasks have been processed successfully and marks the overall
operation as complete when verified.
