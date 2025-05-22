# TreeOfLife toolbox - Scripts Documentation

This document provides detailed explanations of the scripts in the `scripts` folder of the TreeOfLife-toolbox
package. These scripts are used to submit jobs to Slurm and execute various tasks of this package.

## Submission Scripts

## tools_submit.sh

### Purpose

`tools_submit.sh` is designed for submitting tool-related jobs to Slurm with specific resource requirements for tooling
operations. It supports both regular and Spark-based tool submissions and handles job dependencies.

### Usage

```bash
./tools_submit.sh script tool_name [dependency] [--spark]
```

### Arguments

1. `script`: The script file to submit to Slurm
2. `tool_name`: The name of the tool to be run
3. `dependency`: (Optional) The job ID that this job depends on
4. `--spark`: (Optional) Flag indicating this is a Spark-based job

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

### Example

```bash
# Submit a regular tool job
./tools_submit.sh /path/to/tool_script.slurm resize

# Submit a tool job with a dependency
./tools_submit.sh /path/to/tool_script.slurm resize 12345

# Submit a Spark-based tool job
./tools_submit.sh /path/to/spark_tool_script.slurm resize --spark

# Submit a Spark-based tool job with a dependency
./tools_submit.sh /path/to/spark_tool_script.slurm resize 12345 --spark
```

## Tools Slurm Script Architecture

The TreeOfLife-toolbox package includes several specialized Slurm scripts to handle different aspects of the tools
pipeline. These scripts are designed to work with the tools framework which processes downloaded images in various ways.

> [!IMPORTANT]
> ### Job Output Format Requirement
>
> All Slurm scripts must output the job ID of the submitted job in a specific format. The job ID must be the last item
> on the line and separated by a space:
>
> ```
> {anything} {id}
> ```
> 
> This format is essential for job dependency tracking.

### tools_filter.slurm

**Purpose**: Performs the first step in the tool pipeline, filtering images based on specific criteria.

**Key Components**:

- Uses Spark for distributed processing of large dataset files
- Calls `main/filter.py` with the specified tool name
- Creates CSV files containing references to images that match the filtering criteria
- The Typical run time is 1 hour
- Requires significant memory for driver (110GB) and executors (64GB)

**Example**:
For a size-based filter tool, this script would identify all images smaller than a threshold size and write their UUIDs,
server names, and partition IDs to CSV files.

### tools_scheduler.slurm

**Purpose**: Creates execution schedules for the tool workers based on filtered data.

**Key Components**:

- Runs on a single node
- Calls `main/scheduler.py` with the specified tool name
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
- Calls `main/runner.py` with the specified tool name
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
- Calls `main/verification.py` with the specified tool name
- Checks if all scheduled tasks have been completed
- Updates the completion status in the tool's checkpoint file
- Typical run time is 5 minutes
- Sets the "completed" flag when all processing is done

**Example**:
For any tool, the verifier checks if all scheduled tasks have been processed successfully and marks the overall
operation as complete when verified.

### Environment Configuration

All tools scripts share similar environment configuration:

- They rely on Intel MPI 2021.10 for communication
- They use Miniconda3 23.3.1 with Python 3.10
- They set PYTHONPATH to include the repository source directories

### Unique Aspects

- **tools_filter.slurm**: Uses Spark instead of MPI for distributed processing
- **tools_worker.slurm**: Uses the full configured node count and workers per node
- **tools_scheduler.slurm** and **tools_verifier.slurm**: Run on a single node as they perform organizational tasks

### Resource Management

The tools scripts use environment variables to determine resource allocation:

- `TOOLS_MAX_NODES`: Number of nodes for tool workers
- `TOOLS_WORKERS_PER_NODE`: Number of tool workers per node
- `TOOLS_CPU_PER_WORKER`: Number of CPUs per tool worker

These values are set from the configuration file and passed through the environment by the submission scripts.
