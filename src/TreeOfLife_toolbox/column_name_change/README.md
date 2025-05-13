# Column Name Change Tool

A distributed tool for renaming columns in Parquet files that follow the `distributed-downloader` directory structure.
This tool enables batch renaming of columns across multiple Parquet files in parallel using MPI, making it efficient for
large datasets.

## Overview

The Column Name Change tool provides a way to update column names in Parquet files without having to reload or
reconstruct the entire dataset. This is particularly useful when:

- Schema requirements change in downstream applications
- Column names need standardization across multiple datasets
- Fixing typos or inconsistencies in column naming
- Adapting to new naming conventions

## Implementation

The tool consists of three main components:

1. **Filter** (`ColumnNameChangeFilter`): Identifies all server_name/partition_id combinations that contain Parquet
   files needing column renaming.

2. **Scheduler** (`ColumnNameChangeScheduleCreation`): Distributes the workload across available workers using a
   round-robin approach for balanced processing.

3. **Runner** (`ColumnNameChangeRunner`): Performs the actual column renaming operation in parallel using MPI,
   processing each assigned partition.

## Required Configuration Fields

The tool requires the following configuration:

- Standard TreeOfLife-toolbox configurations for distributed processing
- `name_mapping`: A dictionary mapping old column names to new column names

Example configuration YAML:

```yaml
# Standard tool configuration
# ...

# Tool-specific configuration
name_mapping:
  old_column_name1: new_column_name1
  old_column_name2: new_column_name2
  # Add more mappings as needed
```

## Pre-conditions

- The input image directory must follow the distributed-downloader structure:

  ```
  <path_to_output_folder>/<images_folder>/
  ├── server_name=<server1>
  │   ├── partition_id=<id>
  │   │   ├── successes.parquet  # Contains data with columns to be renamed
  │   │   ├── errors.parquet
  │   │   └── completed
  │   └── partition_id=<id2>
  │       ├── ...
  └── server_name=<server2>
      └── ...
  ```

- The Parquet files must exist and be readable
- The columns specified in `name_mapping` must exist in the Parquet files (if a column doesn't exist, that specific
  mapping will be ignored)

## Post-conditions

After running this tool:

- The column names in all `successes.parquet` files will be changed according to the provided mapping
- File structure and other metadata remain unchanged
- Data content (rows and values) remain unchanged
- Original compression and partitioning are preserved
