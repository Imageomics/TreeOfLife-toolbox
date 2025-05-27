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
