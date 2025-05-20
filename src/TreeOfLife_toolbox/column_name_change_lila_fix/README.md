# Column Name Change Lila Fix

A specialized tool built to correct column naming errors in Lila BC dataset parquet files.

## Overview

This tool fixes a specific issue where parquet files from the `storage.googleapis.com` server in the Lila BC dataset
have incorrect column names (`uuid_y` instead of `uuid` and `source_id_y` instead of `source_id`). The tool:

1. Filters for files only from the `storage.googleapis.com` server
2. Creates a schedule to distribute work across MPI workers
3. Processes each file by renaming the columns and saving to a new location

## Configuration Requirements

### Required Config Fields

- `uuid_table_path`: Path to the CSV file containing the UUID table with file paths to process

## Prerequisites

Before running this tool:

1. The UUID table must exist at the specified path
2. The table must contain at least the following columns:
    - `server`: Used to filter for only `storage.googleapis.com` entries
    - `path`: The full path to the parquet file to be processed
3. Original parquet files must be accessible at the paths specified in the UUID table
4. The worker nodes must have sufficient permissions to read source files and write to the destination folder

## Process Flow

1. **Filtering**: The filter component extracts paths from the UUID table, keeping only those from the
   `storage.googleapis.com` server
2. **Scheduling**: The scheduler distributes the paths across available worker nodes
3. **Processing**: Each worker:
    - Loads the assigned parquet file
    - Renames the columns according to the mapping:
        - `uuid_y` → `uuid`
        - `source_id_y` → `source_id`
    - Saves the corrected file to a new location with zstd compression

## Output and Post-conditions

After successful execution:

1. Corrected parquet files will be saved to:
   `/fs/scratch/PAS2136/gbif/processed/lilabc/name_fix/server=storage.googleapis.com/`

2. The directory structure of the output will preserve the original filenames

3. Each processed file will have correctly named columns:
    - `uuid` (previously `uuid_y`)
    - `source_id` (previously `source_id_y`)
    - All other columns remain unchanged

4. A verification table will be created in the tool's directory, tracking which files were successfully processed

5. The tool's checkpoint will be marked as completed when all files have been processed

## Limitations

- This tool can only process files from the `storage.googleapis.com` server
- The column mapping is hardcoded to fix specifically `uuid_y` and `source_id_y`
- The output path is hardcoded to `/fs/scratch/PAS2136/gbif/processed/lilabc/name_fix/server=storage.googleapis.com/`
- There is a 150-second time limit for processing each file

> ⚠️ **Note**: This is a specialized tool built for a specific dataset issue. It should not be used for other cases
> without code modifications.
