# EoL Rename Tool

This tool enhances the Encyclopedia of Life (EoL) dataset by adding source identifiers to downloaded images.

## Overview

The EoL Rename tool processes images downloaded using the `distributed-downloader` tool. It enriches the dataset by:

1. Reading image data from the downloaded images directory
2. Reading original batch data containing EoL content and page IDs
3. Merging these datasets on the "uuid" field
4. Creating a new "source_id" field by concatenating "EOL content ID" and "EOL page ID"
5. Saving the updated data back to the original parquet files

This process ensures that images can be traced back to their source EoL content and pages.

## Components

The tool consists of three main components:

- **EoLRenameFilter**: Registers the 'eol_rename' filter in the system
- **EoLRenameScheduleCreation**: Creates execution schedules for rename operations
- **EoLRenameRunner**: Executes the actual renaming process by adding source IDs to image data

## Configuration

No additional configuration fields are required beyond the standard TreeOfLife toolbox configuration:

- Standard path configurations for downloaded images and URL folders are used
- The system will automatically locate the required data based on server_name and partition_id

## Pre-conditions

The tool requires the following to be true before running:

- Images must be downloaded using the `distributed-downloader` tool
- No additional data processing should have been performed on the dataset
- The folder structure must follow the distributed-downloader's conventions:
  - Downloaded images stored in paths with `server_name` and `partition_id` partitions
  - Original batch data available in the URLs folder with similar partitioning

## Post-conditions

After running the tool:

- The source_id will be in the format `{EOL content ID}_{EOL page ID}`
- Original parquet files will be updated in-place with the new field
- No duplicate or additional files will be created
