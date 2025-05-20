# Research Filtering Tool

## Overview

The Research Filtering Tool allows filtering of TreeOfLife datasets based on the `basisOfRecord` field from occurrence
data. This tool is designed to selectively remove entries from the dataset where the `basisOfRecord` matches specific
criteria defined in the configuration.

## How It Works

The tool operates in three phases:

1. **Filtering Phase**: Reads occurrence data and the main dataset, filters occurrences based on the specified
   `basisOfRecord` value, and creates a filter table identifying entries to be removed.

2. **Scheduling Phase**: Creates a schedule for distributed processing of the filtered data, organized by file paths.

3. **Running Phase**: Applies the filtering operation across distributed nodes, removing entries from parquet files
   based on UUIDs identified in the filtering step.

## Configuration Requirements

The following fields must be specified in the configuration file:

* `occurrences_path`: Path to the occurrences table containing `gbifID` and `basisOfRecord` fields
* `data_path`: Path to the TreeOfLife dataset root directory
* `basis_of_record`: Value or pattern to match in the `basisOfRecord` field for entries that should be filtered out
* `save_path_folder`: Path where filtered data should be saved (if applicable)

## Initial Assumptions / Preconditions

- The dataset must follow the TreeOfLife format structure
- The occurrences data must include `gbifID` and `basisOfRecord` fields
- The dataset must be organized in parquet files following the pattern: `/source=*/server=*/data_*.parquet`
- Each data file must contain at least `uuid` and `source_id` fields

## Post-conditions

After successful execution:

- Parquet files in the dataset will be filtered to exclude entries where the `basisOfRecord` matches the specified
  criteria
- Original dataset files will be overwritten with filtered versions
- Entries with matching criteria will be completely removed from the dataset
- All files processed will be logged in the verification folder
- Compression will be applied using zstd level 3 to maintain efficient storage
