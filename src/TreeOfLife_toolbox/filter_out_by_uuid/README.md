# Filter Out By UUID Tool

## Summary

The `filter_out_by_uuid` tool is designed to remove specific entries from a dataset based on their UUIDs. It works by:

1. **Filtering**: Loading a table of UUIDs that need to be removed and a lookup table that maps UUIDs to file paths
2. **Scheduling**: Creating a work schedule based on the file paths that need processing
3. **Running**: Processing the files in parallel using MPI, removing entries with matching UUIDs

This tool is particularly useful when you need to clean a dataset by removing specific entries identified by their UUID.

## Required Configuration

In addition to the standard configuration fields required by all tools, this tool requires the following specific
fields:

* `uuid_table_path` - Path to the table containing UUIDs that should be filtered out
* `look_up_table_path` - Path to the lookup table with `uuid` and `path` columns, where `path` points to the file
  containing that UUID

## Pre-conditions

The tool requires the following to be in place before execution:

- A table with UUIDs to filter out (at the path specified by `uuid_table_path`). This table must have a `uuid` column.
- A lookup table with UUID to file path mapping (at the path specified by `look_up_table_path`). This table must have at
  least `uuid` and `path` columns, where `path` contains the absolute path to the file.
- The files referenced in the lookup table must exist and be in parquet format.
- The files must have a column named `uuid` that can be used for filtering.

## Execution Process

1. The filter step joins the UUID table with the lookup table to produce a list of UUIDs and corresponding file paths
2. The scheduler creates a work distribution plan based on the unique file paths that need processing
3. The runner processes the files in parallel, reading each file, filtering out rows with matching UUIDs, and writing
   the filtered data back to the same file

## Post-conditions

After successful execution of the tool:

- All entries with UUIDs listed in the input UUID table will be removed from the dataset files
- The original files will be replaced with filtered versions (retaining the same path and name)
- Entries that do not match the filter criteria remain unchanged
- A verification table is created to track which files have been successfully processed

## Notes

- This tool was originally designed for the Tree Of Life structured dataset, but should also work with
  `distributed-downloader` structured datasets.
- The tool uses the pandas `read_parquet` function with filters, which allows efficient filtering without loading the
  entire dataset into memory.
- Files are saved with zstd compression (level 3) to balance between compression ratio and performance.
- If a file is completely filtered out (i.e., all entries match the filter criteria), an empty file will be written
  back.