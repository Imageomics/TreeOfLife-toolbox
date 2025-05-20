# MAM ANSP Duplication Fix Tool

## Overview

This tool addresses a specific duplication issue found in data from the "mam.ansp.org" server (from the GBIF source) in
the Tree of Life dataset. It identifies, processes, and removes duplicate UUID entries within parquet files, ensuring
data integrity and consistency.

The tool consists of three main components:

1. **Filter (MamAnspFixFilter)**: Identifies files from the mam.ansp.org server that need deduplication based on a
   provided UUID table.
2. **Scheduler (MamAnspFixScheduleCreation)**: Distributes the workload of file processing across available workers.
3. **Runner (MamAnspFixRunner)**: Performs the actual deduplication process by reading each file, removing duplicate
   UUIDs, and saving the cleaned data to a specified location.

## Configuration Requirements

The following fields must be included in the configuration file:

* `uuid_table_path`: Path to the CSV file containing the table of UUIDs with information about duplicated entries. This
  file must include "server" and "path" columns.
* `save_path_folder`: Directory where the deduplicated parquet files will be saved.

## Prerequisites (Pre-conditions)

Before running this tool, ensure:

- The dataset follows the Tree of Life format structure
- The UUIDs table contains accurate information about mam.ansp.org server entries
- The `uuid_table_path` CSV file contains at minimum these columns: "server" and "path"

## Guarantees (Post-conditions)

After successful execution:

- The dataset maintains the Tree Of Life format
- Duplicate UUID entries in files from mam.ansp.org server have been removed
- The original files remain untouched
