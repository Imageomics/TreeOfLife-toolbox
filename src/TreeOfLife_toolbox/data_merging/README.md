# Data Merging Tool

This tool identifies and filters out duplicate images between a target dataset and a newly downloaded source dataset
based on image hash values. It's designed to ensure data uniqueness when integrating new data into an existing
repository.

## What it does

The tool performs the following steps:

1. Scans a target dataset (existing data collection)
2. Scans a source dataset (newly downloaded data)
3. Identifies duplicate images by comparing `hashsum_original` values
4. Creates a filter table of duplicates that can be used for further processing
5. Filters out duplicated entries from the source dataset

## Required fields in config

- `merge_target`: Path to the target dataset folder that will be checked against the new dataset

## Initial assumptions/preconditions

**Target dataset:**

- It follows the following structure:

  ```
    <dst_image_folder>/
    ├── server=<server1>
    │   ├── data_<uuid1>.parquet
    │   ├── data_<uuid2>.parquet
    │   └── ...
    └── server=<server2>
        └── ...
    ```
  
    - Entries in the target dataset are unique on `hashsum_original` column
    - Each file follows the `successes` scheme from `distributed-downloader`

**Source dataset:**

- It follows the following structure:

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

- Each file follows the `successes` scheme from `distributed-downloader`
- Entries in the source are unique on `hashsum_original` column

## Post conditions

After successful execution, the tool guarantees:

- A CSV filter table is created containing metadata about all duplicated entries, including their source and target
  UUIDs
- The duplicated records are accessible via the tool's filter table, showing the relationship between source and target
  entries
- The number of duplicated entries is logged for verification
- Processing is distributed across available nodes for efficient execution
- Original data files remain untouched - this is a non-destructive analysis

## Implementation Details

The tool consists of three main components:

1. **DataMergedDupCheckFilter** - Identifies duplicate records by comparing hashsums
2. **DataMergedDupCheckScheduleCreation** - Handles task scheduling for distributed processing
3. **DataMergedDupCheckRunner** - Executes the actual filtering process

The implementation relies on Apache Spark for efficient distributed data processing, making it suitable for large
datasets.