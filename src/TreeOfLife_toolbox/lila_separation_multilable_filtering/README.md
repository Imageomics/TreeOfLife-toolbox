# LILA Separation Multilabel Filtering Tool

## Overview

This tool extracts and processes multi-labeled images from the LILA (LILA BC - Labeled Information Library of
Alexandria: Biodiversity Catalog) dataset. Multi-labeled images are those containing multiple objects of interest, each
with its own label. The tool creates a new dataset containing only these multi-labeled images with proper metadata,
effectively separating them from single-labeled images for specialized analysis or training.

The workflow consists of three main components:

1. **Filter (`LilaSeparationFilter`)**: Identifies multi-labeled images by joining metadata with a provided multi-label
   entries file.
2. **Scheduler (`LilaSeparationScheduleCreation`)**: Creates a processing schedule to distribute work across compute
   nodes.
3. **Runner (`LilaSeparationRunner`)**: Processes images according to the schedule, extracting and storing multi-labeled
   images in a new location.

## Configuration Requirements

The following fields must be specified in the configuration file:

- `new_images_path`: Destination path where processed multi-labeled images will be stored
- `new_urls_folder`: Destination path where metadata/URLs for multi-labeled images will be stored
- `multilabel_data_path`: Path to CSV file containing only multi-label entries

## Pre-conditions

For the tool to work correctly:

- The input `multilabel_data_path` CSV file must:
    - Contain only multi-label entries
    - Include both `uuid` and `partition_id` columns
    - Have the `partition_id` already repartitioned for the new dataset

- The original LILA dataset must:
    - Follow the distributed-downloader format
    - Contain the original images referenced in the multi-label CSV

## Post-conditions

After successful execution:

- A new dataset containing only multi-labeled images will be created at `new_images_path`
- Corresponding metadata will be stored at `new_urls_folder`
- The new dataset will follow the distributed-downloader format
