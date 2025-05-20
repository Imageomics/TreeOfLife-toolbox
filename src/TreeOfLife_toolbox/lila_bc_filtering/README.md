# LILA Biodiversity Catalog Filtering Tool

## Overview

The LILA BC Filtering tool is a specialized component of the TreeOfLife toolbox designed to filter out images from a
dataset based on their labels. Specifically, it removes images whose original labels match those specified in an
exclusion list. This tool is primarily built to work with the LILA (Labeled Information Library of Alexandria)
Biodiversity Catalog dataset, but can be applied to any dataset adhering to the proper format requirements.

## How It Works

The tool operates in three sequential stages:

1. **Filtering (LilaBCFilter)**:
    - Loads image data from parquet files
    - Loads original labels from the URLs table
    - Identifies images with labels matching the exclusion list
    - Creates a filter table containing UUIDs of images to be removed

2. **Scheduling (LilaBCScheduleCreation)**:
    - Creates a work distribution schedule for parallel processing
    - Assigns batches of images to different workers

3. **Running (LilaBCRunner)**:
    - Executes the actual filtering operation using MPI
    - Removes matched images from the dataset
    - Retains the same directory structure and filenames

## Required Configuration

### Mandatory Config Fields

- `path_to_excluding_labels`: Path to a CSV file containing the labels to be excluded from the dataset

## Prerequisites

### Pre-conditions

- The dataset must follow the `distributed-downloader` format structure
- The dataset's URL table must contain a column named `original_label`
- The CSV file specified in `path_to_excluding_labels` must exist with proper headers
- The exclusion labels CSV must have a column that matches the 'original_label' values in the dataset

### Input Format

- The exclusion labels file should be a CSV file with column headers
- At minimum, it must contain an 'original_label' column with the labels to exclude

## Outcomes

### Post-conditions

- The dataset will be filtered to exclude all images with labels matching those in the exclusion list
- Original parquet files will be replaced with filtered versions (retaining the same paths and names)
- The tool will maintain a record of all filtered images
- The filtering process is idempotent - running it multiple times will not cause additional data loss

### Output Files

- Filtered parquet files in the original dataset structure
- Filtering logs and statistics in the tools directory
- Verification files to confirm successful processing of each partition

## Notes

- In theory, this tool can be used for any dataset, as long as the original dataset contains an `original_label` column.
- The filtering process is performed in-place, so make backups if you need to preserve the original data.
- The tool is designed to run efficiently on distributed systems using MPI.
