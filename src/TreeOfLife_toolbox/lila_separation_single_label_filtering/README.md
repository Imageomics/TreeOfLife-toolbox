# LILA Separation Single Label Filtering Tool

## Overview

This tool is designed to filter a dataset to retain only single-label images by removing multi-label images (images that
have several objects labeled on them). It processes datasets in the distributed-downloader format and performs filtering
based on a provided CSV file containing UUID identifiers.

## Components

### Filter Component

`LilaSeparationSingleLabelFilteringFilter` copies the input CSV file containing single-label image information to the
appropriate filter table directory, setting up the filtering process.

### Scheduler Component

`LilaSeparationSingleLabelFilteringScheduleCreation` creates a distributed work schedule based on server names and
partition IDs to efficiently process the dataset across multiple workers.

### Runner Component

`LilaSeparationSingleLabelFilteringRunner` performs the actual filtering by reading the provided UUIDs and applying the
filter to each partition of the dataset, removing any images that don't match the criteria.

## Configuration Requirements

The tool requires the following configuration fields:

- `data_path`: Path to the CSV table containing single-label images (must include a `uuid` column)

## Prerequisites

- The CSV table specified in `data_path` must contain entries identified by a `uuid` column
- The dataset must be in `distributed-downloader` format with appropriate server_name and partition_id organization
- Standard TreeOfLife toolbox environment and dependencies must be set up

## Post Conditions

- The resulting dataset will maintain the `distributed-downloader` format
- Filtering is performed in-place, modifying the original dataset
- The tool's checkpoint system tracks progress, allowing for resumption after interruptions
- Verification ensures all partitions are processed before marking the tool as completed
