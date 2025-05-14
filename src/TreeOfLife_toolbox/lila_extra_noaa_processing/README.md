# LILA Extra NOAA Processing

## Overview

This tool processes the LILA NOAA (National Oceanic and Atmospheric Administration) dataset and converts it into the
standardized `TreeOfLife-toolbox` format compatible with the distributed-downloader ecosystem. The tool performs the
following key operations:

1. **Filtering**: Loads the NOAA dataset, standardizes column names, generates UUIDs, and partitions the data
2. **Scheduling**: Creates a processing schedule to distribute work across compute resources
3. **Processing**: Loads and crops images according to bounding box coordinates, computes hash values, and saves
   processed data in parquet format

The tool was **specifically** developed to convert LILA NOAA dataset into `distributed-downloader` format. It is not
going to work on anything else.

## Configuration Requirements

### Required Fields in Config

- `og_images_root`: Path to the root directory of the NOAA images (absolute path)

## Assumptions/Pre-conditions

- The NOAA images are available in the `og_images_root` directory.
- The input CSV file contains the following columns:
    - `detection_id`: Unique identifier for each detection
    - `detection_type`: Life stage of the detected organism
    - `rgb_image_path`: Relative path to the image from the root directory
    - `rgb_left`, `rgb_right`, `rgb_top`, `rgb_bottom`: Bounding box coordinates for cropping
- The paths in `rgb_image_path` are relative to the `og_images_root` directory.

## Post-conditions

After successful execution, the following is guaranteed:

1. The processed data is available in the configured output directory with the structure:
   ```
   {images_folder}/server_name=noaa/partition_id={id}/successes.parquet
   ```

2. Each parquet file contains:
    - `uuid`: Unique identifier for each entry
    - `source_id`: Original detection ID
    - `identifier`: Full path to the original image
    - `is_license_full`: Set to False (NOAA data does not include license information)
    - `original_size`: Dimensions of the original image
    - `resized_size`: Dimensions of the cropped image
    - `hashsum_original`: MD5 hash of the original image
    - `hashsum_resized`: MD5 hash of the cropped image
    - `image`: Binary data of the cropped image

3. The verification tables confirm the completion of processing for all partitions.
