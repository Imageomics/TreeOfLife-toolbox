# FathomNet Image Cropping Tool

This tool is used to crop images from the FathomNet dataset using the bounding box coordinates provided in an external
CSV file. The cropping operation extracts regions of interest from full-sized images and preserves them as separate
image entries with updated metadata.

## How It Works

The tool consists of three main components:

1. **FathomnetCropFilter**: Identifies all valid image partitions in the dataset that can be processed. Creates a simple
   list of server/partition pairs for the scheduler.

2. **FathomnetCropScheduleCreation**: Creates a processing schedule by distributing server/partition pairs across
   available worker nodes to balance the workload.

3. **FathomnetCropRunner**: Performs the actual image cropping operation by:
    - Loading images that have corresponding bounding box entries
    - Cropping each image according to the specified coordinates (x, y, width, height)
    - Computing new hash values and metadata for the cropped images
    - Saving the results in a structure that mirrors the original dataset

## Required Configuration

Add these fields to your configuration file:

- `bb_csv_path`: Path to the CSV file containing the bounding box information with columns:
    - `image_uuid`: UUID of the original image
    - `bb_uuid`: UUID for the bounding box entry
    - `tol_uuid`: UUID to be assigned to the cropped image
    - `x`, `y`: Top-left coordinates of the bounding box
    - `width`, `height`: Dimensions of the bounding box

- `image_crop_path`: Path where the cropped images will be stored

## Pre-conditions

- The source image directory follows the distributed-downloader structure:
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
- The bounding box CSV file exists and contains the required columns
- The original images are stored in parquet files with binary image data and metadata
- Images with multi-channel RGB data (3 channels)

## Post-conditions

- Cropped images are saved in the specified output path using the same directory structure as the source
- Each cropped image has:
    - Updated UUID and source_id based on values from the bounding box CSV
    - Updated resized_size reflecting the dimensions of the cropped region
    - New hashsum_resized value calculated from the cropped image data
- If a bounding box extends beyond the image boundaries, it will be automatically clipped to fit within the image
- All original metadata is preserved, except for fields that needed to be updated due to cropping
- A verification structure is created to track which server/partition pairs have been successfully processed
