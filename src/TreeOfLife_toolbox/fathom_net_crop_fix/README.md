# FathomNet Crop Fix Tool

## Overview

This tool corrects improperly cropped FathomNet images by reprocessing them from the original source images using
improved cropping algorithms. It addresses boundary issues in the previous cropping implementation by enforcing proper
bounds checking and preventing out-of-bounds errors.

The tool follows a three-stage processing pipeline:

1. **Filter Stage**: Identifies affected images by joining UUID tables with lookup information
2. **Scheduler Stage**: Organizes processing by server to enable efficient parallel execution
3. **Runner Stage**: Performs the actual image recropping using correct boundary parameters

> ⚠️ **Note**: This is a specialized tool built for a specific dataset issue. It should not be used for other cases
> without code modifications.

## Configuration Requirements

The following fields must be defined in your configuration file:

| Field                      | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `uuid_table_path`          | Path to CSV/parquet with UUIDs of images needing recropping                 |
| `look_up_table_path`       | Path to lookup table with `uuid` to `file_name` mapping information         |
| `filtered_by_size`         | Path to original CSV containing bounding box coordinates and UUID matches   |
| `data_transfer_table`      | Path to CSV mapping ToL dataset files to original image locations           |
| `base_path`                | Base directory where images were transferred using the `data_transfer` tool |
| `original_image_base_path` | Base directory where original uncropped images are stored                   |
| `image_crop_path`          | Output directory where corrected cropped images will be saved               |

## Pre-Conditions

For the tool to work correctly, the following conditions must be met:

- Original uncropped images still exist and are accessible
- Original images have not been modified since initial cropping
- Initial cropping was performed using the `fathom_net_crop` tool
- Images were transferred and restructured using the `data_transfer` tool
- Transfer logs are available to provide mapping between new filenames and original files
- The provided `filtered_by_size` CSV contains valid bounding box information (x, y, width, height)

## Processing Details

The tool applies the following corrections to each image:

- Ensures crop boundaries stay within the original image dimensions
- Applies proper bounds checking to prevent negative coordinates
- Ensures maximum bounds do not exceed image dimensions
- Recalculates image hashes for the properly cropped images
- Preserves all original metadata while updating size information

## Post-Conditions

After successful execution:

- Corrected cropped images are saved to the `image_crop_path` directory
- Images are organized in a server-based directory structure
- Each output file contains properly cropped images with corrected dimensions
- Each cropped image maintains its original UUID and source identification
- New hashsum values are calculated for the corrected images
- Verification data is created to track processing completion

## Usage Notes

The tool is designed to run in a distributed environment using MPI. It handles processing in batches by server to
maximize efficiency and manages timeouts to ensure job completion within allocation constraints.

**Technical Implementation**: The core fix applies proper boundary checking to ensure crop coordinates are within valid
image dimensions:

```python
# Calculate corrected crop coordinates with proper bounds checking
min_y = min(image_size[0], max(row["y"], 0))
min_x = min(image_size[1], max(row["x"], 0))
max_y = min(image_size[0], max(row["y"] + row["height"], 0))
max_x = min(image_size[1], max(row["x"] + row["width"], 0))
```

This prevents both negative coordinates and exceeding image dimensions, which were the main causes of errors in the
original cropping implementation.