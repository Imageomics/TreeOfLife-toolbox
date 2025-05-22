# FathomNet Download Guide

This guide provides instructions for downloading and processing marine life images from the [FathomNet database](https://database.fathomnet.org/fathomnet/#/) as part of the TreeOfLife-200M collection.

## Overview

FathomNet is a specialized database focusing on marine organisms. It contains annotated underwater imagery that requires specific processing, including bounding box cropping, to properly incorporate into the TreeOfLife-200M dataset.

## Step-by-Step Instructions

### 1. Setup and Download

1. **Install and Configure Downloader**
   - Set up the `distributed-downloader` package following the [official instructions](https://github.com/Imageomics/distributed-downloader/blob/9ef8b0d297f7a868fac31b2b9c3d5f3aa5533472/docs/scripts_README.md)
   - This will prepare all necessary scripts for the download process

2. **Create Configuration File**
   - Create a configuration file for your FathomNet download
   - Use [fathomNet_download_config.yaml](../config/tree_of_life_200M/fathomNet_download_config.yaml) as a template
   - Note: The preconfigured values are optimized for FathomNet's data structure and should not be modified unless necessary

3. **Execute Download**
   - Run the downloader with your configuration:

   ```bash
   distributed_downloader configs/fathomNet_download_config.yaml
   ```

4. **Monitor and Resume**
   - The complete download process may take some time.
   - If the download is interrupted or workers are depleted, simply restart the command in step 3
   - Check download progress by examining the `inner_checkpoint_file` in your download directory

### 2. Post-Processing

1. **Crop Images to Bounding Boxes**
   - FathomNet images require cropping to isolate the organisms based on provided bounding box coordinates
   - Use the specialized cropping tool:

   ```bash
   tree_of_life_toolbox {config path} tol200m_fathom_net_crop
   ```

   - Replace `{config path}` with the path to your job configuration file
   - `fathomNet_download_config.yaml` can be used as a template again
   - This step is essential for extracting the relevant organisms from the underwater imagery

2. **Transfer to Dataset Structure**
   - After cropping, transfer processed images to the TreeOfLife-200M dataset structure by running:

   ```bash
   tree_of_life_toolbox {config path} data_transfer
   ```

   - Replace `{config path}` with the path to your job configuration file
   - `fathomNet_download_config.yaml` can be used as a template again

## Troubleshooting

If you encounter issues during download:

- Check your API access to FathomNet if applicable
- The cropping process may be resource-intensive; ensure your system has sufficient memory
- Verify that bounding box metadata is present in the downloaded data before proceeding to the cropping step
