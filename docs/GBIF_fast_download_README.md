# GBIF Fast Download Guide

This guide provides instructions for the fast, distributed download of images from the Global Biodiversity Information Facility (GBIF) source as part of the TreeOfLife200M dataset.

## Overview

The GBIF fast download method allows you to efficiently retrieve the majority of images from the GBIF source using distributed processing. This approach is optimized for speed and will retrieve most images in the dataset.

## Step-by-Step Instructions

### 1. Setup and Download

1. **Install and Configure Downloader**
   - Set up the `distributed-downloader` package following the [official instructions](https://github.com/Imageomics/distributed-downloader/blob/9ef8b0d297f7a868fac31b2b9c3d5f3aa5533472/docs/scripts_README.md)
   - This will prepare all necessary scripts for the download process

2. **Create Configuration File**
   - Create a configuration file for your download
   - Use [general_download_config.yaml](../config/tree_of_life_200M/general_download_config.yaml) as a template
   - Note: Many values in this config are preconfigured and should not be changed unless you understand their purpose

3. **Execute Download**
   - Run the downloader with your configuration:

   ```bash
   distributed_downloader configs/general_download_config.yaml
   ```

4. **Monitor and Resume**
   - The complete download process may take several weeks
   - If the download is interrupted or workers are depleted, simply restart the command in step 3
   - Check download progress by examining the `inner_checkpoint_file` in your download directory

### 2. Post-Processing

1. **Transfer to Dataset Structure**
   - After completing the download, transfer images to the TreeOfLife200M dataset structure using:

   ```bash
   tree_of_life_toolbox {config path} data_transfer
   ```

   - Replace `{config path}` with the path to your job configuration file

## Troubleshooting

If you encounter issues during download:

- Verify your network connection is stable
- Ensure you have sufficient disk space for the entire download
- Check logs for any error messages that might indicate configuration issues
