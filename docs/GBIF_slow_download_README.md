# GBIF Safe Download Guide

This guide provides instructions for the safe, rate-limited download of images from the [GBIF snapshot](https://doi.org/10.15468/dl.bfv433) used in the TreeOfLife-200M dataset.

## Overview

The GBIF safe download method is designed for retrieving images from heavily rate-limited servers within the GBIF source. While slower than the fast download method, this approach is more reliable for certain servers and ensures compliance with their access policies.

## Step-by-Step Instructions

### 1. Setup and Download

1. **Install and Configure Downloader**
   - Set up the `distributed-downloader` package following the [installation instructions](https://github.com/Imageomics/distributed-downloader?tab=readme-ov-file#installation-instructions).
   - This will prepare all necessary scripts for the download process

2. **Create Configuration File**
   - Create a configuration file for your safe download
   - Use [safe_download_config.yaml](../config/tree_of_life_200M/safe_download_config.yaml) as a template
   - Note: The preconfigured values handle rate limiting appropriately and should not be modified unless necessary
   - Please ensure that the `excluded_servers_path` points to the provided [excluded_servers_safe_tol.csv](../config/tree_of_life_200M/excluded_servers_safe_tol.csv) file

3. **Execute Download**
   - Run the downloader with your configuration:

   ```bash
   distributed_downloader configs/safe_download_config.yaml
   ```

4. **Monitor and Resume**
   - Due to rate limiting, this download process will take several weeks to complete
   - If the download is interrupted or workers are depleted, simply restart the command in step 3
   - Check download progress by examining the `inner_checkpoint_file` in your download directory

### 2. Post-Processing

1. **Transfer to Dataset Structure**
   - After completing the download, transfer images to the TreeOfLife200M dataset structure using:

   ```bash
   tree_of_life_toolbox {config path} data_transfer
   ```

   - Replace `{config path}` with the path to your job configuration file
   - `safe_download_config.yaml` can be used as a template again

## Troubleshooting

If you encounter issues during download:

- This method is specifically designed for rate-limited servers, so slower download speeds are normal
- Ensure your system maintains a consistent network connection
- If you receive repeated access denied errors, you may need to temporarily pause downloads to that specific server
