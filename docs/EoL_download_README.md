# Encyclopedia of Life (EoL) Download Guide

This guide provides instructions for downloading images from the [Encyclopedia of Life (EOL)](https://eol.org) data provider as part of the TreeOfLife-200M dataset.

## Overview

The Encyclopedia of Life (EoL) component contains unique biodiversity images that complement the GBIF sources.

## Step-by-Step Instructions

### 1. Setup and Download

1. **Install and Configure Downloader**
   - Set up the `distributed-downloader` package following the [official instructions](https://github.com/Imageomics/distributed-downloader/blob/9ef8b0d297f7a868fac31b2b9c3d5f3aa5533472/docs/scripts_README.md)
   - This will prepare all necessary scripts for the download process

2. **Create Configuration File**
   - Create a configuration file for your EoL download
   - Use [eol_download_config.yaml](../config/tree_of_life_200M/eol_download_config.yaml) as a template
   - Note: The preconfigured values are optimized for EoL's servers and should not be modified unless necessary

3. **Execute Download**
   - Run the downloader with your configuration:

   ```bash
   distributed_downloader configs/eol_download_config.yaml
   ```

4. **Monitor and Resume**
   - The complete download process may take several weeks
   - If the download is interrupted or workers are depleted, simply restart the command in step 3
   - Check download progress by examining the `inner_checkpoint_file` in your download directory

### 2. Post-Processing

1. **Transfer to Dataset Structure**
   - After completing the download, transfer images to the TreeOfLife-200M dataset structure using:

   ```bash
   tree_of_life_toolbox {config path} data_transfer
   ```

   - Replace `{config path}` with the path to your job configuration file
   - `eol_download_config.yaml` can be used as a template again

## Troubleshooting

If you encounter issues during download:

- EoL may occasionally update their API or website structure; check for any announcements
- The download is configured with appropriate rate limits, but you may need to adjust these if you receive HTTP 429 errors
- Some EoL images may be hosted on third-party servers which can occasionally be unavailable
