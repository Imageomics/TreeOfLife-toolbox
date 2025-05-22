# BIOSCAN Download Guide

This guide provides instructions for downloading and integrating the BIOSCAN dataset as part of the TreeOfLife200M collection.

## Overview

BIOSCAN is a specialized dataset available through Google Drive rather than requiring the distributed downloader. This simpler download process is followed by integration into the TreeOfLife200M dataset structure.

## Step-by-Step Instructions

### 1. Download Process

1. **Access and Download Dataset**
   - Download the BIOSCAN dataset from the official Google Drive location: [BIOSCAN Download](https://drive.google.com/drive/u/1/folders/1Jc57eKkeiYrnUBc9WlIp-ZS_L1bVlT-0)
   - You may need appropriate permissions to access this drive folder

2. **Extract the Dataset**
   - Unzip the downloaded file to your preferred location
   - Ensure you have sufficient disk space for the extracted contents

### 2. Post-Processing

1. **Transfer to Dataset Structure**
   - After extraction, transfer the images into the TreeOfLife200M dataset structure
   - Use the specialized BIOSCAN transfer tool:

   ```bash
   tree_of_life_toolbox {config path} tol200m_bioscan_data_transfer
   ```

   - Replace `{config path}` with the path to your job configuration file
   - Use [bioscan_download_config.yaml](../config/tree_of_life_200M/bioscan_download_config.yaml) as a template
   - Note: Many values in this config are preconfigured and should not be changed unless you understand their purpose
   - This tool handles the specific metadata and directory structure of the BIOSCAN dataset

## Troubleshooting

If you encounter issues:

- Verify you have appropriate access to the Google Drive link
- Check for corrupted files if the extraction process fails
- Ensure your configuration file correctly points to the BIOSCAN extracted location
