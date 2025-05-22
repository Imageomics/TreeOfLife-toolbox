# BIOSCAN Download Guide

This guide provides instructions for downloading and integrating the BIOSCAN-5M dataset as part of the TreeOfLife-200M collection.

## Overview

[BIOSCAN-5M](https://github.com/bioscan-ml/BIOSCAN-5M) is a specialized dataset available through Google Drive (and [others](https://github.com/bioscan-ml/BIOSCAN-5M?tab=readme-ov-file#dataset-access)) rather than requiring the distributed downloader. This simpler download process is followed by integration into the TreeOfLife-200M dataset structure.

## Step-by-Step Instructions

### 1. Download Process

1. **Access and Download Dataset**
   - Download the BIOSCAN dataset from the official Google Drive location: [BIOSCAN Download](https://drive.google.com/drive/u/1/folders/1Jc57eKkeiYrnUBc9WlIp-ZS_L1bVlT-0)
   - If this does not work for some reason, please see their [dataset access](https://github.com/bioscan-ml/BIOSCAN-5M?tab=readme-ov-file#dataset-access) options and instructions.

2. **Extract the Dataset**
   - Unzip the downloaded file to your preferred location
   - Ensure you have sufficient disk space for the extracted contents

### 2. Post-Processing

1. **Transfer to Dataset Structure**
   - After extraction, transfer the images into the TreeOfLife-200M dataset structure using the specialized BIOSCAN transfer tool:

   ```bash
   tree_of_life_toolbox {config path} tol200m_bioscan_data_transfer
   ```

   - Replace `{config path}` with the path to your job configuration file
   - Use [bioscan_download_config.yaml](../config/tree_of_life_200M/bioscan_download_config.yaml) as a template
   - Note: Many values in this config are preconfigured and should not be changed unless you understand their purpose.
   - This tool handles the specific metadata and directory structure of the [BIOSCAN-5M dataset](https://github.com/bioscan-ml/BIOSCAN-5M).

## Troubleshooting

If you encounter issues:

- Verify you have appropriate access to the Google Drive link (or alternate [data access method](https://github.com/bioscan-ml/BIOSCAN-5M?tab=readme-ov-file#dataset-access)).
- Check for corrupted files if the extraction process fails.
- Ensure your configuration file correctly points to the BIOSCAN-5M extracted location.
