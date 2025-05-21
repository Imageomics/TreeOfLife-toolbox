# TreeOfLife200M Dataset Download Guide

Welcome to the TreeOfLife200M dataset download instructions! This guide provides instructions for downloading the
comprehensive TreeOfLife200M dataset using the `distributed-downloader` package and related tools.

## Introduction

The TreeOfLife200M dataset is a vast collection of biological images from multiple sources. Due to its size and
complexity, the download process is split into several components that can be run in parallel for efficiency.

## Download Components

The dataset is divided into the following components, each with its own download instructions:

- [`gbif/fast`](GBIF_fast_download_README.md) - Fast, distributed download for the majority of images from the GBIF
  source
- [`gbif/safe`](GBIF_slow_download_README.md) - Alternative slower download method for the GBIF source, used for
  rate-limited servers
- [`eol`](EoL_download_README.md) - Download instructions for Encyclopedia of Life (EoL) source images
- [`bioscan`](BIOSCAN_download_README.md) - Specialized download process for the BIOSCAN dataset component
- [`fathomNet`](FathomNet_download_README.md) - Specialized download process for the FathomNet marine life dataset

## Getting Started

1. Install the required tools as specified in each component guide
2. Configure your download environment based on available bandwidth and storage
3. Launch download processes in parallel for maximum efficiency
4. Monitor progress using the provided utilities

Each download component has its own README with detailed instructions. Click on the links above to access specific
guidance for each data source.
