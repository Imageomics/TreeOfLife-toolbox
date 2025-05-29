# Processing Jobs

This folder contains scripts used for processing the downloaded **TOL-200M** images (determining which would and would not be included in the [TreeOfLife-200M dataset](https://huggingface.co/datasets/imageomics/TreeOfLife-200M)). Detailed script documentation is provided within the Python scripts located in [src/processing](../../src/processing).

## Submitting Jobs

To submit these tasks via Bash:

```bash
scripts/submit_slurm.sh scripts/<JOB_NAME>.slurm
```

Be sure to set the appropriate account name (replace `YOUR_ACCOUNT`), set `BASE_DIR`, and replace `PATH/TO/ENVIRONMENT`. Method of loading environments may vary between clusters, so please check code compatibility with your system. See [docs/create_processing_environment.md](../../docs/create_processing_environment.md) for more setup instructions.

## **Decode**
### `decode_images.slurm`

TOL-200M images are stored as binary columns in Parquet files. This job uses [`src/decode_images.py`](../../src/processing/decode_images.py) to decode these images and save them to disk. Optionally, the output images can be compressed into a `TAR` or `TAR.GZ` archive.

## **Model Filters**
### `run_megadetector.slurm`

This job uses [`src/run_megadetector.py`](../../src/processing/run_megadetector.py) to run the [MegaDetector](https://github.com/microsoft/CameraTraps) model on a directory of **decoded** images. The detection results are then saved to disk.


## **Generate Lookup Table**
### `create_lookup_tbl.slurm`

TOL-200M images are distributed across multiple Parquet files. This script generates a **lookup table** that maps UUIDs to their corresponding file paths.

The number of image data files referenced in a single lookup table can be extremely large, leading to **out-of-memory (OOM) issues** when used for downstream tasks such as extracting subsets of the TOL-200M dataset.

To mitigate this, [`src/create_lookup_tbl.py`](../../src/processing/create_lookup_tbl.py) creates lookup tables for specific pre-defined groups by filtering and joining relevant data.

The lookup table is **partitioned into batches**, each containing a limited number of data file paths to prevent excessive memory usage.

## **Extraction**
### `extract_images.slurm`

This job uses [`src/extract_images.py`](../../src/processing/extract_images.py) to extract a subset of images from the **TOL-200M** dataset using a **lookup table** and write them to disk in Parquet files.

The lookup table should be generated using the [`src/create_lookup_tbl.py`](../../src/processing/create_lookup_tbl.py) script.

### `extract_museum_specimen.slurm`

Similar to `extract_images.slurm`, but **specialized** for extracting **GBIF Museum Specimen** records that contain multiple images per occurrence.

### `extract_citizen_science.slurm`

Similar to `extract_images.slurm`, but **specialized** for extracting **GBIF Citizen Science** images, which are more numerous than other categories. 
