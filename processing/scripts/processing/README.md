# Processing Jobs

This folder contains scripts used for processing the downloaded **TOL-200M** images (determining which would and would not be included in the [TreeOfLife-200M dataset](https://huggingface.co/datasets/imageomics/TreeOfLife-200M)). Detailed script documentation is provided within the Python scripts located in [src/processing](../../src/processing).

## Submitting Jobs

To submit these tasks via Bash:

```bash
scripts/submit_slurm.sh scripts/<JOB_NAME>.slurm
```

Be sure to set the appropriate account name (replace `YOUR_ACCOUNT`), set `BASE_DIR`, and replace `PATH/TO/ENVIRONMENT`. Method of loading environments may vary between clusters, so please check code compatibility with your system. See [docs/create_processing_environment.md](../../docs/create_processing_environment.md) for more setup instructions.

## **Decode:** `decode_images.slurm`

TOL-200M images are stored as binary columns in Parquet files. This job uses [`src/decode_images.py`](../../src/processing/decode_images.py) to decode these images and save them to disk. Optionally, the output images can be compressed into a `TAR` or `TAR.GZ` archive.

## **Generate Lookup Table:** `create_lookup_tbl.slurm`

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

## **PDQ Hash: **`create_pdq_hash.slurm`

This job computes [PDQ](https://github.com/facebook/ThreatExchange/tree/main/pdq) perceptual hashes for images stored in Parquet files using PySpark, enabling scalable and distributed processing of large image datasets. It reads images and metadata from Parquet files (with images stored as binary columns), computes the PDQ hash for each image, and writes the results—containing UUIDs and their corresponding PDQ hashes—to new Parquet files organized by batch.

## **Model Filters**

The scripts in this section use pre-trained models to perform inference on TOL-200M images stored as binary columns in Parquet files.

They are designed for scalable, distributed processing across multiple compute nodes and GPUs. Each process (or "rank") handles a subset of the data and writes its results to disk. Please adjust the allocated resources in the SLURM scripts to match your hardware and workload.

To prevent time-outs and interruptions, each process saves checkpoints (tracking progress and intermediate results) and writes a log file listing processed files.

For example, a job using 4 nodes with 2 GPUs each (8 ranks) produces the following output structure:

``` bash
.
├── checkpoints_rank0               # Checkpoints for rank 0
├── checkpoints_rank1
├── ...
├── checkpoints_rank6
├── checkpoints_rank7
├── logs                            
│   ├── processed_files_rank0.log   # Log of files processed by rank 0
│   ├── processed_files_rank1.log
│   ├── ...
│   ├── processed_files_rank6.log
│   └── processed_files_rank7.log
├── predictions_rank0.parquet       # Model inference results for rank 0
├── predictions_rank1.parquet
├── ...
├── predictions_rank6.parquet
└── predictions_rank7.parquet
```

We provide a summary of the available jobs below. For job script arguments and implementation details, please refer to the docstrings in the corresponding scripts. Follow the documentation at [`docs/create_processing_environment.md`](../../docs/create_processing_environment.md) to set up the processing environment.

### `run_megadetector_distributed.slurm`

This job uses [`src/run_megadetector_distributed.py`](../../src/processing/run_megadetector_distributed.py) to run the [MegaDetector](https://github.com/microsoft/CameraTraps) model, which identifies empty frames in camera-trap image data (i.e., images with no animal presence). Users need to specify the directory containing the Parquet data and the output location for inference results.

### `run_face_detection_distributed.slurm`

This job uses [`src/run_face_detection_distributed.py`](../../src/processing/run_face_detection_distributed.py) to run the [MTCNN](https://github.com/timesler/facenet-pytorch) model, which detects images containing human faces. Users need to specify the directory containing the Parquet data and the output location for inference results.

### `run_clip_distributed.slurm`

This job uses [`src/run_clip_distributed.py`](../../src/processing/run_clip_distributed.py) to run OpenAI's [CLIP](https://github.com/openai/CLIP) model for classification tasks. In addition to specifying the Parquet data directory and output location, users must provide a path to a dictionary containing class labels and their corresponding CLIP embeddings. Note: This script uses the CLIP `ViT-L/14@336px` model.

This job was primarily used to identify invalid data (e.g., files, tags, and boxes) among museum specimen images. Support set embedding dictionaries for different museum specimen categories are provided in [`processing/data/support_set_embeddings/museum_specimen`](../../processing/data/support_set_embeddings/museum_specimen).

