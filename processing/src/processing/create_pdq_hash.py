"""
create_pdq_hash.py

This script computes PDQ (Perceptual hash for Digital images and video) hashes for images stored in Parquet files using PySpark. It is designed for distributed and scalable processing of large image datasets.

Features:
- Reads images and metadata from Parquet files, where images are stored as binary columns.
- Computes PDQ hashes for each image using the pdqhash library.
- Processes files in configurable batches to optimize memory and resource usage.
- Supports checkpointing: maintains a log of processed files to avoid redundant computation and enable resumable runs.
- Outputs Parquet files containing UUIDs and their corresponding PDQ hashes, organized by batch.

Usage:
    python create_pdq_hash.py \
        --target_dir /path/to/parquet/files \
        --output_dir /path/to/output \
        [--file_paths /path/to/file_list.txt] \
        [--processed_files_log_path /path/to/processed_files.log] \
        [--batch_size 30]

Arguments:
    --target_dir: Directory containing input Parquet files.
    --output_dir: Directory to save output Parquet files with PDQ hashes.
    --file_paths: (Optional) Path to a text file listing specific Parquet files to process.
    --processed_files_log_path: (Optional) Path to a log file tracking processed files.
    --batch_size: (Optional) Number of files to process per batch (default: 30).

Requirements:
    - PySpark
    - pdqhash
    - numpy
    - Pillow (PIL)

Output:
    For each batch, a subdirectory (e.g., batch_0000) is created in the output directory containing a Parquet file with:
        - uuid: Unique identifier for each image.
        - hash_pdq: The computed PDQ hash (as binary).
    A log file is updated with the paths of processed files.

"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import struct, col, udf
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, BooleanType,
    ArrayType, LongType, BinaryType
)

import argparse
import os
from pathlib import Path
import logging

import pdqhash
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def get_img_data_schema():
    schema = StructType([
        StructField("uuid", StringType(), nullable=True),
        StructField("source_id", StringType(), nullable=True),
        StructField("identifier", DoubleType(), nullable=True),
        StructField("is_license_full", BooleanType(), nullable=True),
        StructField("license", StringType(), nullable=True),
        StructField("source", StringType(), nullable=True),
        StructField("title", DoubleType(), nullable=True),
        StructField("hashsum_original", StringType(), nullable=True),
        StructField("hashsum_resized", StringType(), nullable=True),
        StructField("original_size", ArrayType(LongType(), containsNull=True), nullable=True),
        StructField("resized_size", ArrayType(LongType(), containsNull=True), nullable=True),
        StructField("image", BinaryType(), nullable=True),
        StructField("server", StringType(), nullable=True),
    ])
    return schema

def init_spark() -> SparkSession:

    spark = (
        SparkSession.builder
        .appName("GBIF EDA")
        .config("spark.executor.instances", "80")
        .config("spark.executor.memory", "75G")
        .config("spark.executor.cores", "12")
        .config("spark.driver.memory", "64G")
        # Additional Tunning
        .config("spark.sql.shuffle.partitions", "1000")
        #.config("spark.sql.files.maxPartitionBytes", "256MB")
        .config("spark.sql.parquet.enableVectorizedReader", "false") 
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )
    
    return spark

def decode_image(row_data) -> np.ndarray:
    N_CHANNELS = 3
    image_bytes = row_data["image"]
    np_image = np.frombuffer(image_bytes, dtype=np.uint8)
    
    
    for key in ["original_size", "resized_size"]:
        if key in row_data:
            height, width = row_data[key]
            expected_size = height * width * N_CHANNELS
            if np_image.size == expected_size:
                img_array = np_image.reshape((height, width, N_CHANNELS))
                return img_array[..., ::-1]

    logging.warning("Image size does not match expected dimensions.")
    return None


def hash_image(row_data):
    np_img = decode_image(row_data)
    if np_img is None:
        return None
    try:
        hash_bits, _ = pdqhash.compute(np_img)
        bits_array = np.packbits(hash_bits)
        return bytes(bits_array)
    except Exception:
        logging.exception("Failed to compute hash")
        return None
    
hash_image_udf = udf(hash_image, BinaryType())

def save_processed_file(processed_files, processed_files_log):

    log_dir = os.path.dirname(processed_files_log)
    if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    
    with open(processed_files_log, "a") as f:
        for file_path in processed_files:
            f.write(f"{file_path}\n")

def load_processed_files(processed_files_log):
    """Load processed files from the log."""
    if os.path.exists(processed_files_log):
        with open(processed_files_log, "r") as f:
            return set(f.read().splitlines())
    return set()


def main(target_dir: str,  output_dir: str, file_paths: str = None, processed_files_log_path: str = None,  batch_size: int=30):

    schema = get_img_data_schema()
    spark = init_spark()

    
    
    if file_paths is not None:
        logging.info(f"Loading processed files from: {file_paths}")
        parquet_files = load_processed_files(file_paths)
    else: 
        parquet_files = [str(p) for p in Path(target_dir).rglob('*.parquet')]

    # Filter out already processed files
    if processed_files_log_path is not None:
        processed_files = load_processed_files(processed_files_log_path)
        parquet_files = [f for f in parquet_files if f not in processed_files]
    
    logging.info(f"Found {len(parquet_files)} parquet files to process.")
    if not parquet_files:
        logging.info("No new files to process.")
        return

    for i in range(0, len(parquet_files), batch_size):
        batch_files = parquet_files[i:i+batch_size]
        logging.info(f"Processing batch {i // batch_size + 1} of {len(parquet_files) // batch_size + 1}")

        df = spark.read.schema(schema).parquet(*batch_files)

        df_hashed = (
            df
            .withColumn("hash_pdq", hash_image_udf(struct("image", "original_size", "resized_size")))
            .select("uuid", "hash_pdq")
        )

        # Use append mode and a unique output subdir per batch
        batch_output_path = f"{output_dir}/batch_{i // batch_size:04d}"
        df_hashed.write.mode("overwrite").parquet(batch_output_path)
        save_processed_file(batch_files, processed_files_log_path)

        logging.info(f"Wrote batch to: {batch_output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute PDQ hashes for images in parquet files.")
    parser.add_argument("--target_dir", type=str, help="Directory containing parquet files.")
    parser.add_argument("--output_dir", type=str, help="Output directory for hashed data.")
    parser.add_argument("--file_paths", type=str, required=False, help="File paths to process.")
    parser.add_argument("--processed_files_log_path", type=str, required=False,  help="Path to log file for processed files.")
    parser.add_argument("--batch_size", type=int, default=30, help="Number of files to process in each batch.")

    args = parser.parse_args()
    
    main(args.target_dir, args.output_dir, args.file_paths, args.processed_files_log_path, args.batch_size)