"""
decode_images.py

This script processes images that stored in Parquet files as a binary column, decodes them, and saves them to disk. Optionally, it can compress the saved images into a TAR or TAR.GZ file.

Usage:
    python decode_images.py <input_path> <output_path> [--compress] [--format <format>] [--partition_size <partition_size>]

Arguments:
    input_path (str): Path to the input DataFrame in Parquet format, binary image data should be stored in a column named "image". Could be a parquet file or a Spark write-out directory containing parquet files.
    output_path (str): Path to the output directory for saving decoded images.
    --compress: Optional flag to compress the images into a TAR.GZ file.
    --format (str): Image format for saving, choose between 'jpeg' or 'png' (default 'jpeg').
    --partition_size (int): Number of images to process in each partition (default 100), change based on available spark executors memory.

Dependencies:
    - os
    - logging
    - argparse
    - pyspark
    - PIL (Pillow)
    - numpy
    - tarfile
    - shutil
    - tqdm

Functions:
    init_spark():
        Initializes and returns a SparkSession.

    decode_image(row_data):
        Decodes binary image data into a RGB NumPy array. `row_data` should contain "image", "original_size", and "resized_size" fields.

    decode_image_to_pil(row_data):
        Decodes binary image data into a PIL Image. Wrapper function for `decode_image`.

    compress_dir(dir_path, tar_file_path, use_gzip=True):
        Compresses a directory into a TAR or TAR.GZ file and deletes the original directory.

    save_images_partition(rows, output_dir, compress=False, format="jpeg"):
        Decodes and saves images from a list of rows (a Spark DataFrame partition) to the specified output directory.

    flatten_dir(target_dir):
        Moves all files from subdirectories into the target directory and removes the subdirectories. Helps to merge all partition write results into a single directory.

    main(input_path, output_path, compress=False, format="jpeg", partition_size=100):
        Main function to process images from the input Parquet data, decode and save them to disk.

Example:
    python decode_images.py /path/to/input_data /path/to/output --compress --format png --partition_size 50
"""

from typing import Union, List, Literal
import os
import logging
import argparse
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import broadcast

from PIL import Image
import numpy as np
import tarfile
import shutil
from PIL import Image
from tqdm import tqdm

import os
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Constants
N_EXECUTORS = 64

def init_spark() -> SparkSession:

    spark = (
        SparkSession.builder
        .appName("GBIF")
        .config("spark.executor.instances", f"{N_EXECUTORS}")
        .config("spark.executor.memory", "75G")
        .config("spark.executor.cores", "12")
        .config("spark.driver.memory", "64G")
        # Additional Tunning
        #.config("spark.sql.shuffle.partitions", "1000")
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

def decode_image_to_pil(row_data: Row):
    """
    Decode binary image data from a DataFrame row and convert it to a PIL Image.

    Parameters:
    - row_data: pyspark.sql.Row
        A row containing `image`, `original_size`, and `resized_size` fields.

    Returns:
    - pil_image: PIL.Image.Image
        The decoded and converted PIL Image.
    """
    image_array = decode_image(row_data)

    # Convert NumPy array to PIL Image
    if image_array is not None:
        return Image.fromarray(image_array, mode="RGB")
    return None

def compress_dir(dir_path, tar_file_path, use_gzip=True):
    """
    Replace a folder with its TAR or TAR.GZ file.

    Parameters:
    - dir_path (str): Path to the folder to archive and delete.
    - tar_file_path (str): Path for the resulting TAR file (without .tar/.tar.gz extension).
    - use_gzip (bool): Whether to compress the archive with Gzip (default True).
    """
    mode = "w:gz" if use_gzip else "w"  # Compression mode
    extension = ".tar.gz" if use_gzip else ".tar"
    tar_file_path += extension  # Add appropriate extension

    try:
        # Step 1: Create the TAR archive
        with tarfile.open(tar_file_path, mode) as tar:
            tar.add(dir_path, arcname=os.path.basename(dir_path))
        logging.info(f"Successfully created TAR file: {tar_file_path}")

        # Step 2: Verify the TAR file creation
        if os.path.exists(tar_file_path):
            # Step 3: Delete the original folder
            shutil.rmtree(dir_path)
            logging.info(f"Original folder '{dir_path}' deleted after archiving.")
        else:
            logging.error(f"Error: TAR file not created. Folder '{dir_path}' was not deleted.")
    except Exception as e:
        logging.error(f"Error during archiving or deletion: {e}")

def save_images_partition(rows, output_dir: str, compress: bool = False, format: Literal["jpeg", "png"] = "jpeg"):
    """
    Save images from a list of rows to the specified output directory.

    Parameters:
    - rows: iterable
        List of pyspark.sql.Row objects containing `image`, `original_size`, `resized_size`, and `uuid` fields.
    - output_dir: str
        The directory where images will be saved.
    - compress: boolean
        Whether to compress the images into a TAR file (default False).
    - format: str
        The image format to use for saving choose between 'jpeg' or 'png' (default 'jpeg').
    """
    rows = list(rows)
    # print(f"Processing {len(rows)} rows in partition by PID: {os.getpid()}")

    # Create the partition-specific directory
    partition_output_dir = os.path.join(output_dir, f"partition_{os.getpid()}")
    os.makedirs(partition_output_dir, exist_ok=True)
    
    # Wrap the rows in tqdm for progress tracking
    for row in tqdm(rows, desc=f"PID {os.getpid()} Progress", unit="image"):
        pil_image = decode_image_to_pil(row)
        if pil_image is not None:
            uuid = row["uuid"]
            output_path = os.path.join(partition_output_dir, f"{uuid}.{format}")
            pil_image.save(output_path)
        else:
            logging.error(f"Error saving image for UUID: {row['uuid']}")
    logging.info(f"Written {len(rows)} images to {partition_output_dir}")

    # Compresss folder into tar files
    if compress:
        logging.info(f"Compressing...")        
        compress_dir(
            dir_path = partition_output_dir,
            tar_file_path = output_dir,
            use_gzip = True
        )
        logging.info(f"Images compressed into {output_dir}")

def flatten_dir(target_dir):
    """
    Moves all files from subdirectories into the target directory
    and removes the subdirectories afterward. It also validates that
    no files are lost in the process.

    Parameters:
        target_dir (str): The main directory where files should be moved.
    """
    # Count total files before moving
    original_file_count = sum([len(files) for _, _, files in os.walk(target_dir)])
    moved_file_count = 0
    
    for root, _, files in os.walk(target_dir, topdown=False):
        if root == target_dir:
            continue  # Skip the main directory
        
        for file in files:
            src_path = os.path.join(root, file)
            dst_path = os.path.join(target_dir, file)
            
            # Ensure unique filenames by adding suffix if needed
            counter = 1
            while os.path.exists(dst_path):
                file_name, file_ext = os.path.splitext(file)
                dst_path = os.path.join(target_dir, f"{file_name}_{counter}{file_ext}")
                counter += 1
            
            shutil.move(src_path, dst_path)
            moved_file_count += 1
        
        # Remove empty subdirectory
        os.rmdir(root)
    
    # Count total files after moving
    final_file_count = sum([len(files) for _, _, files in os.walk(target_dir)])
    
    if final_file_count == original_file_count:
        print(f"Successfully moved {moved_file_count} files. No files lost.")
    else:
        print(f"Validation Failed! Expected {original_file_count}, but found {final_file_count} files.")

def main(input_path: str, output_path: str, compress: bool = False, format: Literal["jpeg", "png"] = "jpeg", partition_size: int = 100) -> None:

    spark = init_spark()

    df = spark.read.parquet(input_path)
    n_images = df.count()
    logging.info(f"Loaded {n_images} images from {input_path}")

    # Repartition the DataFrame
    n_partitions = max(1, n_images // partition_size)
    df = df.repartition(n_partitions)
    logging.info(f"Processing {n_images} images in {n_partitions} partitions")

    # Process images in parallel
    # Not compressing the images in partitions
    df.foreachPartition(lambda partition: save_images_partition(partition, output_path, compress=False, format=format))
    
    # Extract images from partitions write-out into one directory
    flatten_dir(output_path)

    # Compress the images into a TAR file
    if compress:
        logging.info(f"Compressing...")        
        compress_dir(
            dir_path = output_path,
            tar_file_path = output_path,
            use_gzip = True
        )
    else:
        logging.info(f"Successfully saved images to {output_path}")
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save images from a DataFrame to disk.")
    parser.add_argument("input_path", type=str, help="Path to the input DataFrame in Parquet format.")
    parser.add_argument("output_path", type=str, help="Path to the output directory for saving images.")
    parser.add_argument("--compress", action="store_true", help="Compress the images into a TAR file.")
    parser.add_argument("--format", type=str, default="jpeg", choices=["jpeg", "png"], help="Image format for saving.")
    parser.add_argument("--partition_size", type=int, default=100, help="Number of images to process in each partition.")
    args = parser.parse_args()

    main(args.input_path, args.output_path, args.compress, args.format, args.partition_size)


    

