from typing import Union, List, Literal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import tarfile
import shutil
from PIL import Image
from tqdm import tqdm

from pyspark.sql import DataFrame, SparkSession, Row
from pyspark.sql.functions import col, broadcast, row_number
from ipywidgets import interact, fixed

def find_image(
    uuid: Union[str, List[str], DataFrame],
    lookup_tbl: DataFrame,
    spark: SparkSession
) -> DataFrame:
    
    # ==================== #
    # ---- Type Check ----
    # ==================== #
    # Goal here is to create a spark dataframe object
    # containing the uuid, then used for broadcast joins
    if isinstance(uuid, str):
        # If `uuid` is a string, convert to a Spark DataFrame
        uuid_df = spark.createDataFrame([(uuid,)], ["uuid"])
    
    elif isinstance(uuid, list):
        # If `uuid` is a list of strings, convert to a Spark DataFrame
        uuid_df = spark.createDataFrame([(u,) for u in uuid], ["uuid"])
    
    elif isinstance(uuid, DataFrame):
        # If `uuid` is a DataFrame, verify it contains the column `uuid`
        if "uuid" not in uuid.columns:
            raise ValueError("The provided DataFrame must contain a 'uuid' column.")
        uuid_df = uuid.select("uuid")
    
    else:
        raise TypeError("The `uuid` parameter must be a string, list of strings, or a Spark DataFrame that contains `uuid` column.")
    
    uuid_df = uuid_df.select("uuid").distinct() # Prevent duplication
    # ======================== #
    # ---- Find All Files ----
    # ======================== #

    file_path_df = lookup_tbl.join(broadcast(uuid_df), on = "uuid", how="inner")
    unique_paths = [row['path'] for row in file_path_df.select("path").distinct().collect()]
    if not unique_paths:
        raise ValueError("No matching file paths found for the provided UUIDs.")

    # ============================ #
    # ---- Load, Filter, Save ----
    # ============================ #
    
    # load all matched files at once
    combined_df = spark.read.parquet(*unique_paths)
    image_df = (
        combined_df
        .join(broadcast(uuid_df), on="uuid", how="inner")
        .select(
            ["uuid", "source_id", "hashsum_original", "hashsum_resized", "original_size", "resized_size", "image"]
        )
    )

    return image_df

# def process_image(row_data) -> np.ndarray:
#     image_bytes = row_data["image"]
#     try:
#         original_height, original_width = row_data["original_size"]
#         img_array = np.frombuffer(image_bytes, dtype=np.uint8).reshape((original_height, original_width, 3))
#     except Exception as e:
#         resized_height, resized_width = row_data["resized_size"]
#         img_array = np.frombuffer(image_bytes, dtype=np.uint8).reshape((resized_height, resized_width, 3))
    
#     # Convert BGR to RGB
#     return img_array[..., ::-1]


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

    raise ValueError("Image size does not match any expected dimensions.")

def decode_image_to_pil(row_data) -> Image.Image:
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
    pil_image = Image.fromarray(image_array, mode="RGB")

    return pil_image

def show_image_table(index: int, image_df: pd.DataFrame) -> None:
    row_data = image_df.iloc[index]
    img_rgb = decode_image(row_data)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    plt.subplots_adjust(wspace=0.8)

    ax[0].imshow(img_rgb)
    ax[0].axis('off')

    table_data = [[field, value] for field, value in row_data.drop(['original_size', 'resized_size', 'image', 'path']).items()]
    ax[1].axis('off')
    table = ax[1].table(cellText=table_data, colLabels=["Field", "Value"], cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(2.3, 2.3)

    plt.show()


def show_images(image_df: pd.DataFrame, 
                index: int | list[int] = None, 
                cols: int = 1, 
                size: int = 10) -> None:
    """
    Display one or multiple images from a Pandas or Spark DataFrame.

    Parameters:
    - image_df: pd.DataFrame 
    - index: int | list[int] | None -> Index or list of indices to display. If None, all images are shown.
    - cols: int -> Number of columns in the grid layout.
    - size: int -> Size of each image in inches.

    If a single index is provided, displays one image.
    If a list of indices is provided, displays only those images in a grid.
    If no index is provided, displays all images in a grid.
    """
    
    if index is not None:
        if isinstance(index, int):
            index = [index]  # Convert single index to list
        image_df = image_df.iloc[index]
            

    num_images = len(image_df)
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    axes = axes.flatten() if num_images > 1 else [axes]

    for i, ax in enumerate(axes):
        if i < num_images:
            row_data = image_df.iloc[i]
            img_rgb = decode_image_to_pil(row_data)  # Use decode function
            ax.imshow(img_rgb)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


def show_image_interact(image_df) -> None:
    interact(
        show_images,
        index=(0, len(image_df) - 1),
        image_df=fixed(image_df)
    )


def compress_dir(dir_path, tar_file_path, compress=True):
    """
    Replace a folder with its TAR or TAR.GZ file.

    Parameters:
    - dir_path (str): Path to the folder to archive and delete.
    - tar_file_path (str): Path for the resulting TAR file (without .tar/.tar.gz extension).
    - compress (bool): Whether to compress the archive with Gzip (default True).
    """
    mode = "w:gz" if compress else "w"  # Compression mode
    extension = ".tar.gz" if compress else ".tar"
    tar_file_path += extension  # Add appropriate extension

    try:
        # Step 1: Create the TAR archive
        with tarfile.open(tar_file_path, mode) as tar:
            tar.add(dir_path, arcname=os.path.basename(dir_path))
        print(f"Successfully created TAR file: {tar_file_path}")

        # Step 2: Verify the TAR file creation
        if os.path.exists(tar_file_path):
            # Step 3: Delete the original folder
            shutil.rmtree(dir_path)
            print(f"Original folder '{dir_path}' deleted after archiving.")
        else:
            print(f"Error: TAR file not created. Folder '{dir_path}' was not deleted.")
    except Exception as e:
        print(f"Error during archiving or deletion: {e}")

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
        try:
            # Decode the image and get the UUID
            pil_image = decode_image_to_pil(row)
            uuid = row["uuid"]
            
            # Define the output file path and save the image
            output_path = os.path.join(partition_output_dir, f"{uuid}.{format}")
            pil_image.save(output_path)
            
        except Exception as e:
            print(f"Error saving image for UUID {row['uuid']}: {e}")
    print(f"Written {len(rows)} images to {partition_output_dir}")

    # Compresss folder into tar files
    if compress:
        print(f"Compressing...")        
        compress_dir(
            dir_path = partition_output_dir,
            tar_file_path = output_dir,
            compress = True
        )
        print(f"Images compressed into {output_dir}")
