"""
Parquet to HDF5 WebP conversion tool for TreeOfLife dataset.

This tool converts TreeOfLife Parquet files (containing raw image bytes) into:
- HDF5 files with UUID-indexed lossless WebP compressed images
- Separate Parquet metadata files (without image data)

Uses hardcoded WebP lossless compression with method=6 for optimal space efficiency.
"""

import os
import glob
import time
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np
import polars as pl
import h5py
from PIL import Image
import io
from tqdm import tqdm

from TreeOfLife_toolbox.main.config import Config
from TreeOfLife_toolbox.main.filters import SparkFilterToolBase, FilterRegister
from TreeOfLife_toolbox.main.runners import MPIRunnerTool, RunnerRegister
from TreeOfLife_toolbox.main.schedulers import DefaultScheduler, SchedulerRegister
import pyspark.sql.functions as func


@FilterRegister("parquet_to_hdf5")
class ParquetToHDF5Filter(SparkFilterToolBase):
    """
    Spark-based filter for discovering and batching Parquet files for HDF5 conversion.

    This class uses Spark to efficiently discover TreeOfLife Parquet files containing
    image data and creates work batches for distributed conversion to HDF5 format.
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.filter_name: str = "parquet_to_hdf5"

        # Get batch size from config (files per conversion job)
        self.batch_size: int = self.config.get("tools_parameters", {}).get("batch_size", 5)

    def run(self):
        """
        Execute Spark-based file discovery and batching.

        Discovers all parquet files in the input path and creates batches
        for distributed processing. Saves the schedule as CSV files.
        """
        input_path = self.config.get("path_to_input", "")
        output_path = self.config.get("path_to_output_folder", "")

        if not input_path or not output_path:
            raise ValueError("path_to_input and path_to_output_folder must be specified in config")

        self.logger.info(f"Discovering parquet files in: {input_path}")

        # Create list of files to process
        parquet_files = []

        if os.path.isfile(input_path) and input_path.endswith('.parquet'):
            # Single file input
            parquet_files.append(input_path)
        elif os.path.isdir(input_path):
            # Directory input - find all parquet files recursively
            pattern = os.path.join(input_path, "**", "*.parquet")
            parquet_files.extend(glob.glob(pattern, recursive=True))
        else:
            raise ValueError(f"Input path {input_path} is not a valid file or directory")

        if not parquet_files:
            raise ValueError(f"No parquet files found in {input_path}")

        self.logger.info(f"Found {len(parquet_files)} parquet files")

        # Create Spark DataFrame with file information
        file_data = []
        for pq_file in parquet_files:
            base_name = Path(pq_file).stem
            # Maintain relative directory structure in output
            if os.path.isdir(input_path):
                rel_path = os.path.relpath(os.path.dirname(pq_file), input_path)
                output_dir = os.path.join(output_path, rel_path) if rel_path != '.' else output_path
            else:
                output_dir = output_path

            file_data.append({
                'input_path': pq_file,
                'output_dir': output_dir,
                'base_name': base_name
            })

        # Convert to Spark DataFrame
        files_df = self.spark.createDataFrame(file_data)

        # Add batch assignments
        total_files = len(parquet_files)
        batch_count = max(1, (total_files + self.batch_size - 1) // self.batch_size)

        self.logger.info(f"Creating {batch_count} batches with up to {self.batch_size} files each")

        # Add row numbers and batch IDs
        from pyspark.sql.window import Window
        window_spec = Window.orderBy("input_path")

        batched_df = (files_df
                     .withColumn("row_number", func.row_number().over(window_spec))
                     .withColumn("batch_id", (func.col("row_number") - 1) / self.batch_size)
                     .withColumn("batch_id", func.floor(func.col("batch_id")).cast("int"))
                     .drop("row_number"))

        # Save the batched schedule using the framework's save_filter method
        self.save_filter(batched_df)

        self.logger.info(f"Saved conversion schedule with {total_files} files in {batch_count} batches")


@SchedulerRegister("parquet_to_hdf5")
class ParquetToHDF5Scheduler(DefaultScheduler):
    """
    Scheduler for organizing Parquet to HDF5 conversion jobs.

    Creates a combined schedule file from all filter results.
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.filter_name: str = "parquet_to_hdf5"
        # Override scheme to match our filter output
        self.scheme: List[str] = ["base_name", "input_path", "output_dir", "batch_id"]


@RunnerRegister("parquet_to_hdf5")
class ParquetToHDF5Runner(MPIRunnerTool):
    """
    MPI-based runner for executing Parquet to HDF5 conversions.

    This class uses the framework's MPIRunnerTool pattern to process batches
    of parquet files in parallel, converting them to HDF5 format with:
    1. WebP lossless compression (method=6)
    2. UUID-indexed HDF5 storage
    3. Separate metadata Parquet files
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.filter_name: str = "parquet_to_hdf5"

        # Define the data schema for batch processing (from filter stage)
        # Note: After merge, pandas adds suffixes to resolve column conflicts
        # We use the _y columns (from schedule) as they have the authoritative values
        self.data_scheme = ["input_path", "base_name_y", "output_dir_y", "batch_id_y"]

        # Define verification schema for tracking results (only common columns for merging)
        self.verification_scheme = ["input_path"]

        self.total_time = 3600  # 1 hour max per job

        # WebP settings - hardcoded for consistency
        self.image_format = "webp"
        self.webp_method = 6
        self.webp_lossless = True
        self.webp_quality = 100

        # Processing settings
        self.num_cores = self.config["tools_parameters"].get("cpu_per_worker", 32)

    def apply_filter(self, filtering_df: pd.DataFrame, batch_id: int) -> int:
        """
        Process a batch of parquet files for HDF5 conversion.

        This method is called by the MPIRunnerTool framework for each batch.
        It converts a group of parquet files to HDF5 format with WebP compression.

        Args:
            filtering_df (pd.DataFrame): DataFrame with files in this batch
            batch_id (int): ID of the current batch being processed

        Returns:
            int: Number of files successfully processed
        """
        self.is_enough_time()

        self.logger.info(f"Processing batch {batch_id} with {len(filtering_df)} files")

        successful_count = 0

        # Process each file in the batch
        for _, row in filtering_df.iterrows():
            result = self.process_single_file(
                row["input_path"],
                row["output_dir"],
                row["base_name"]
            )

            # Log the result
            if result["success"]:
                self.logger.info(
                    f"✅ Converted {result['input_path']}: "
                    f"{result['successful_images']} images, "
                    f"{result['compression_ratio']:.1f}:1 compression"
                )
                successful_count += 1
            else:
                self.logger.error(
                    f"❌ Failed {result['input_path']}: {result['error_message']}"
                )

            # Store verification data for later writing (to avoid pickle issues)
            # TODO: Write verification records after batch completion

        self.logger.info(f"Batch {batch_id} completed: {successful_count}/{len(filtering_df)} files successful")
        return successful_count

    def runner_fn(self, df_local: pd.DataFrame) -> int:
        """
        Override runner_fn to handle our specific data structure.

        Our tool doesn't use server_name/partition_id like other tools,
        instead we group by batch_id and process files within each batch.
        """
        filtering_df = df_local.reset_index(drop=True)

        # Rename columns back to original names (remove _y suffix from merge)
        filtering_df = filtering_df.rename(columns={
            'base_name_y': 'base_name',
            'output_dir_y': 'output_dir',
            'batch_id_y': 'batch_id'
        })

        # Get batch_id for logging
        batch_id = filtering_df.iloc[0]["batch_id"]

        try:
            # Call our apply_filter method with batch_id as the identifier
            filtered_count = self.apply_filter(filtering_df, batch_id)
            return filtered_count
        except Exception as e:
            self.logger.error(f"Error processing batch {batch_id}: {e}")
            return 0

    def convert_image(self, image_bytes: bytes, width: int, height: int) -> bytes:
        """Convert raw BGR bytes to WebP lossless format."""
        # Convert BGR -> RGB and create PIL Image
        img_np = np.frombuffer(image_bytes, dtype=np.uint8).reshape((height, width, 3))
        img_np_rgb = img_np[:, :, ::-1]  # BGR -> RGB (more efficient than fancy indexing)
        img = Image.fromarray(img_np_rgb)

        buffer = io.BytesIO()
        img.save(buffer, format='WebP',
                lossless=self.webp_lossless,
                quality=self.webp_quality,
                method=self.webp_method)
        return buffer.getvalue()

    @staticmethod
    def convert_chunk(chunk_id: int, records: List[dict], output_path: Path, image_format: str, webp_method: int, webp_lossless: bool, webp_quality: int) -> Tuple[int, int, int, int, int]:
        """
        Convert a chunk of records to HDF5.

        Returns:
            Tuple[chunk_id, successful_images, failed_images, original_bytes, compressed_bytes]
        """
        successful_images = 0
        failed_images = 0
        total_original_bytes = 0
        total_compressed_bytes = 0

        with h5py.File(output_path, 'w') as h5f:
            h5f.attrs['chunk_id'] = chunk_id
            h5f.attrs['image_format'] = image_format
            h5f.attrs['webp_method'] = webp_method
            h5f.attrs['webp_lossless'] = webp_lossless
            images_group = h5f.create_group("images")

            for rec in records:
                uuid = rec.get("uuid")
                image_bytes = rec.get("image")
                dims = rec.get("resized_size")

                if not all([uuid, image_bytes, dims]) or len(dims) < 2:
                    failed_images += 1
                    continue

                # EOL data has simple integer dimensions
                height, width = dims[0], dims[1]

                if height <= 0 or width <= 0:
                    failed_images += 1
                    continue

                try:
                    # Convert to WebP
                    total_original_bytes += len(image_bytes)

                    # Convert BGR -> RGB and create PIL Image
                    img_np = np.frombuffer(image_bytes, dtype=np.uint8).reshape((height, width, 3))
                    img_np_rgb = img_np[:, :, ::-1]  # BGR -> RGB
                    img = Image.fromarray(img_np_rgb)

                    buffer = io.BytesIO()
                    img.save(buffer, format='WebP',
                            lossless=webp_lossless,
                            quality=webp_quality,
                            method=webp_method)
                    compressed_bytes = buffer.getvalue()

                    total_compressed_bytes += len(compressed_bytes)

                    # Store in HDF5
                    compressed_array = np.frombuffer(compressed_bytes, dtype=np.uint8)
                    images_group.create_dataset(
                        uuid,
                        data=compressed_array,
                        fletcher32=True
                    )
                    successful_images += 1

                except Exception as e:
                    failed_images += 1

            # Store statistics
            h5f.attrs['successful_images'] = successful_images
            h5f.attrs['failed_images'] = failed_images
            h5f.attrs['original_bytes'] = total_original_bytes
            h5f.attrs['compressed_bytes'] = total_compressed_bytes

        return chunk_id, successful_images, failed_images, total_original_bytes, total_compressed_bytes

    def process_single_file(self, input_path: str, output_dir: str, base_name: str) -> dict:
        """
        Process a single Parquet file to HDF5 + metadata.

        Returns:
            dict: Processing results including success status, statistics, etc.
        """
        start_time = time.time()
        result = {
            "input_path": input_path,
            "output_hdf5_path": "",
            "output_metadata_path": "",
            "success": False,
            "error_message": "",
            "processing_time": 0,
            "total_images": 0,
            "successful_images": 0,
            "failed_images": 0,
            "compression_ratio": 0
        }

        try:
            # Setup paths
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            chunk_dir = output_dir / f"{base_name}_chunks"
            chunk_dir.mkdir(exist_ok=True)

            final_hdf5_path = output_dir / f"{base_name}_images.h5"
            final_metadata_path = output_dir / f"{base_name}_metadata.parquet"

            result["output_hdf5_path"] = str(final_hdf5_path)
            result["output_metadata_path"] = str(final_metadata_path)

            # Read Parquet file
            df = pl.read_parquet(input_path)
            total_records = df.height
            result["total_images"] = total_records

            if total_records == 0:
                result["error_message"] = "No records found in Parquet file"
                return result

            # Split into chunks for parallel processing
            chunk_size = (total_records + self.num_cores - 1) // self.num_cores
            tasks = []

            for i in range(self.num_cores):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, total_records)
                if start_idx >= total_records:
                    continue

                chunk_df = df.slice(start_idx, end_idx - start_idx)
                records = chunk_df.to_dicts()
                output_path = chunk_dir / f"chunk_{i:02d}_images.h5"
                tasks.append((i, records, output_path, self.image_format, self.webp_method, self.webp_lossless, self.webp_quality))

            # Process chunks in parallel
            total_successful = 0
            total_failed = 0
            total_original_bytes = 0
            total_compressed_bytes = 0

            with ProcessPoolExecutor(max_workers=self.num_cores) as executor:
                future_to_chunk = {
                    executor.submit(ParquetToHDF5Runner.convert_chunk, *task): task[0]
                    for task in tasks
                }

                for future in as_completed(future_to_chunk):
                    chunk_id, successful, failed, orig_bytes, comp_bytes = future.result()
                    total_successful += successful
                    total_failed += failed
                    total_original_bytes += orig_bytes
                    total_compressed_bytes += comp_bytes

            result["successful_images"] = total_successful
            result["failed_images"] = total_failed

            if total_compressed_bytes > 0:
                result["compression_ratio"] = total_original_bytes / total_compressed_bytes

            # Merge chunks into final HDF5
            self.merge_chunks(chunk_dir, final_hdf5_path, base_name)

            # Create metadata file
            metadata_df = df.drop("image")
            metadata_df.write_parquet(final_metadata_path, compression="zstd")

            # Cleanup chunks
            import shutil
            shutil.rmtree(chunk_dir)

            result["success"] = True

        except Exception as e:
            result["error_message"] = str(e)
            result["success"] = False

        finally:
            result["processing_time"] = time.time() - start_time

        return result

    def merge_chunks(self, chunk_dir: Path, output_path: Path, base_name: str):
        """Merge chunk HDF5 files into final output."""
        chunk_files = sorted(chunk_dir.glob("chunk_*_images.h5"))

        if not chunk_files:
            raise ValueError("No chunk files found to merge")

        with h5py.File(output_path, 'w') as output_f:
            images_group = output_f.create_group("images")

            total_images = 0
            total_failed = 0
            total_original_bytes = 0
            total_compressed_bytes = 0

            for chunk_file in chunk_files:
                with h5py.File(chunk_file, 'r') as chunk_f:
                    # Copy all images
                    chunk_images = chunk_f['images']
                    for uuid in chunk_images.keys():
                        chunk_f.copy(f'images/{uuid}', images_group)

                    # Accumulate statistics
                    total_images += chunk_f.attrs.get('successful_images', 0)
                    total_failed += chunk_f.attrs.get('failed_images', 0)
                    total_original_bytes += chunk_f.attrs.get('original_bytes', 0)
                    total_compressed_bytes += chunk_f.attrs.get('compressed_bytes', 0)

            # Store final attributes
            output_f.attrs['total_images'] = total_images
            output_f.attrs['failed_images'] = total_failed
            output_f.attrs['original_image_bytes'] = total_original_bytes
            output_f.attrs['compressed_image_bytes'] = total_compressed_bytes
            output_f.attrs['compression_ratio'] = total_original_bytes / total_compressed_bytes if total_compressed_bytes > 0 else 0
            output_f.attrs['image_format'] = self.image_format
            output_f.attrs['webp_method'] = self.webp_method
            output_f.attrs['webp_lossless'] = self.webp_lossless

