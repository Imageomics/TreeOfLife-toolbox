import hashlib
import os
import uuid
from typing import List

import cv2
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

from TreeOfLife_toolbox.main.config import Config
from TreeOfLife_toolbox.main.filters import FilterRegister, SparkFilterToolBase
from TreeOfLife_toolbox.main.runners import MPIRunnerTool, RunnerRegister
from TreeOfLife_toolbox.main.schedulers import DefaultScheduler, SchedulerRegister
from TreeOfLife_toolbox.main.utils import load_dataframe


@udf(returnType=StringType())
def get_uuid():
    """
    Generate a random UUID string for uniquely identifying each image entry.
    
    Returns:
        str: A string representation of a random UUID.
    """
    return str(uuid.uuid4())


@FilterRegister("lila_extra_noaa_processing")
class LilaExtraNoaaFilter(SparkFilterToolBase):
    """
    Filter to process LILA NOAA datasets and prepare them for further processing.
    
    This filter loads data from the input CSV file, standardizes column names,
    generates unique identifiers, and splits the data into batches for parallel processing.
    
    Attributes:
        filter_name (str): Name of the filter, used as identifier in the toolbox.
        og_images_root (str): Root path to the original NOAA images, read from config.
    """
    def __init__(self, cfg: Config, spark: SparkSession = None):
        """
        Initialize the LilaExtraNoaaFilter with configuration and spark session.
        
        Args:
            cfg (Config): Configuration object containing parameters for the filter.
            spark (SparkSession, optional): Existing SparkSession to use. If None, a new one will be created.
        """
        super().__init__(cfg, spark)
        self.filter_name: str = "lila_extra_noaa_processing"
        self.og_images_root = self.config["og_images_root"]

    def run(self):
        """
        Execute the filtering process on LILA NOAA dataset.
        
        This method:
        1. Loads the input dataset and renames columns to match standard format
        2. Prepends the image root path to the identifier
        3. Sets server name to 'noaa' 
        4. Generates unique UUIDs for each row
        5. Splits the dataset into batches (partitions)
        6. Saves the filtered data for downstream processing
        """
        # Load the multimedia dataframe and standardize column names
        multimedia_df = (
            load_dataframe(self.spark, self.config["path_to_input"])
            .repartition(20)
            .withColumnsRenamed(
                {
                    "detection_id": "source_id",
                    "detection_type": "life_stage",
                    "rgb_left": "left",
                    "rgb_right": "right",
                    "rgb_top": "top",
                    "rgb_bottom": "bottom",
                }
            )
        )

        # Construct full image paths
        multimedia_df_prep = multimedia_df.withColumn(
            "identifier", F.concat(F.lit(self.og_images_root), F.col("rgb_image_path"))
        )

        # Set server name and generate UUID for each row
        multimedia_df_prep = multimedia_df_prep.withColumn("server_name", F.lit("noaa"))
        multimedia_df_prep = multimedia_df_prep.withColumn("uuid", get_uuid())

        columns = multimedia_df_prep.columns

        self.logger.info("Starting batching")

        # Group by server name and calculate batch counts
        servers_grouped = (
            multimedia_df_prep.select("server_name")
            .groupBy("server_name")
            .count()
            .withColumn(
                "batch_count",
                F.floor(
                    F.col("count") / self.config["downloader_parameters"]["batch_size"]
                ),
            )
        )

        # Partition the dataset
        window_part = Window.partitionBy("server_name").orderBy("server_name")
        master_df_filtered = (
            multimedia_df_prep.withColumn(
                "row_number", F.row_number().over(window_part)
            )
            .join(servers_grouped, ["server_name"])
            .withColumn("partition_id", F.col("row_number") % F.col("batch_count"))
            .withColumn(
                "partition_id",
                (
                    F.when(F.col("partition_id").isNull(), 0).otherwise(
                        F.col("partition_id")
                    )
                ),
            )
            .select(*columns, "partition_id")
        )

        self.logger.info("Writing to parquet")

        # Write partitioned data as parquet files
        (
            master_df_filtered.repartition("server_name", "partition_id")
            .write.partitionBy("server_name", "partition_id")
            .mode("overwrite")
            .format("parquet")
            .save(self.urls_path)
        )

        # Prepare the filter table with selected columns
        filtered_df = master_df_filtered.select(
            "uuid",
            "source_id",
            "identifier",
            "left",
            "right",
            "top",
            "bottom",
            "server_name",
            "partition_id",
        )

        # Save filter table for scheduler
        self.save_filter(filtered_df)

        self.logger.info("Finished batching")
        self.logger.info(f"Too small images number: {master_df_filtered.count()}")


@SchedulerRegister("lila_extra_noaa_processing")
class LilaExtraNoaaScheduleCreation(DefaultScheduler):
    """
    Scheduler for the LILA NOAA processing pipeline.
    
    This scheduler leverages the default scheduling mechanism with a specific filter name.
    It creates a schedule for distributing the workload across available workers.
    """
    def __init__(self, cfg: Config):
        """
        Initialize the scheduler with configuration.
        
        Args:
            cfg (Config): Configuration object containing parameters for scheduling.
        """
        super().__init__(cfg)

        self.filter_name: str = "lila_extra_noaa_processing"


@RunnerRegister("lila_extra_noaa_processing")
class LilaExtraNoaaRunner(MPIRunnerTool):
    """
    MPI-based runner for processing LILA NOAA images.
    
    This runner processes the images according to the schedule created by the scheduler,
    cropping images based on bounding box coordinates and saving them as parquet files.
    
    Attributes:
        filter_name (str): Name of the filter, matching the one used in registry.
        data_scheme (List[str]): Column names for data processing.
        verification_scheme (List[str]): Column names for schedule verification.
        total_time (int): Maximum processing time in seconds before timeout.
    """
    def __init__(self, cfg: Config):
        """
        Initialize the LILA NOAA runner.
        
        Args:
            cfg (Config): Configuration object containing parameters for the runner.
        """
        super().__init__(cfg)

        self.filter_name: str = "lila_extra_noaa_processing"
        self.data_scheme: List[str] = [
            "uuid",
            "source_id",
            "left",
            "right",
            "top",
            "bottom",
            "server_name",
            "partition_id",
        ]
        self.verification_scheme: List[str] = ["server_name", "partition_id"]
        self.total_time = 1000

    def apply_filter(
            self, filtering_df: pd.DataFrame, server_name: str, partition_id: str
    ) -> int:
        """
        Process a batch of images from the LILA NOAA dataset.
        
        For each image in the batch:
        1. Loads the original image from the provided path
        2. Crops the image according to the bounding box coordinates
        3. Computes hash values for both original and cropped images
        4. Saves the processed data as a parquet file
        
        Args:
            filtering_df (pd.DataFrame): DataFrame containing images to process
            server_name (str): Name of the server (always 'noaa' for this dataset)
            partition_id (str): ID of the partition being processed
            
        Returns:
            int: Number of successfully processed images
            
        Raises:
            TimeoutError: If processing exceeds the allocated time
        """
        self.is_enough_time()

        # Create output directory for this partition
        parquet_folder_path = os.path.join(
            self.downloaded_images_path,
            f"server_name={server_name}",
            f"partition_id={partition_id}",
        )
        os.makedirs(parquet_folder_path)
        parquet_path = os.path.join(parquet_folder_path, "successes.parquet")

        images = []
        for _, row in filtering_df.iterrows():
            # Initialize an entry for the current image
            new_entry = {
                "uuid": row["uuid"],
                "source_id": row["source_id"],
                "identifier": row["identifier"],
                "is_license_full": False,
                "license": None,
                "source": None,
                "title": None,
                "original_size": "",
                "resized_size": "",
                "hashsum_original": "",
                "hashsum_resized": "",
                "image": "",
            }

            # Load and crop the image according to bounding box coordinates
            image = cv2.imread(row["identifier"])
            cropped = image[row["bottom"]: row["top"], row["left"]: row["right"]]
            cropped_binary = cropped.tobytes()

            # Set additional metadata for the image
            new_entry["original_size"] = image.shape[:2]
            new_entry["resized_size"] = cropped.shape[:2]
            new_entry["hashsum_original"] = hashlib.md5(image.tobytes()).hexdigest()
            new_entry["hashsum_resized"] = hashlib.md5(cropped_binary).hexdigest()
            new_entry["image"] = cropped_binary

            images.append(new_entry)

        # Create DataFrame from processed images
        filtered_parquet = pd.DataFrame(images)

        self.is_enough_time()

        # Save processed images as parquet file
        filtered_parquet.to_parquet(
            parquet_path, index=False, compression="zstd", compression_level=3
        )

        return len(filtered_parquet)
