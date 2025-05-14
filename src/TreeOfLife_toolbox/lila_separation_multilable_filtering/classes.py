import os
import shutil
from typing import List

import pandas as pd

from TreeOfLife_toolbox.main.config import Config
from TreeOfLife_toolbox.main.filters import (
    FilterRegister,
    SparkFilterToolBase,
)
from TreeOfLife_toolbox.main.runners import MPIRunnerTool, RunnerRegister
from TreeOfLife_toolbox.main.schedulers import DefaultScheduler, SchedulerRegister


@FilterRegister("lila_separation_multilable_filtering")
class LilaSeparationFilter(SparkFilterToolBase):
    """
    Filter class for separating multi-labeled images from LILA dataset.
    
    This class handles the initial filtering step of the workflow by:
    1. Copying the multi-label data table to the tool's filter folder
    2. Joining image metadata with the multi-label filter data
    3. Creating a new dataset with only the multi-labeled entries
    
    Attributes:
        filter_name (str): Name of the filter tool
        new_urls_folder (str): Target location for storing filtered metadata/URLs
        data_path (str): Path to CSV containing multi-label image entries
    """

    def __init__(self, cfg: Config):
        """
        Initialize the LILA separation filter.
        
        Args:
            cfg (Config): Configuration object containing paths and parameters
        """
        super().__init__(cfg)

        self.filter_name: str = "lila_separation_multilable_filtering"
        self.new_urls_folder = cfg["new_urls_folder"]
        self.data_path = cfg["multilabel_data_path"]

    def run(self):
        """
        Execute the filtering process.
        
        This method:
        1. Sets up the filter table directory
        2. Copies the multi-label data to the filter table location
        3. Loads the original metadata and filter table into Spark DataFrames
        4. Joins the tables to identify multi-label images
        5. Saves the filtered metadata to the new URLs folder
        """
        filter_table_folder = os.path.join(
            self.tools_path, self.filter_name, "filter_table"
        )
        os.makedirs(filter_table_folder, exist_ok=True)
        filter_table_folder += "/table.csv"

        shutil.copyfile(self.data_path, filter_table_folder)

        metadata_df = self.spark.read.parquet(self.urls_path).drop("partition_id")
        filter_df = self.spark.read.csv(self.data_path, header=True).select(
            "uuid", "partition_id"
        )

        df = metadata_df.join(filter_df, on="uuid", how="inner")

        (
            df.repartition("server_name", "partition_id")
            .write.partitionBy("server_name", "partition_id")
            .mode("overwrite")
            .format("parquet")
            .save(self.new_urls_folder)
        )


@SchedulerRegister("lila_separation_multilable_filtering")
class LilaSeparationScheduleCreation(DefaultScheduler):
    """
    Scheduler class for orchestrating the LILA multi-label separation process.
    
    This class inherits from DefaultScheduler and manages the creation of the
    processing schedule for separating multi-labeled images.
    
    The scheduler creates a list of tasks based on server_name and partition_id
    combinations that need to be processed.
    
    Attributes:
        filter_name (str): Name of the filter tool
    """

    def __init__(self, cfg: Config):
        """
        Initialize the LILA separation scheduler.
        
        Args:
            cfg (Config): Configuration object containing paths and parameters
        """
        super().__init__(cfg)

        self.filter_name: str = "lila_separation_multilable_filtering"


@RunnerRegister("lila_separation_multilable_filtering")
class LilaSeparationRunner(MPIRunnerTool):
    """
    Runner class for executing the LILA multi-label image separation.
    
    This class performs the actual data processing to extract multi-labeled 
    images from the original dataset and store them in a new location with 
    proper metadata.
    
    The runner works in a distributed MPI environment, with each process
    handling a subset of the data partitions.
    
    Attributes:
        filter_name (str): Name of the filter tool
        data_scheme (List[str]): Column structure of the input dataset
        verification_scheme (List[str]): Columns used for verifying task completion
        new_images_path (str): Path where separated images will be stored
        total_time (int): Maximum processing time in seconds before timeout
    """

    def __init__(self, cfg: Config):
        """
        Initialize the LILA separation runner.
        
        Args:
            cfg (Config): Configuration object containing paths and parameters
        """
        super().__init__(cfg)
        self.filter_name: str = "lila_separation_multilable_filtering"

        self.data_scheme: List[str] = [
            "uuid",
            "source_id",
            "uuid_main",
            "source_id_main",
            "server_name",
            "old_partition_id",
            "partition_id",
        ]
        self.verification_scheme: List[str] = ["server_name", "partition_id"]
        self.new_images_path = cfg["new_images_path"]
        self.total_time = 600

    def apply_filter(
        self, filtering_df: pd.DataFrame, server_name: str, partition_id: str
    ) -> int:
        """
        Extract multi-labeled images from the original dataset.
        
        This method processes a specific server/partition combination:
        1. Groups filter data by server_name and old_partition_id
        2. Reads image data from original locations based on UUIDs
        3. Merges image data with filter information
        4. Saves the multi-labeled images to new locations
        
        Args:
            filtering_df (pd.DataFrame): DataFrame containing filter information
            server_name (str): Name of the server to process
            partition_id (str): ID of the partition to process
            
        Returns:
            int: Number of images processed and saved
            
        Raises:
            TimeoutError: If processing exceeds the allocated time limit
        """
        self.is_enough_time()

        # Group filtering data by server and partition to find original image locations
        filtering_df_grouped = filtering_df.groupby(["server_name", "old_partition_id"])
        separated_dict = []
        for name, group in filtering_df_grouped:
            parquet_path = os.path.join(
                self.downloaded_images_path,
                f"server_name={name[0]}",
                f"partition_id={name[1]}",
                "successes.parquet",
            )
            if not os.path.exists(parquet_path):
                self.logger.info(f"Path doesn't exists: {server_name}/{partition_id}")
                continue

            # Read original image data matching the UUIDs in our filter group
            partial_df = pd.read_parquet(
                parquet_path, filters=[("uuid", "in", group["uuid_main"])]
            )
            
            # Merge image data with filter information
            partial_merged_df = pd.merge(
                partial_df,
                group,
                left_on="uuid",
                right_on="uuid_main",
                suffixes=("_x", "_y"),
                sort=False,
                how="right",
            )

            # Select and rename columns for the output dataset
            partial_merged_df = partial_merged_df[
                [
                    "uuid_y",
                    "source_id_y",
                    "identifier",
                    "is_license_full",
                    "license",
                    "source",
                    "title",
                    "hashsum_original",
                    "hashsum_resized",
                    "original_size",
                    "resized_size",
                    "image",
                ]
            ]
            separated_dict.extend(
                partial_merged_df.rename(
                    {"uuid_y": "uuid", "source_id_y": "source_id"}, inplace=True
                ).to_dict("records")
            )

        # Create DataFrame from collected records
        merged_df = pd.DataFrame.from_records(separated_dict)

        self.is_enough_time()

        # Create output directory and save processed data
        save_path = os.path.join(
            self.new_images_path,
            f"server_name={server_name}",
            f"partition_id={partition_id}",
        )
        os.makedirs(save_path, exist_ok=True)

        if len(merged_df) == 0:
            self.logger.info(f"Empty: {server_name}/{partition_id}")

        # Save processed data to parquet file
        merged_df.to_parquet(
            save_path + "/successes.parquet",
            index=False,
            compression="zstd",
            compression_level=3,
        )

        return len(merged_df)
