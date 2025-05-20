import os
from typing import List

import pandas as pd

from TreeOfLife_toolbox.main.config import Config
from TreeOfLife_toolbox.main.filters import FilterRegister, SparkFilterToolBase
from TreeOfLife_toolbox.main.runners import MPIRunnerTool, RunnerRegister
from TreeOfLife_toolbox.main.schedulers import DefaultScheduler, SchedulerRegister
from TreeOfLife_toolbox.main.utils import load_dataframe


@FilterRegister("filter_out_by_uuid")
class FilterOutByUUIDFilter(SparkFilterToolBase):
    """
    Filter that creates a dataset of UUIDs that need to be filtered out, along with their file paths.
    
    This class implements the filtering step of the filter_out_by_uuid tool. It takes a table of UUIDs
    to be filtered out, and a lookup table mapping UUIDs to file paths. It joins these tables and 
    creates a filter table that will be used by the runner to remove the specified UUIDs from the dataset.
    
    Attributes:
        filter_name (str): The name of the filter, used for identifying the tool.
    """
    def __init__(self, cfg: Config):
        """
        Initialize the FilterOutByUUIDFilter.
        
        Args:
            cfg (Config): Configuration object containing settings for the filter.
        """
        super().__init__(cfg)

        self.filter_name: str = "filter_out_by_uuid"

    def run(self):
        """
        Execute the filtering process.
        
        Loads the UUID table and lookup table, joins them on the UUID field, and writes the resulting
        table to the filter table folder. This table will contain all UUIDs to be filtered out along
        with the paths where they are located.
        """
        # Load the table containing UUIDs to be filtered out
        uuid_table_df = load_dataframe(
            self.spark, self.config["uuid_table_path"]
        ).repartition(20)
        
        # Load the lookup table containing mapping of UUIDs to file paths
        lookup_table_df = load_dataframe(
            self.spark, self.config["look_up_table_path"]
        ).repartition(20)

        # Join the tables to get UUIDs with their file paths
        merged_df = uuid_table_df.join(lookup_table_df, on="uuid", how="left")

        # Save the merged dataframe as a CSV to the filter table directory
        (
            merged_df.repartition(1).write.csv(
                os.path.join(self.tools_path, self.filter_name, "filter_table"),
                header=True,
                mode="overwrite",
            )
        )


@SchedulerRegister("filter_out_by_uuid")
class FilterOutByUUIDScheduleCreation(DefaultScheduler):
    """
    Scheduler for the filter_out_by_uuid tool.
    
    This class creates a schedule for the runner step of the filter_out_by_uuid tool.
    It uses the 'path' column from the filter table to determine which files need
    to be processed by the runners.
    
    Attributes:
        filter_name (str): The name of the filter, used for identifying the tool.
        scheme (List[str]): The columns to use for scheduling.
    """
    def __init__(self, cfg: Config):
        """
        Initialize the FilterOutByUUIDScheduleCreation scheduler.
        
        Args:
            cfg (Config): Configuration object containing settings for the scheduler.
        """
        super().__init__(cfg)

        self.filter_name: str = "filter_out_by_uuid"
        self.scheme = ["path"]


@RunnerRegister("filter_out_by_uuid")
class FilterOutByUUIDRunner(MPIRunnerTool):
    """
    Runner for the filter_out_by_uuid tool.
    
    This class implements the worker step of the filter_out_by_uuid tool. It performs the actual
    filtering operation on the dataset by removing entries with UUIDs that need to be filtered out
    from the specified parquet files.
    
    Attributes:
        filter_name (str): The name of the filter, used for identifying the tool.
        data_scheme (List[str]): Columns needed from the filter table for processing.
        verification_scheme (List[str]): Columns used to track the progress of the filtering.
        total_time (int): Maximum time (in seconds) to allow for the filtering process.
    """
    def __init__(self, cfg: Config):
        """
        Initialize the FilterOutByUUIDRunner.
        
        Args:
            cfg (Config): Configuration object containing settings for the runner.
        """
        super().__init__(cfg)
        self.filter_name: str = "filter_out_by_uuid"
        self.data_scheme: List[str] = ["uuid", "path"]
        self.verification_scheme: List[str] = ["path"]
        self.total_time = 150

    def apply_filter(self, filtering_df: pd.DataFrame, file_path: str) -> int:
        """
        Apply the filtering operation to a specific file.
        
        This method reads a parquet file and filters out rows corresponding to the UUIDs
        in the filtering dataframe. It then writes the filtered data back to the original file.
        
        Args:
            filtering_df (pd.DataFrame): DataFrame containing UUIDs to filter out.
            file_path (str): Path to the parquet file to be filtered.
            
        Returns:
            int: The number of entries remaining in the file after filtering.
            
        Raises:
            TimeoutError: If there is not enough time left to complete the operation.
        """
        # Check if enough time is left to perform the operation
        self.is_enough_time()

        # Check if the file exists
        if not os.path.exists(file_path):
            self.logger.info(f"Path doesn't exists: {file_path}")
            return 0

        # Read the parquet file, excluding rows with UUIDs in the filtering dataframe
        filtered_parquet = pd.read_parquet(
            file_path, filters=[("uuid", "not in", filtering_df["uuid"])]
        )

        # Check again if enough time is left to complete the operation
        self.is_enough_time()

        # Log if all entries were filtered out
        if len(filtered_parquet) == 0:
            self.logger.info(f"Fully filtered out: {file_path}")

        # Save the filtered dataframe back to the original file
        filtered_parquet.to_parquet(
            file_path, index=False, compression="zstd", compression_level=3
        )

        return len(filtered_parquet)

    def runner_fn(self, df_local: pd.DataFrame) -> int:
        """
        Process a chunk of the schedule and apply the filter.
        
        This method is called for each group in the schedule. It extracts the file path
        from the first row of the dataframe and applies the filter to that file.
        
        Args:
            df_local (pd.DataFrame): DataFrame containing UUIDs to filter out for a specific file path.
            
        Returns:
            int: 1 if successful, 0 if an error occurred.
        """
        filtering_df = df_local.reset_index(drop=True)
        file_path = filtering_df.iloc[0]["path"]
        try:
            filtered_parquet_length = self.apply_filter(filtering_df, file_path)
        except NotImplementedError:
            raise NotImplementedError("Filter function wasn't implemented")
        except Exception as e:
            self.logger.exception(e)
            self.logger.error(f"Error occurred: {e}")
            return 0
        else:
            print(f"{file_path}", end="\n", file=self.verification_IO)
            self.logger.debug(
                f"Completed filtering: {file_path} with {filtered_parquet_length}"
            )
            return 1
