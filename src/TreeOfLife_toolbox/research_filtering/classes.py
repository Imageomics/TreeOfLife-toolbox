import os
import re
from typing import List

import pandas as pd

from TreeOfLife_toolbox.main.config import Config
from TreeOfLife_toolbox.main.filters import FilterRegister, SparkFilterToolBase
from TreeOfLife_toolbox.main.runners import MPIRunnerTool, RunnerRegister
from TreeOfLife_toolbox.main.schedulers import DefaultScheduler, SchedulerRegister


@FilterRegister("research_filtering")
class ResearchFilteringFilter(SparkFilterToolBase):
    """
    A filter class for research data filtering operations using Spark.
    
    This class filters a dataset based on the basisOfRecord field and creates
    a filtered table that identifies entries to be removed from the TreeOfLife dataset.
    
    Attributes:
        filter_name (str): Name identifier for the filter.
        string_to_remove (str): String prefix to remove from file paths.
    """

    def __init__(self, cfg: Config):
        """
        Initialize the research filtering filter.
        
        Args:
            cfg (Config): Configuration object containing parameters for filtering.
        """
        super().__init__(cfg)

        self.filter_name: str = "research_filtering"
        self.string_to_remove = "file:/"

    def run(self):
        """
        Run the filtering process.
        
        This method:
        1. Reads occurrence data and selects relevant columns
        2. Reads data files and extracts paths
        3. Filters occurrences based on the basisOfRecord value specified in config
        4. Joins filtered occurrences with data to create a filtered table
        5. Saves the filtered table for later processing
        """
        import pyspark.sql.functions as func

        occurrences_df = (
            self.spark.read.parquet(self.config["occurrences_path"])
            .select("gbifID", "basisOfRecord")
            .withColumnRenamed("gbifID", "source_id")
        )

        data_df = (
            self.spark.read.option("basePath", self.config["data_path"])
            .parquet(f"{self.config['data_path']}/source=*/server=*/data_*.parquet")
            .select("uuid", "source_id")
            .withColumn(
                "path",
                func.substring(
                    func.input_file_name(), len(self.string_to_remove), 2000000
                ),
            )
        )

        occurrences_df_filtered = occurrences_df.where(
            occurrences_df["basisOfRecord"].contains(self.config["basis_of_record"])
        )
        data_merged = data_df.join(occurrences_df_filtered, on="source_id", how="inner")

        (
            data_merged.repartition(1).write.csv(
                os.path.join(self.tools_path, self.filter_name, "filter_table"),
                header=True,
                mode="overwrite",
            )
        )


@SchedulerRegister("research_filtering")
class ResearchFilteringScheduleCreation(DefaultScheduler):
    """
    Scheduler class for research filtering operations.
    
    Creates the schedule for distributed processing of filtered data.
    
    Attributes:
        filter_name (str): Name identifier for the scheduler.
        scheme (List[str]): Column schema for scheduling, specifies which fields
                           are used to group tasks for distribution.
    """

    def __init__(self, cfg: Config):
        """
        Initialize the research filtering scheduler.
        
        Args:
            cfg (Config): Configuration object containing parameters for scheduling.
        """
        super().__init__(cfg)

        self.filter_name: str = "research_filtering"
        self.scheme = ["path"]


@RunnerRegister("research_filtering")
class ResearchFilteringRunner(MPIRunnerTool):
    """
    Runner class that applies the research filtering operation across distributed nodes.
    
    This runner filters out data entries from parquet files based on UUIDs
    identified in the filtering step.
    
    Attributes:
        filter_name (str): Name identifier for the runner.
        server_pattern (str): Regex pattern to extract server name from a path.
        source_pattern (str): Regex pattern to extract source name from a path.
        data_scheme (List[str]): Schema for data columns used in processing.
        verification_scheme (List[str]): Schema for verification columns.
        total_time (int): Maximum execution time in seconds.
        save_path_folder (str): Folder path to save filtered results.
    """

    server_pattern = r"server=([^/]+)"
    source_pattern = r"source=([^/]+)"

    def __init__(self, cfg: Config):
        """
        Initialize the research filtering runner.
        
        Args:
            cfg (Config): Configuration object containing parameters for the runner.
        """
        super().__init__(cfg)

        self.filter_name: str = "research_filtering"
        self.data_scheme: List[str] = ["uuid", "path"]
        self.verification_scheme: List[str] = ["path"]
        self.total_time = 150
        self.save_path_folder = cfg["save_path_folder"]

    def apply_filter(self, filtering_df: pd.DataFrame, file_path: str) -> int:
        """
        Apply filtering to a specific parquet file.
        
        This method:
        1. Checks if there is enough time remaining
        2. Verifies if the target file exists
        3. Reads the parquet file
        4. Filters out entries based on UUIDs
        5. Saves the filtered parquet file
        
        Args:
            filtering_df (pd.DataFrame): DataFrame containing UUIDs to filter out
            file_path (str): Path to the parquet file to be filtered
            
        Returns:
            int: Count of remaining entries after filtering
            
        Raises:
            TimeoutError: If there's not enough time left to complete the operation
        """
        self.is_enough_time()

        if not os.path.exists(file_path):
            self.logger.info(f"Path doesn't exists: {file_path}")
            return 0

        server_name = re.findall(r"server=([^/]+)", file_path)[0]
        filename_path = os.path.basename(file_path)

        filtered_parquet = pd.read_parquet(
            file_path, filters=[("uuid", "not in", filtering_df["uuid"])]
        )

        self.is_enough_time()
        if len(filtered_parquet) == 0:
            self.logger.info(f"Fully filtered out: {server_name}/{filename_path}")

        filtered_parquet.to_parquet(
            file_path, index=False, compression="zstd", compression_level=3
        )

        return len(filtered_parquet)

    def runner_fn(self, df_local: pd.DataFrame) -> int:
        """
        Runner function that processes a chunk of data.
        
        This method:
        1. Extracts the file path from the input dataframe
        2. Calls apply_filter to process the file
        3. Logs the result and writes to verification file
        
        Args:
            df_local (pd.DataFrame): DataFrame containing paths and UUIDs to process
            
        Returns:
            int: 1 if successful, 0 if an error occurred
            
        Raises:
            NotImplementedError: If filter function wasn't implemented
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
