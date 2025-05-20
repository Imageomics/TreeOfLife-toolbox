import os
from typing import List

import pandas as pd

from TreeOfLife_toolbox.main.config import Config
from TreeOfLife_toolbox.main.filters import FilterRegister, SparkFilterToolBase
from TreeOfLife_toolbox.main.runners import MPIRunnerTool, RunnerRegister
from TreeOfLife_toolbox.main.schedulers import DefaultScheduler, SchedulerRegister


@FilterRegister("mam_ansp_fix")
class MamAnspFixFilter(SparkFilterToolBase):
    """
    Filter class specifically designed to handle duplication issues with the mam.ansp.org server.
    
    This class loads a table of UUIDs containing duplicated entries from the mam.ansp.org server
    and prepares the filter table for processing. It extracts the relevant paths for files that
    need deduplication.
    
    Attributes:
        filter_name (str): The name of the filter, set to "mam_ansp_fix".
    """
    def __init__(self, cfg: Config):
        """
        Initialize the MamAnspFixFilter.
        
        Args:
            cfg (Config): Configuration object containing settings for the filter.
        """
        super().__init__(cfg)
        self.filter_name: str = "mam_ansp_fix"

    def run(self):
        """
        Execute the filtering process.
        
        Reads the UUID table from the path specified in the config, filters for records
        from the "mam.ansp.org" server, and extracts unique file paths. The resulting
        paths are saved to a CSV file for further processing.
        """
        uuid_table_df = pd.read_csv(self.config["uuid_table_path"], low_memory=False)
        uuid_table_df = uuid_table_df[uuid_table_df["server"] == "mam.ansp.org"][
            ["path"]
        ].drop_duplicates()

        uuid_table_df.to_csv(
            os.path.join(
                self.tools_path, self.filter_name, "filter_table", "table.csv"
            ),
            index=False,
        )


@SchedulerRegister("mam_ansp_fix")
class MamAnspFixScheduleCreation(DefaultScheduler):
    """
    Scheduler class for the mam.ansp.fix tool.
    
    Creates a schedule for processing the filtered paths, distributing the workload
    across available workers. Uses the DefaultScheduler functionality with a custom
    schema.
    
    Attributes:
        filter_name (str): The name of the filter, set to "mam_ansp_fix".
        scheme (List[str]): The column scheme for scheduling, set to ["path"].
    """
    def __init__(self, cfg: Config):
        """
        Initialize the MamAnspFixScheduleCreation.
        
        Args:
            cfg (Config): Configuration object containing settings for the scheduler.
        """
        super().__init__(cfg)
        self.filter_name: str = "mam_ansp_fix"
        self.scheme = ["path"]


@RunnerRegister("mam_ansp_fix")
class MamAnspFixRunner(MPIRunnerTool):
    """
    Runner class for the mam.ansp.fix tool.
    
    This class handles the actual deduplication process for the files from the mam.ansp.org
    server. It reads each parquet file, removes duplicate UUIDs, and saves the deduplicated 
    data to a new location.
    
    Attributes:
        filter_name (str): The name of the filter, set to "mam_ansp_fix".
        data_scheme (List[str]): The column scheme for the data, set to ["path"].
        verification_scheme (List[str]): The column scheme for verification, set to ["path"].
        total_time (int): Maximum processing time allowed, set to 150 seconds.
        save_path_folder (str): Path where deduplicated files will be saved.
    """
    def __init__(self, cfg: Config):
        """
        Initialize the MamAnspFixRunner.
        
        Args:
            cfg (Config): Configuration object containing settings for the runner,
                          including the save path for processed files.
        """
        super().__init__(cfg)
        self.filter_name: str = "mam_ansp_fix"
        self.data_scheme: List[str] = ["path"]
        self.verification_scheme: List[str] = ["path"]
        self.total_time = 150
        self.save_path_folder = cfg["save_path_folder"]

    def apply_filter(self, filtering_df: pd.DataFrame, file_path: str) -> int:
        """
        Apply the deduplication filter to a specific file.
        
        This method reads a parquet file, removes duplicated entries based on UUID,
        and saves the deduplicated data to the designated save path.
        
        Args:
            filtering_df (pd.DataFrame): DataFrame containing filter information.
            file_path (str): Path to the parquet file that needs deduplication.
            
        Returns:
            int: The number of records in the deduplicated result.
            
        Raises:
            Various exceptions can be raised if file operations fail.
        """
        self.is_enough_time()

        if not os.path.exists(file_path):
            self.logger.info(f"Path doesn't exists: {file_path}")
            return 0

        filtered_parquet = pd.read_parquet(file_path)

        self.is_enough_time()

        if len(filtered_parquet) == 0:
            self.logger.info(f"Fully filtered out: {file_path}")

        filtered_parquet = filtered_parquet.drop_duplicates("uuid")
        save_path = os.path.join(self.save_path_folder, os.path.basename(file_path))
        os.makedirs(self.save_path_folder, exist_ok=True)

        filtered_parquet.to_parquet(
            save_path, index=False, compression="zstd", compression_level=3
        )

        return len(filtered_parquet)

    def runner_fn(self, df_local: pd.DataFrame) -> int:
        """
        Process a batch of files from the schedule.
        
        This method handles a batch of paths (typically one path at a time),
        applies the deduplication filter, and records the completion status.
        
        Args:
            df_local (pd.DataFrame): DataFrame containing the file paths to process.
            
        Returns:
            int: 1 if successful, 0 if an error occurred.
            
        Raises:
            NotImplementedError: If the filter function is not implemented.
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
