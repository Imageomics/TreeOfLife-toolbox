import glob
import os
from typing import List

import pandas as pd

from TreeOfLife_toolbox.main.config import Config
from TreeOfLife_toolbox.main.runners import MPIRunnerTool
from TreeOfLife_toolbox.main.schedulers import DefaultScheduler


class ScheduleCreation(DefaultScheduler):
    """
    A scheduler class that creates the data transfer and type change schedule.
    
    This class is responsible for reading the filter table files that contain information
    about which files need to be processed, and creating a schedule file that will be used
    by runner instances to process the data in parallel. The schedule assigns each file to
    a specific rank for balanced workload distribution.
    """
    def __init__(self, cfg: Config, seq_id: int):
        """
        Initialize the ScheduleCreation scheduler.
        
        Args:
            cfg (Config): Configuration object containing settings for the tool
            seq_id (int): Sequence ID of the batch to process
        """
        super().__init__(cfg)

        self.filter_name: str = "transfer_and_type_change"
        self.seq_id = seq_id

    def run(self):
        """
        Execute the scheduling process.
        
        This method:
        1. Reads all CSV files from the filter table directory for the current sequence ID
        2. Combines them into a single DataFrame
        3. Extracts the necessary columns (source, server, file_name)
        4. Removes duplicates
        5. Assigns a rank to each file for load balancing
        6. Saves the schedule to a CSV file
        
        Raises:
            ValueError: If filter_name is not set
        """
        assert self.filter_name is not None, ValueError("filter name is not set")

        filter_folder = os.path.join(
            self.tools_path, self.filter_name, str(self.seq_id).zfill(4)
        )
        filter_table_folder = os.path.join(filter_folder, "filter_table")

        all_files = glob.glob(os.path.join(filter_table_folder, "*.csv"))
        df: pd.DataFrame = pd.concat(
            (pd.read_csv(f) for f in all_files), ignore_index=True
        )
        df = df[["source", "server", "file_name"]]
        df = df.drop_duplicates(subset=["source", "server", "file_name"]).reset_index(
            drop=True
        )
        df["rank"] = df.index % self.total_workers

        df.to_csv(os.path.join(filter_folder, "schedule.csv"), header=True, index=False)


class Runner(MPIRunnerTool):
    """
    A runner class that executes the data transfer and type change operations.
    
    This class is responsible for processing files according to the schedule.
    For each file, it:
    1. Reads the parquet file from the source location
    2. Converts the 'source_id' column from its original type to string
    3. Saves the modified file to the destination location
    4. Removes the original file to free up storage space
    
    The class uses MPI for parallel processing across multiple nodes and cores.
    """
    def __init__(self, cfg: Config, seq_id: int):
        """
        Initialize the Runner.
        
        Args:
            cfg (Config): Configuration object containing settings for the tool
            seq_id (int): Sequence ID of the batch to process
        """
        super().__init__(cfg)

        self.filter_name: str = "transfer_and_type_change"
        self.data_scheme: List[str] = ["source", "server", "file_name"]
        self.verification_scheme: List[str] = ["source", "server", "file_name"]
        self.total_time = 150  # Maximum execution time in seconds
        self.seq_id = seq_id

        self.src_path = self.config["src_path"]
        self.dst_path = self.config["dst_path"]

    def ensure_folders_created(self):
        """
        Ensure that all necessary folders for processing exist.
        
        This method creates the verification folder and other required directories
        if they don't already exist.
        
        Raises:
            ValueError: If filter_name or verification_scheme is not set
        """
        assert self.filter_name is not None, ValueError("filter name is not set")
        assert self.verification_scheme is not None, ValueError(
            "verification scheme is not set"
        )

        self.filter_folder = os.path.join(
            self.tools_path, self.filter_name, str(self.seq_id).zfill(4)
        )
        self.filter_table_folder = os.path.join(self.filter_folder, "filter_table")
        self.verification_folder = os.path.join(self.filter_folder, "verification")

        os.makedirs(self.verification_folder, exist_ok=True)

    def apply_filter_different(
            self, filtering_df: pd.DataFrame, source: str, server: str, file_name: str
    ) -> int:
        """
        Apply type conversion to a specific file and transfer it to the destination.
        
        This method:
        1. Constructs source and destination paths
        2. Creates destination directories if needed
        3. Reads the source parquet file
        4. Converts the 'source_id' column to string type
        5. Writes the modified data to the destination
        6. Removes the source file to free up space
        
        Args:
            filtering_df (pd.DataFrame): DataFrame containing filter information
            source (str): Source name (used to construct the path)
            server (str): Server name (used to construct the path)
            file_name (str): File name (used to construct the path)
            
        Returns:
            int: Number of rows in the processed file, or 0 if the file doesn't exist
        """
        self.is_enough_time()

        src_path = os.path.join(
            self.src_path, f"source={source}", "data", f"server={server}", file_name
        )
        dst_path = os.path.join(
            self.dst_path, f"source={source}", f"server={server}", file_name
        )
        os.makedirs(
            os.path.join(
                self.dst_path,
                f"source={source}",
                f"server={server}",
            ),
            exist_ok=True,
        )

        if not os.path.exists(src_path):
            self.logger.info(f"Path doesn't exists: {src_path}")
            return 0

        renamed_parquet = pd.read_parquet(src_path)

        self.is_enough_time()

        renamed_parquet = renamed_parquet.astype({"source_id": "string"})

        renamed_parquet.to_parquet(
            dst_path, index=False, compression="zstd", compression_level=3
        )

        os.remove(src_path)

        return len(renamed_parquet)

    def runner_fn(self, df_local: pd.DataFrame) -> int:
        """
        Process a single file according to the schedule.
        
        This method extracts information about the file to process,
        calls the apply_filter_different method to perform the actual processing,
        and records the result in the verification file.
        
        Args:
            df_local (pd.DataFrame): DataFrame containing information about the file to process
            
        Returns:
            int: 1 if processing was successful, 0 if an error occurred
        """
        filtering_df = df_local.reset_index(drop=True)
        source = filtering_df.iloc[0]["source"]
        server = filtering_df.iloc[0]["server"]
        file_name = filtering_df.iloc[0]["file_name"]
        try:
            filtered_parquet_length = self.apply_filter_different(
                filtering_df, source, server, file_name
            )
        except NotImplementedError:
            raise NotImplementedError("Filter function wasn't implemented")
        except Exception as e:
            self.logger.exception(e)
            self.logger.error(f"Error occurred: {e}")
            return 0
        else:
            print(f"{source},{server},{file_name}", end="\n", file=self.verification_IO)
            self.logger.debug(
                f"Completed filtering: {source}/{server}/{file_name} with {filtered_parquet_length}"
            )
            return 1
