"""
Encyclopedia of Life (EoL) dataset renaming module.

This module provides components for renaming images in the EoL dataset by merging
source identifiers from the original batch data. It includes filter, scheduler,
and runner classes for the EoL rename operation within the TreeOfLife toolbox.
"""

import os
from typing import List

import pandas as pd

from TreeOfLife_toolbox.main.config import Config
from TreeOfLife_toolbox.main.filters import PythonFilterToolBase, FilterRegister
from TreeOfLife_toolbox.main.runners import MPIRunnerTool, RunnerRegister
from TreeOfLife_toolbox.main.schedulers import DefaultScheduler, SchedulerRegister


@FilterRegister("eol_rename")
class EoLRenameFilter(PythonFilterToolBase):
    """
    Filter class for EoL rename operations.
    
    This class registers the 'eol_rename' filter in the filtering system.
    It doesn't override any methods as it uses the default behavior from
    the PythonFilterToolBase class.
    
    Attributes:
        filter_name: Name of the filter used for registration and identification.
    """
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.filter_name: str = "eol_rename"


@SchedulerRegister("eol_rename")
class EoLRenameScheduleCreation(DefaultScheduler):
    """
    Scheduler class for EoL rename operations.
    
    This scheduler is responsible for creating the execution schedule for 
    the EoL rename operation. It uses the default scheduling behavior from
    the DefaultScheduler class.
    
    Attributes:
        filter_name: Name of the filter used for registration and identification.
    """
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.filter_name: str = "eol_rename"


@RunnerRegister("eol_rename")
class EoLRenameRunner(MPIRunnerTool):
    """
    Runner class for executing the EoL rename operations.
    
    This class handles the actual processing of the EoL dataset images,
    adding source identifiers by merging information from batch data
    with downloaded image data.
    
    Attributes:
        filter_name: Name of the filter used for registration and identification.
        data_scheme: List of fields used to partition the dataset.
        verification_scheme: List of fields used for verification.
        total_time: Maximum allowed execution time in seconds.
    """
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.filter_name: str = "eol_rename"
        self.data_scheme: List[str] = ["server_name", "partition_id"]
        self.verification_scheme: List[str] = ["server_name", "partition_id"]
        self.total_time = 150

    def apply_filter(
            self, filtering_df: pd.DataFrame, server_name: str, partition_id: int
    ) -> int:
        """
        Apply the EoL rename filter to a specific partition of data.
        
        This method adds source identifiers to the downloaded images data by 
        merging 'EOL content ID' and 'EOL page ID' from the original batch data.
        It concatenates these IDs to create a 'source_id' field and saves the 
        updated data back to the original successes.parquet file.
        
        Args:
            filtering_df: DataFrame containing the filter data.
            server_name: Name of the server containing the data.
            partition_id: Partition ID within the server.
            
        Returns:
            int: Number of records processed.
            
        Notes:
            - Checks for time constraints during operation.
            - Skips processing if the parquet path doesn't exist.
        """
        self.is_enough_time()

        parquet_path = os.path.join(
            self.downloaded_images_path,
            f"server_name={server_name}",
            f"partition_id={partition_id}",
            "successes.parquet",
        )
        server_batch_path = os.path.join(
            self.config.get_folder("urls_folder"),
            f"server_name={server_name}",
            f"partition_id={partition_id}",
        )

        if not os.path.exists(parquet_path):
            self.logger.info(f"Path doesn't exists: {parquet_path}")
            return 0

        parquet = pd.read_parquet(parquet_path)
        server_batch = pd.read_parquet(
            server_batch_path, columns=["EOL content ID", "EOL page ID", "uuid"]
        )

        self.is_enough_time()

        parquet = parquet.merge(server_batch, on="uuid", how="left", validate="1:1")
        parquet["source_id"] = parquet["EOL content ID"] + "_" + parquet["EOL page ID"]

        parquet.to_parquet(
            parquet_path, index=False, compression="zstd", compression_level=3
        )

        return len(parquet)
