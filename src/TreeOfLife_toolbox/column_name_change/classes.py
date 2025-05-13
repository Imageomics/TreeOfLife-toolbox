import os
from typing import List

import pandas as pd

from TreeOfLife_toolbox.main.config import Config
from TreeOfLife_toolbox.main.filters import PythonFilterToolBase, FilterRegister
from TreeOfLife_toolbox.main.runners import MPIRunnerTool, RunnerRegister
from TreeOfLife_toolbox.main.schedulers import DefaultScheduler, SchedulerRegister


@FilterRegister("column_name_change")
class ColumnNameChangeFilter(PythonFilterToolBase):
    """
    Filter class for the Column Name Change tool.

    This class implements the filtering stage of the column name change process.
    It identifies all partitions in the downloaded image directory structure
    that need to have their column names modified.

    Inherits from PythonFilterToolBase which provides the implementation
    for traversing the directory structure and identifying all server_name/partition_id
    combinations with parquet files.
    """

    def __init__(self, cfg: Config):
        """
        Initialize the Column Name Change Filter.

        Args:
            cfg (Config): Configuration object containing parameters for the tool.
        """
        super().__init__(cfg)

        self.filter_name: str = "column_name_change"


@SchedulerRegister("column_name_change")
class ColumnNameChangeScheduleCreation(DefaultScheduler):
    """
    Scheduler class for the Column Name Change tool.

    This class creates a work schedule for the column name change process,
    distributing the partitions across available workers. It uses the
    DefaultScheduler implementation which assigns partitions to workers
    based on a round-robin assignment for load balancing.
    """

    def __init__(self, cfg: Config):
        """
        Initialize the Column Name Change Scheduler.

        Args:
            cfg (Config): Configuration object containing parameters for the tool.
        """
        super().__init__(cfg)

        self.filter_name: str = "column_name_change"


@RunnerRegister("column_name_change")
class ColumnNameChangeRunner(MPIRunnerTool):
    """
    Runner class for the Column Name Change tool.

    This class performs the actual column name change operation on the parquet files.
    It processes each server_name/partition_id combination assigned to a worker
    by loading the parquet file, renaming columns according to the mapping specified
    in the configuration, and saving the modified file back to disk.

    The MPI approach enables parallel processing across multiple nodes and cores,
    with each worker handling its assigned partitions.
    """

    def __init__(self, cfg: Config):
        """
        Initialize the Column Name Change Runner.

        Args:
            cfg (Config): Configuration object containing parameters for the tool,
                         including the name mapping dictionary.
        """
        super().__init__(cfg)

        self.filter_name: str = "column_name_change"
        self.data_scheme: List[str] = ["server_name", "partition_id"]
        self.verification_scheme: List[str] = ["server_name", "partition_id"]
        self.total_time = 150

        # Load the column name mapping from configuration
        self.name_mapping = cfg["name_mapping"]

    def apply_filter(
        self, filtering_df: pd.DataFrame, server_name: str, partition_id: int
    ) -> int:
        """
        Apply the column name change operation to a specific partition.

        This method loads the parquet file for the specified server_name and partition_id,
        renames the columns according to the mapping provided in the configuration,
        and saves the modified file back to the same location.

        Args:
            filtering_df (pd.DataFrame): DataFrame containing information about partitions to process.
            server_name (str): Name of the server containing the partition.
            partition_id (int): ID of the partition to process.

        Returns:
            int: Number of rows in the processed parquet file, or 0 if the file doesn't exist.

        Note:
            The method also performs time checks to ensure there is enough time left
            in the job to complete the operation.
        """
        self.is_enough_time()

        parquet_path = os.path.join(
            self.downloaded_images_path,
            f"server_name={server_name}",
            f"partition_id={partition_id}",
            "successes.parquet",
        )

        if not os.path.exists(parquet_path):
            self.logger.info(f"Path doesn't exists: {parquet_path}")
            return 0

        renamed_parquet = pd.read_parquet(parquet_path)

        self.is_enough_time()

        renamed_parquet = renamed_parquet.rename(columns=self.name_mapping)

        renamed_parquet.to_parquet(
            parquet_path, index=False, compression="zstd", compression_level=3
        )

        return len(renamed_parquet)
