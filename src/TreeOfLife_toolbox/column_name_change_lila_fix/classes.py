import os
from typing import List

import pandas as pd

from TreeOfLife_toolbox.main.config import Config
from TreeOfLife_toolbox.main.filters import PythonFilterToolBase, FilterRegister
from TreeOfLife_toolbox.main.runners import MPIRunnerTool, RunnerRegister
from TreeOfLife_toolbox.main.schedulers import DefaultScheduler, SchedulerRegister


@FilterRegister("column_name_change_lila_fix")
class ColumnNameChangeLilaFixFilter(PythonFilterToolBase):
    """
    Filter component for the column_name_change_lila_fix tool.

    This filter extracts paths from a provided UUID table, specifically for files
    stored in 'storage.googleapis.com'. It creates a filtering table with these paths
    which will be used by subsequent components to process each file.

    The filter focuses on parquet files that need column name corrections, targeting
    only the specific Google Cloud Storage server.
    """

    def __init__(self, cfg: Config):
        """
        Initialize the filter with configuration.

        Args:
            cfg (Config): Configuration object containing tool settings.
        """
        super().__init__(cfg)

        self.filter_name: str = "column_name_change_lila_fix"

    def run(self):
        """
        Execute the filtering process.

        Reads the UUID table from the config-specified path, filters for entries from
        the 'storage.googleapis.com' server, and extracts just the file paths.
        The result is saved as a CSV file in the tool's filter_table directory.
        """
        # Load the UUID table and filter for Google Cloud Storage entries only
        uuid_table_df = pd.read_csv(self.config["uuid_table_path"], low_memory=False)
        uuid_table_df = uuid_table_df[
            uuid_table_df["server"] == "storage.googleapis.com"
        ][["path"]].drop_duplicates()

        # Save the filtered paths to CSV for the scheduler
        uuid_table_df.to_csv(
            os.path.join(
                self.tools_path, self.filter_name, "filter_table", "table.csv"
            ),
            index=False,
        )


@SchedulerRegister("column_name_change_lila_fix")
class ColumnNameChangeLilaFixScheduleCreation(DefaultScheduler):
    """
    Scheduler component for the column_name_change_lila_fix tool.

    This scheduler creates a task distribution schedule for the MPI workers
    based on the paths identified by the filter component. It inherits from
    DefaultScheduler which handles the distribution logic across available nodes.
    """

    def __init__(self, cfg: Config):
        """
        Initialize the scheduler with configuration.

        Args:
            cfg (Config): Configuration object containing tool settings.
        """
        super().__init__(cfg)

        self.filter_name: str = "column_name_change_lila_fix"
        self.scheme = [
            "path"
        ]  # Define the scheduling scheme to use the file path as the key


@RunnerRegister("column_name_change_lila_fix")
class ColumnNameChangeLilaFixRunner(MPIRunnerTool):
    """
    Runner component for the column_name_change_lila_fix tool.

    This tool is designed to fix incorrect column names in parquet files by renaming them
    according to a predefined mapping. It processes files identified by the filter and
    distributed by the scheduler using MPI parallelization.

    The default column mapping changes 'uuid_y' to 'uuid' and 'source_id_y' to 'source_id',
    correcting a specific issue in the Lila BC dataset.
    """

    def __init__(self, cfg: Config, name_mapping=None):
        """
        Initialize the runner with configuration and optional name mapping.

        Args:
            cfg (Config): Configuration object containing tool settings.
            name_mapping (dict, optional): Dictionary mapping old column names to new ones.
                Defaults to {'uuid_y': 'uuid', 'source_id_y': 'source_id'}.
        """
        super().__init__(cfg)

        if name_mapping is None:
            name_mapping = {"uuid_y": "uuid", "source_id_y": "source_id"}

        self.filter_name: str = "column_name_change_lila_fix"
        self.data_scheme: List[str] = ["path"]
        self.verification_scheme: List[str] = ["path"]
        self.total_time = 150  # Time limit in seconds
        self.save_path_folder = "/fs/scratch/PAS2136/gbif/processed/lilabc/name_fix/server=storage.googleapis.com"

        self.name_mapping = name_mapping

    def apply_filter(self, filtering_df: pd.DataFrame, file_path: str) -> int:
        """
        Apply the column name change fix to a single parquet file.

        This method loads a parquet file, renames columns according to the name mapping,
        and saves the result to a new location. It includes time checks to ensure processing
        doesn't exceed the configured time limit.

        Args:
            filtering_df (pd.DataFrame): DataFrame containing file metadata (not used in this implementation).
            file_path (str): Path to the parquet file that needs to be processed.

        Returns:
            int: Number of rows in the processed parquet file, or 0 if the file doesn't exist.

        Raises:
            TimeoutError: If processing would exceed the allocated time limit.
        """
        self.is_enough_time()

        if not os.path.exists(file_path):
            self.logger.info(f"Path doesn't exists: {file_path}")
            return 0

        renamed_parquet = pd.read_parquet(file_path)

        self.is_enough_time()

        renamed_parquet = renamed_parquet.rename(columns=self.name_mapping)
        save_path = os.path.join(self.save_path_folder, os.path.basename(file_path))
        os.makedirs(self.save_path_folder, exist_ok=True)

        renamed_parquet.to_parquet(
            save_path, index=False, compression="zstd", compression_level=3
        )

        return len(renamed_parquet)

    def runner_fn(self, df_local: pd.DataFrame) -> int:
        """
        Runner function that processes a batch of paths assigned to a worker.

        This method is called by the MPIRunnerTool parent class for each worker's assigned batch.
        It extracts the file path from the dataframe and applies the column name fix,
        then updates the verification log to track completed tasks.

        Args:
            df_local (pd.DataFrame): DataFrame containing paths assigned to this worker.

        Returns:
            int: 1 if processing was successful, 0 if an error occurred.
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
