import glob
import hashlib
import os
import shutil
import uuid
from typing import List, Tuple, Sequence

import numpy as np
import pandas as pd

from TreeOfLife_toolbox.main.config import Config
from TreeOfLife_toolbox.main.filters import PythonFilterToolBase, FilterRegister
from TreeOfLife_toolbox.main.runners import MPIRunnerTool, RunnerRegister
from TreeOfLife_toolbox.main.schedulers import DefaultScheduler, SchedulerRegister
from TreeOfLife_toolbox.main.utils import ensure_created


@FilterRegister("data_transfer")
class DataTransferFilter(PythonFilterToolBase):
    """
    A filter class for data transfer operations.

    This class is responsible for identifying parquet files that need to be transferred
    from a source location to a destination location. It handles both successful downloads
    ('successes.parquet') and error files ('errors.parquet').

    The class generates unique destination paths for each file based on the server name and
    a randomly generated UUID to ensure uniqueness.
    """

    def __init__(self, cfg: Config):
        """
        Initialize the DataTransferFilter.

        Args:
            cfg (Config): Configuration object containing settings for the data transfer process.
        """
        super().__init__(cfg)

        self.filter_name: str = "data_transfer"

    def get_all_paths_to_merge(self) -> pd.DataFrame:
        """
        Identify all parquet files that need to be transferred and generate their destination paths.

        This method:
        1. Finds all completed partitions in the source directory
        2. Extracts server names and file types (successes or errors)
        3. Generates unique destination paths for each file
        4. Ensures destination directories exist

        Returns:
            pd.DataFrame: DataFrame containing source and destination paths for each file to be transferred.
                          Columns: ['src_path', 'dst_path']
        """
        glob_wildcard = self.downloaded_images_path + "/*/*"
        server_name_regex = (
            rf"{self.downloaded_images_path}/server_name=(.*)/partition_id=.*"
        )
        basename_regex = (
            rf"{self.downloaded_images_path}/server_name=.*/partition_id=.*/(.*)"
        )
        os.makedirs(self.config["dst_image_folder"], exist_ok=True)
        os.makedirs(self.config["dst_error_folder"], exist_ok=True)

        src_paths = []

        for path in glob.glob(glob_wildcard):
            if not os.path.exists(os.path.join(path, "completed")):
                continue
            src_paths.append(os.path.join(path, "successes.parquet"))
            src_paths.append(os.path.join(path, "errors.parquet"))

        src_paths_df = pd.DataFrame(src_paths, columns=["src_path"])
        src_paths_df["server_name"] = src_paths_df["src_path"].str.extract(
            server_name_regex
        )
        src_paths_df["corrected_server_name"] = src_paths_df["server_name"].str.replace(
            "%3A", "_"
        )
        src_paths_df["basename"] = src_paths_df["src_path"].str.extract(basename_regex)
        src_paths_df["dst_base_folder"] = np.where(
            src_paths_df["basename"] == "successes.parquet",
            self.config["dst_image_folder"],
            self.config["dst_error_folder"],
        )
        src_paths_df["dst_basename"] = np.where(
            src_paths_df["basename"] == "successes.parquet", "data_", "errors_"
        )
        src_paths_df["uuid"] = src_paths_df.apply(lambda _: str(uuid.uuid4()), axis=1)
        src_paths_df["dst_path"] = (
            src_paths_df["dst_base_folder"]
            + "/server="
            + src_paths_df["corrected_server_name"]
            + "/"
            + src_paths_df["dst_basename"]
            + src_paths_df["uuid"]
            + ".parquet"
        )
        return src_paths_df[["src_path", "dst_path"]]


@SchedulerRegister("data_transfer")
class DataTransferScheduleCreation(DefaultScheduler):
    """
    Scheduler class for data transfer operations.

    This class is responsible for creating a schedule for data transfer operations
    by combining all filter tables into a single schedule file. The schedule file
    contains information about which files need to be transferred and where they
    should be transferred to.
    """
    def __init__(self, cfg: Config):
        """
        Initialize the DataTransferScheduleCreation.

        Args:
            cfg (Config): Configuration object containing settings for the scheduler.
        """
        super().__init__(cfg)

        self.filter_name: str = "data_transfer"

    def run(self):
        """
        Execute the scheduler to create a combined schedule file.

        This method:
        1. Verifies that the filter name is set
        2. Reads all CSV files from the filter table directory
        3. Combines them into a single DataFrame
        4. Writes the combined DataFrame to a schedule file

        Raises:
            ValueError: If filter_name is not set.
        """
        assert self.filter_name is not None, ValueError("filter name is not set")

        filter_folder = os.path.join(self.tools_path, self.filter_name)
        filter_table_folder = os.path.join(filter_folder, "filter_table")

        all_files = glob.glob(os.path.join(filter_table_folder, "*.csv"))
        df: pd.DataFrame = pd.concat(
            (pd.read_csv(f) for f in all_files), ignore_index=True
        )

        df.to_csv(os.path.join(filter_folder, "schedule.csv"), header=True, index=False)


@RunnerRegister("data_transfer")
class DataTransferRunner(MPIRunnerTool):
    """
    Runner class for executing data transfer operations.

    This class is responsible for the actual file transfer process. It uses MPI for 
    parallel execution to efficiently copy files from source to destination locations.
    The class also verifies the integrity of copied files by computing and comparing 
    MD5 hashsums of source and destination files.
    """
    def __init__(self, cfg: Config):
        """
        Initialize the DataTransferRunner.

        Args:
            cfg (Config): Configuration object containing settings for the runner.
        """
        super().__init__(cfg)

        self.mpi_comm = None
        self.mpi_rank = 0

        self.filter_name: str = "data_transfer"
        self.verification_scheme = [
            "src_path",
            "dst_path",
            "hashsum_src",
            "hashsum_dst",
        ]
        self.total_time = 60  # Maximum execution time in seconds

    def get_schedule(self) -> pd.DataFrame:
        """
        Load the schedule file created by the scheduler.

        Returns:
            pd.DataFrame: DataFrame containing the schedule information with source and destination paths.
        """
        return pd.read_csv(os.path.join(self.filter_folder, "schedule.csv"))

    def get_remaining_table(self, schedule: pd.DataFrame) -> pd.DataFrame:
        """
        Determine which files still need to be copied by comparing the schedule with already verified transfers.

        This method:
        1. Loads the verification table containing information about already copied files
        2. Performs an outer join between the schedule and verification table
        3. Filters for files that are only in the schedule (not yet copied)

        Args:
            schedule (pd.DataFrame): DataFrame containing the full schedule of files to copy.

        Returns:
            pd.DataFrame: DataFrame containing only the files that still need to be copied.
                          Columns: ['src_path', 'dst_path']
        """
        verification_df = self.load_table(
            self.verification_folder, self.verification_scheme
        )
        outer_join = schedule.merge(
            verification_df, how="outer", indicator=True, on=["src_path", "dst_path"]
        )
        left_to_copy = outer_join[(outer_join["_merge"] == "left_only")].drop(
            "_merge", axis=1
        )[["src_path", "dst_path"]]
        return left_to_copy

    @staticmethod
    def correct_server_name(server_names: List[str]) -> List[str]:
        """
        Replace special characters in server names with underscores.

        This method specifically replaces '%3A' (URL-encoded colon) with underscores
        to ensure valid directory names.

        Args:
            server_names (List[str]): List of server names to be corrected.

        Returns:
            List[str]: List of corrected server names.
        """
        for i, server in enumerate(server_names):
            server_names[i] = server.replace("%3A", "_")

        return server_names

    def ensure_all_servers_exists(
        self, all_files_df: pd.DataFrame, dst_paths: Sequence[str]
    ) -> None:
        """
        Ensure that destination directories for all servers exist.

        This method:
        1. Extracts unique server names from the source paths
        2. Corrects server names by replacing special characters
        3. Creates destination directories for each server in each destination path

        Args:
            all_files_df (pd.DataFrame): DataFrame containing source paths with server information.
            dst_paths (Sequence[str]): List of destination base paths where server directories should be created.

        Returns:
            None
        """
        server_name_regex = (
            rf"{self.downloaded_images_path}/server_name=(.*)/partition_id=.*"
        )

        server_names_series: pd.Series = all_files_df["src_path"].str.extract(
            server_name_regex, expand=False
        )
        server_names = (
            server_names_series.drop_duplicates().reset_index(drop=True).to_list()
        )
        server_names = self.correct_server_name(server_names)
        for dst_path in dst_paths:
            ensure_created(
                [os.path.join(dst_path, f"server={server}") for server in server_names]
            )

    @staticmethod
    def compute_hashsum(file_path: str, hashsum_alg) -> str:
        """
        Compute a hashsum for a file using the provided hash algorithm.

        This method reads the file in chunks to efficiently handle large files.

        Args:
            file_path (str): Path to the file for which to compute the hashsum.
            hashsum_alg: Hash algorithm object (e.g., from hashlib) to use for computing the hashsum.

        Returns:
            str: Hexadecimal digest of the computed hashsum.
        """
        with open(file_path, "rb") as f:
            while True:
                data = f.read(131_072)  # Read in 128KB chunks
                if not data:
                    break
                hashsum_alg.update(data)
        return hashsum_alg.hexdigest()

    def copy_file(
        self, row: Tuple[pd.Index, pd.Series]
    ) -> Tuple[bool, str, str, str, str]:
        """
        Copy a file from source to destination and verify integrity with hashsums.

        This method:
        1. Computes the MD5 hashsum of the source file
        2. Copies the file to the destination
        3. Computes the MD5 hashsum of the destination file
        4. Periodically checks if there's enough time left to continue execution

        Args:
            row (Tuple[pd.Index, pd.Series]): A tuple containing the index and a Series with 
                                              'src_path' and 'dst_path' columns.

        Returns:
            Tuple[bool, str, str, str, str]: A tuple containing:
                - bool: True if an error occurred, False otherwise
                - str: Source path
                - str: Destination path or error message if an error occurred
                - str: Source file hashsum (empty if error)
                - str: Destination file hashsum (empty if error)
        """
        src_path = row[1]["src_path"]
        dst_path = row[1]["dst_path"]
        try:
            self.is_enough_time()

            hs_src_alg = hashlib.md5()
            hs_src_local = self.compute_hashsum(src_path, hs_src_alg)

            self.is_enough_time()

            shutil.copy(src_path, dst_path)

            self.is_enough_time()

            hs_dest_alg = hashlib.md5()
            hs_dest_local = self.compute_hashsum(dst_path, hs_dest_alg)
            return False, src_path, dst_path, hs_src_local, hs_dest_local
        except Exception as e:
            return True, src_path, str(e), "", ""

    def run(self):
        """
        Execute the data transfer process.

        This method:
        1. Loads the schedule of files to be copied
        2. Determines which files still need to be copied
        3. Ensures all necessary destination directories exist
        4. Uses MPI to parallelize the file copying process
        5. Records verification information for successfully copied files
        6. Logs errors for failed copy operations

        The method uses MPI for parallel execution to efficiently process multiple files
        simultaneously across available compute resources.

        Returns:
            None
        """
        from mpi4py.futures import MPIPoolExecutor

        self.ensure_folders_created()

        schedule = self.get_schedule()
        if len(schedule) == 0:
            self.logger.error(f"Schedule not found or empty for rank {self.mpi_rank}")
            exit(0)

        remaining_table = self.get_remaining_table(schedule)
        self.ensure_all_servers_exists(
            remaining_table,
            [self.config["dst_image_folder"], self.config["dst_error_folder"]],
        )

        self.logger.info("Started copying")
        self.logger.info(f"{len(remaining_table)} files left to copy")
        with self.get_csv_writer(
            f"{self.verification_folder}/verification.csv", self.verification_scheme
        ) as verification_file:
            with MPIPoolExecutor() as executor:
                for is_error, src, dst, hs_src, hs_dest in executor.map(
                    self.copy_file, remaining_table.iterrows()
                ):
                    if is_error:
                        self.logger.error(f"Error {dst} for {src}")
                    else:
                        print(
                            src,
                            dst,
                            hs_src,
                            hs_dest,
                            sep=",",
                            file=verification_file,
                            flush=True,
                        )
                        self.logger.debug(f"Copied file {src} to {dst}")

        self.logger.info("Finished copying")
