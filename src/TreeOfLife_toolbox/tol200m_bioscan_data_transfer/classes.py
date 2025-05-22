import glob
import hashlib
import math
import os
import uuid
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import pyspark.sql.functions as func
from pyspark.sql import Window

from TreeOfLife_toolbox.main.config import Config
from TreeOfLife_toolbox.main.filters import SparkFilterToolBase, FilterRegister
from TreeOfLife_toolbox.main.runners import (
    RunnerRegister,
    MPIRunnerTool,
)
from TreeOfLife_toolbox.main.schedulers import SchedulerRegister, SchedulerToolBase
from TreeOfLife_toolbox.main.utils import load_dataframe


@FilterRegister("tol200m_bioscan_data_transfer")
class ToLBioscanDataTransferFilter(SparkFilterToolBase):
    """
    A filter class that processes BIOSCAN data and prepares it for transfer to the Tree of Life format.
    
    This class processes input data from BIOSCAN, joins it with provenance data, and 
    partitions it into batches for further processing.
    """
    def __init__(self, cfg: Config):
        """
        Initialize the BIOSCAN data transfer filter.
        
        Args:
            cfg (Config): Configuration object containing necessary parameters
        """
        super().__init__(cfg)
        self.filter_name: str = "tol200m_bioscan_data_transfer"
        self.batch_size: int = self.config.get("downloader_parameters", {}).get(
            "batch_size", 10_000
        )

        self.input_path = self.config["path_to_input"]
        self.output_path = self.config.get_folder("urls_folder")
        self.provenance_path = self.config["provenance_path"]

    def run(self):
        """
        Execute the data processing pipeline.
        
        Loads data from the input path, joins it with provenance data, and partitions
        it into batches based on the configured batch size. The resulting batches are
        written as CSV files to the output path.
        
        Returns:
            None
        """
        multimedia_df = load_dataframe(self.spark, self.input_path)
        provenance_df = self.spark.read.parquet(self.provenance_path)

        multimedia_df = multimedia_df.join(
            provenance_df,
            multimedia_df["processid"] == provenance_df["source_id"],
            "inner",
        )
        columns = multimedia_df.columns

        self.logger.info("Starting batching")

        batch_count = math.floor(multimedia_df.count() / self.batch_size)

        window_part = Window.orderBy("processid")
        master_df_filtered = (
            multimedia_df.withColumn("row_number", func.row_number().over(window_part))
            .withColumn("partition_id", func.col("row_number") % batch_count)
            .withColumn(
                "partition_id",
                (
                    func.when(func.col("partition_id").isNull(), 0).otherwise(
                        func.col("partition_id")
                    )
                ),
            )
            .select(*columns, "partition_id")
        )

        self.logger.info("Writing to parquet")

        (
            master_df_filtered.write.partitionBy("partition_id").csv(
                self.output_path, header=True, mode="overwrite"
            )
        )

        self.logger.info("Finished batching")


@SchedulerRegister("tol200m_bioscan_data_transfer")
class ToLBioscanDataTransferScheduleCreation(SchedulerToolBase):
    """
    A scheduler class that creates a schedule for transferring BIOSCAN data to TOL format.
    
    This class generates a schedule file that maps source partition paths to destination
    paths in the Tree of Life data structure, assigning UUIDs to each partition.
    """
    def __init__(self, cfg: Config):
        """
        Initialize the scheduler for BIOSCAN data transfer.
        
        Args:
            cfg (Config): Configuration object containing necessary parameters
        """
        super().__init__(cfg)
        self.filter_name: str = "tol200m_bioscan_data_transfer"

    def run(self):
        """
        Create a schedule for data transfer.
        
        Identifies partitions in the processed data folder and creates a mapping between
        source paths and destination paths in the Tree of Life data structure.
        The resulting schedule is saved as a CSV file.
        
        Returns:
            None
        """
        processed_path = self.config.get_folder("urls_folder")
        partitions = [
            f"{processed_path}/{folder}"
            for folder in os.listdir(processed_path)
            if os.path.isdir(f"{processed_path}/{folder}")
        ]
        name_table_df = pd.DataFrame(partitions, columns=["src_path"])
        name_table_df["uuid"] = name_table_df.apply(lambda _: str(uuid.uuid4()), axis=1)
        name_table_df["dst_path"] = (
            f"{self.config['path_to_tol_folder']}/source=bioscan/server=bioscan"
            + "/data_"
            + name_table_df["uuid"]
            + ".parquet"
        )
        name_table_df = name_table_df[["src_path", "dst_path"]]
        name_table_df.to_csv(
            os.path.join(
                os.path.join(self.tools_path, self.filter_name), "schedule.csv"
            ),
            header=True,
            index=False,
        )


@RunnerRegister("tol200m_bioscan_data_transfer")
class ToLBioscanDataTransferRunner(MPIRunnerTool):
    """
    A runner class that executes the transfer of BIOSCAN data to the Tree of Life format.
    
    This class processes the partitioned data according to the schedule, reads and resizes images,
    calculates hashsums, and saves the data in the Tree of Life format at the destination paths.
    It uses MPI for parallel processing.
    """
    target_columns = [
        "uuid",
        "source_id",
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

    def __init__(self, cfg: Config):
        """
        Initialize the BIOSCAN data transfer runner.
        
        Args:
            cfg (Config): Configuration object containing necessary parameters
        """
        super().__init__(cfg)

        self.mpi_comm = None
        self.mpi_rank = 0

        self.filter_name: str = "tol200m_bioscan_data_transfer"
        self.verification_scheme = [
            "src_path",
            "dst_path",
        ]
        self.total_time = 60  # Maximum execution time in seconds

        self.max_size = self.config.get("downloader_parameters", {}).get(
            "image_size", 720
        )
        self.image_folder = self.config["bioscan_image_folder"]

    def get_schedule(self) -> pd.DataFrame:
        """
        Load the transfer schedule from a CSV file.
        
        Returns:
            pd.DataFrame: DataFrame containing source and destination paths
        """
        return pd.read_csv(os.path.join(self.filter_folder, "schedule.csv"))

    def get_remaining_table(self, schedule: pd.DataFrame) -> pd.DataFrame:
        """
        Determine which files from the schedule still need to be processed.
        
        Args:
            schedule (pd.DataFrame): DataFrame containing the full schedule
            
        Returns:
            pd.DataFrame: DataFrame containing only the files that still need to be processed
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
    def image_resize(
        image: np.ndarray, _max_size=720
    ) -> Tuple[np.ndarray[int, np.dtype[np.uint8]], Tuple[int, int]]:
        """
        Resize an image while preserving its aspect ratio.
        
        Args:
            image (np.ndarray): Input image to resize
            _max_size (int, optional): Maximum dimension size. Defaults to 720.
            
        Returns:
            Tuple[np.ndarray, Tuple[int, int]]: Resized image and its new dimensions (height, width)
        """
        h, w = image.shape[:2]
        if h > w:
            new_h = _max_size
            new_w = int(w * (new_h / h))
        else:
            new_w = _max_size
            new_h = int(h * (new_w / w))
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA), (
            new_h,
            new_w,
        )

    def read_images(self, row: pd.Series) -> pd.Series:
        """
        Read and process an image for a given data row.
        
        Loads the image corresponding to the source_id, calculates original hashsum,
        resizes if necessary, and calculates resized hashsum.
        
        Args:
            row (pd.Series): DataFrame row containing source_id and other metadata
            
        Returns:
            pd.Series: Series containing image data and metadata
        """
        image_path = f"{self.image_folder}/{row['split']}/"
        if row["chunk"] not in [np.nan, None]:
            image_path += f"{row['chunk']}/"
        image_path += f"{row['source_id']}.jpg"
        image = cv2.imread(image_path)

        original_hashsum = hashlib.md5(image.tobytes()).hexdigest()

        original_shape = image.shape[:2]
        resized_size = image.shape[:2]
        if original_shape[0] > self.max_size or original_shape[1] > self.max_size:
            image, resized_size = self.image_resize(image, self.max_size)

        resized_hashsum = hashlib.md5(image.tobytes()).hexdigest()

        return pd.Series(
            {
                "uuid": row["uuid"],
                "hashsum_original": original_hashsum,
                "hashsum_resized": resized_hashsum,
                "original_size": original_shape,
                "resized_size": resized_size,
                "image": image.tobytes(),
            },
            index=[
                "uuid",
                "hashsum_original",
                "hashsum_resized",
                "original_size",
                "resized_size",
                "image",
            ],
        )

    def copy_file(self, row: Tuple[pd.Index, pd.Series]) -> Tuple[bool, str, str]:
        """
        Process and copy a file from source to destination.
        
        Reads data from the source path, processes images, and saves in the Tree of Life format
        at the destination path.
        
        Args:
            row (Tuple[pd.Index, pd.Series]): Row from the schedule with source and destination paths
            
        Returns:
            Tuple[bool, str, str]: Error flag, source path, and either destination path or error message
        """
        src_path = row[1]["src_path"]
        dst_path = row[1]["dst_path"]
        try:
            self.is_enough_time()

            all_files = glob.glob(os.path.join(src_path, "*.csv"))
            df: pd.DataFrame = pd.concat(
                (pd.read_csv(f) for f in all_files), ignore_index=True
            )

            df.rename(
                {"processid": "source_id"}, inplace=True, errors="raise", axis="columns"
            )
            df.loc[:, ["identifier", "source", "title"]] = np.nan
            df["is_license_full"] = False
            df["license"] = "https://creativecommons.org/licenses/by/3.0/"
            self.logger.debug(f"Got data for {src_path}")

            df_image = df.apply(self.read_images, axis=1)
            self.logger.debug(f"Got images for {src_path}")

            df = df.merge(
                df_image, on="uuid", how="inner", validate="1:1", suffixes=("_x", "")
            )
            df = df[self.target_columns]
            df.to_parquet(
                dst_path, index=False, compression="zstd", compression_level=3
            )

            return False, src_path, dst_path
        except Exception as e:
            return True, src_path, str(e)

    def run(self):
        """
        Execute the data transfer process using MPI for parallel processing.
        
        Loads the schedule, determines which files still need processing,
        and uses MPI to distribute the workload across available processes.
        Progress is tracked in a verification file.
        
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

        with self.get_csv_writer(
            f"{self.verification_folder}/verification.csv", self.verification_scheme
        ) as verification_file:
            with MPIPoolExecutor() as executor:
                for is_error, src, dst in executor.map(
                    self.copy_file, remaining_table.iterrows()
                ):
                    if is_error:
                        self.logger.error(f"Error {dst} for {src}")
                    else:
                        print(
                            src,
                            dst,
                            sep=",",
                            file=verification_file,
                            flush=True,
                        )
                        self.logger.debug(f"Transferred file {src} to {dst}")

        self.logger.info("Finished copying")
