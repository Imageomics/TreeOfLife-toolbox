import hashlib
import os
import uuid
from typing import List

import numpy as np
import pandas as pd
import pyspark.sql.functions as func

from TreeOfLife_toolbox.main.config import Config
from TreeOfLife_toolbox.main.filters import FilterRegister, SparkFilterToolBase
from TreeOfLife_toolbox.main.runners import MPIRunnerTool, RunnerRegister
from TreeOfLife_toolbox.main.schedulers import DefaultScheduler, SchedulerRegister
from TreeOfLife_toolbox.main.utils import load_dataframe

server_pattern = r"server=([^/]+)"


@FilterRegister("fathom_net_crop_fix")
class FathomnetCropFixFilter(SparkFilterToolBase):
    """
    Filter to prepare data for the FathomNet crop fix operation.

    This class loads UUID tables with incorrectly cropped images and lookup tables, then joins them to create
    a mapping between UUIDs and their corresponding file paths. The resulting
    data is partitioned by server for distributed processing.
    """

    def __init__(self, cfg: Config):
        """
        Initialize the FathomNet crop fix filter with configuration.

        Args:
            cfg (Config): Configuration object with paths and settings.
        """
        super().__init__(cfg)

        self.filter_name: str = "fathom_net_crop_fix"

    def run(self):
        """
        Run the filter to create a table of images that need to be reprocessed.

        Loads the UUID table with incorrectly cropped images and lookup table, joins them to match UUIDs with
        file paths, extracts server information from paths, and saves the result
        as a CSV in the filter table directory.
        """
        uuid_table_df = load_dataframe(
            self.spark, self.config["uuid_table_path"]
        ).repartition(20)
        lookup_table_df = load_dataframe(
            self.spark, self.config["look_up_table_path"]
        ).repartition(20)

        merged_df = uuid_table_df.join(
            lookup_table_df, on="uuid", how="left"
        ).withColumn("server", func.regexp_extract("path", server_pattern, 1))

        (
            merged_df.repartition(1).write.csv(
                os.path.join(self.tools_path, self.filter_name, "filter_table"),
                header=True,
                mode="overwrite",
            )
        )


@SchedulerRegister("fathom_net_crop_fix")
class FathomnetCropFixScheduleCreation(DefaultScheduler):
    """
    Scheduler for FathomNet crop fix operations.

    Creates a schedule for distributed processing by server, ensuring
    all images from the same server are processed together.
    """

    def __init__(self, cfg: Config):
        """
        Initialize the scheduler with configuration.

        Args:
            cfg (Config): Configuration object with settings.
        """
        super().__init__(cfg)

        self.filter_name: str = "fathom_net_crop_fix"
        self.scheme = ["server"]  # Group tasks by server


@RunnerRegister("fathom_net_crop_fix")
class FathomnetCropFixRunner(MPIRunnerTool):
    """
    Runner for FathomNet crop fix operations.

    This class implements the actual cropping correction logic. It loads
    original images, applies the correct cropping parameters, and saves
    the properly cropped images to the specified output location.
    """

    def __init__(self, cfg: Config):
        """
        Initialize the runner with configuration and load necessary data tables.

        Args:
            cfg (Config): Configuration object with paths and settings.
        """
        super().__init__(cfg)
        self.filter_name: str = "fathom_net_crop_fix"
        self.data_scheme: List[str] = ["uuid", "server", "path"]
        self.verification_scheme: List[str] = ["server"]
        self.total_time = 150  # Time buffer for timeout checking in seconds

        # Load reference dataframes
        self.data_transfer_df = pd.read_csv(cfg["data_transfer_table"])
        self.bb_df = pd.read_csv(cfg["filtered_by_size"])  # Bounding box information

        # Set path configurations
        self.image_crop_path = cfg["image_crop_path"]  # Output path for corrected crops
        self.base_path = cfg["base_path"]  # Base path for current dataset
        self.original_image_base_path = cfg[
            "original_image_base_path"
        ]  # Path to original uncropped images

    def apply_filter(self, filtering_df: pd.DataFrame, server_name: str) -> int:
        """
        Apply the cropping fix to images from a specific server.

        This function:
        1. Finds the matching UUIDs that need fixing
        2. For each path, loads the original uncropped image
        3. Applies the correct cropping parameters
        4. Saves the properly cropped images

        Args:
            filtering_df (pd.DataFrame): DataFrame with UUIDs and paths to process
            server_name (str): Name of the server being processed

        Returns:
            int: Number of images successfully processed
        """
        self.is_enough_time()
        # Find matching UUIDs that need cropping fixes
        uuids_df = self.bb_df.merge(
            filtering_df[["uuid"]],
            left_on="tol_uuid",
            right_on="uuid",
            how="inner",
            validate="1:1",
        )
        cropped_images = []

        for full_path, images_df in filtering_df.groupby("path"):
            assert isinstance(full_path, str), "Not a string"

            # Find the original image path using the data transfer mapping table
            file_name = os.path.basename(full_path)
            original_image_path = (
                self.original_image_base_path
                + self.data_transfer_df[
                    self.data_transfer_df["dst_path"]
                    == os.path.join(self.base_path, f"server={server_name}", file_name)
                ].iloc[0]["src_path"][
                    67:
                ]  # Specific to our dataset, will need to be changed in yours
            )

            if not os.path.exists(original_image_path):
                self.logger.info(f"Path doesn't exists: {original_image_path}")
                return 0

            # Load the original full-sized images that need fixing
            full_image = pd.read_parquet(
                original_image_path,
                filters=[("source_id", "in", uuids_df["image_uuid"])],
            )

            self.is_enough_time()  # Check if we still have enough time to continue

            columns = full_image.columns
            # Merge to get bounding box information for each image
            full_image = full_image.merge(
                self.bb_df,
                left_on="source_id",
                right_on="image_uuid",
                how="inner",
                validate="1:m",
            )

            for _, row in full_image.iterrows():
                # Create a new entry for the fixed cropped image
                cropped_entry = row[columns].to_dict()
                image_binary = row["image"]
                image_size = row["resized_size"]

                # Convert binary image data to numpy array
                image_np = np.frombuffer(image_binary, dtype=np.uint8).reshape(
                    [image_size[0], image_size[1], 3]
                )

                # Calculate corrected crop coordinates with proper bounds checking
                min_y = min(image_size[0], max(row["y"], 0))
                min_x = min(image_size[1], max(row["x"], 0))
                max_y = min(image_size[0], max(row["y"] + row["height"], 0))
                max_x = min(image_size[1], max(row["x"] + row["width"], 0))

                # Crop the image with corrected coordinates
                image_cropped = image_np[min_y:max_y, min_x:max_x]

                # Update entry with the new cropped image data
                cropped_entry["image"] = image_cropped.tobytes()
                cropped_entry["resized_size"] = (max_y - min_y, max_x - min_x)
                cropped_entry["hashsum_resized"] = hashlib.md5(
                    cropped_entry["image"]
                ).hexdigest()
                cropped_entry["uuid"] = row["tol_uuid"]
                cropped_entry["source_id"] = row["bb_uuid"]

                # Verify the image data size matches dimensions
                assert len(cropped_entry["image"]) == (
                    cropped_entry["resized_size"][0]
                    * cropped_entry["resized_size"][1]
                    * 3
                ), f"Size mismatch for {row['tol_uuid']}"

                cropped_images.append(cropped_entry)

        self.is_enough_time()
        # Create a DataFrame from all processed images
        cropped_image = pd.DataFrame(cropped_images)
        output_path = os.path.join(self.image_crop_path, f"server={server_name}")
        os.makedirs(output_path, exist_ok=True)

        # Save the corrected images with a unique filename
        cropped_image.to_parquet(
            os.path.join(output_path, f"data_{uuid.uuid4()}.parquet"),
            index=False,
            compression="zstd",
            compression_level=3,
        )

        return len(cropped_image)

    def runner_fn(self, df_local: pd.DataFrame) -> int:
        """
        Process function called for each batch of data in the distributed execution.

        Handles the execution of the crop fix operation for a specific server,
        with error handling and reporting.

        Args:
            df_local (pd.DataFrame): Local partition of the data to process

        Returns:
            int: 1 if successful, 0 if failed
        """
        filtering_df = df_local.reset_index(drop=True)
        server_name = filtering_df.iloc[0]["server"]
        try:
            filtered_parquet_length = self.apply_filter(filtering_df, server_name)
        except NotImplementedError:
            raise NotImplementedError("Filter function wasn't implemented")
        except Exception as e:
            self.logger.exception(e)
            self.logger.error(f"Error occurred: {e}")
            return 0
        else:
            # Log successful completion for verification
            print(f"{server_name}", end="\n", file=self.verification_IO)
            self.logger.debug(
                f"Completed filtering: {server_name} with {filtered_parquet_length}"
            )
            return 1
