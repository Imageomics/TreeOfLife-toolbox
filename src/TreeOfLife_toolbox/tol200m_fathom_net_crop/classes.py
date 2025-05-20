import hashlib
import os
import shutil
from typing import List

import numpy as np
import pandas as pd

from TreeOfLife_toolbox.main.config import Config
from TreeOfLife_toolbox.main.filters import PythonFilterToolBase, FilterRegister
from TreeOfLife_toolbox.main.runners import MPIRunnerTool, RunnerRegister
from TreeOfLife_toolbox.main.schedulers import DefaultScheduler, SchedulerRegister


@FilterRegister("tol200m_fathom_net_crop")
class ToLFathomnetCropFilter(PythonFilterToolBase):
    """
    Filter tool that prepares a list of image files from the downloaded dataset
    that need to be processed for the FathomNet crop operation.

    This class identifies all available image partitions from the downloaded dataset
    that can be processed for cropping based on the bounding box information.
    """

    def __init__(self, cfg: Config):
        """
        Initialize the ToLFathomnetCropFilter with configuration.

        Args:
            cfg (Config): Configuration object with paths and settings.
        """
        super().__init__(cfg)
        self.filter_name: str = "tol200m_fathom_net_crop"


@SchedulerRegister("tol200m_fathom_net_crop")
class ToLFathomnetCropScheduleCreation(DefaultScheduler):
    """
    Scheduler for the FathomNet crop operation.

    Creates a schedule for distributed processing of image cropping tasks,
    assigning server/partition pairs to worker ranks for load balancing.
    Uses the default scheduling algorithm from the DefaultScheduler class.
    """

    def __init__(self, cfg: Config):
        """
        Initialize the ToLFathomnetCropScheduleCreation with configuration.

        Args:
            cfg (Config): Configuration object with paths and settings.
        """
        super().__init__(cfg)
        self.filter_name: str = "tol200m_fathom_net_crop"


@RunnerRegister("tol200m_fathom_net_crop")
class ToLFathomnetCropRunner(MPIRunnerTool):
    """
    Runner tool that performs the actual cropping operation on images
    based on bounding box information.

    This class loads images from the distributed dataset, crops them according
    to the bounding box coordinates, and saves the cropped images to a new location.
    It operates in a distributed manner using MPI for parallel processing.
    """

    def __init__(self, cfg: Config):
        """
        Initialize the ToLFathomnetCropRunner with configuration.

        Args:
            cfg (Config): Configuration object with paths and settings.
                          Must contain the 'image_crop_path' key.
        """
        super().__init__(cfg)
        self.filter_name: str = "tol200m_fathom_net_crop"
        self.data_scheme: List[str] = ["server_name", "partition_id"]
        self.verification_scheme: List[str] = ["server_name", "partition_id"]
        self.total_time = 150
        # Load bounding box information from the CSV file specified in the config
        self.bb_df = self.__read_parquets(
            os.path.join(self.config.get_folder("tools_folder"), "uuid_ref")
        ).rename({"source_id": "bb_uuid", "uuid": "tol_uuid"}, axis=1)
        # Path where cropped images will be stored
        self.image_crop_path = self.config["image_crop_path"]

    @staticmethod
    def __read_parquets(path: str) -> pd.DataFrame:
        """
        Read parquet files from the specified path and return a DataFrame.

        Args:
            path (str): Path to the directory containing parquet files.

        Returns:
            pd.DataFrame: DataFrame containing the data from the parquet files.
        """
        # Read all parquet files in the directory and concatenate them into a single DataFrame
        return pd.concat(
            [
                pd.read_parquet(os.path.join(path, file))
                for file in os.listdir(path)
                if file.endswith(".parquet")
            ]
        )

    def apply_filter(
        self, filtering_df: pd.DataFrame, server_name: str, partition_id: int
    ) -> int:
        """
        Process a batch of images from a specific server and partition,
        cropping them according to bounding box information.

        This method:
        1. Checks if enough time remains to process the batch
        2. Locates and loads the relevant images
        3. Filters images to only those with bounding box information
        4. Crops each image according to its bounding box
        5. Saves the cropped images to a new location

        Args:
            filtering_df (pd.DataFrame): DataFrame with filtering information
            server_name (str): Name of the server containing the images
            partition_id (int): ID of the partition to process

        Returns:
            int: Number of images successfully processed

        Notes:
            - If a bounding box extends outside the image, it will be clipped
              to the image boundaries
            - Cropped images maintain the same metadata as the original,
              with updated size and hash information
        """
        self.is_enough_time()

        input_path = os.path.join(
            self.downloaded_images_path,
            f"server_name={server_name}",
            f"partition_id={partition_id}",
        )
        parquet_path = os.path.join(input_path, "successes.parquet")

        if not os.path.exists(parquet_path):
            self.logger.info(f"Path doesn't exists: {parquet_path}")
            return 0

        # Load only images that have corresponding bounding box information
        full_image = pd.read_parquet(
            parquet_path, filters=[("uuid", "in", self.bb_df["image_uuid"])]
        )

        self.is_enough_time()

        columns = full_image.columns
        # Merge with bounding box information
        full_image = full_image.merge(
            self.bb_df,
            left_on="uuid",
            right_on="image_uuid",
            how="inner",
            validate="1:m",
        )
        cropped_images = []
        for _, row in full_image.iterrows():
            cropped_entry = row[columns].to_dict()
            image_binary = row["image"]
            image_size = row["resized_size"]
            # Convert binary image data to numpy array with proper dimensions
            image_np = np.frombuffer(image_binary, dtype=np.uint8).reshape(
                [image_size[0], image_size[1], 3]
            )
            # Ensure bounding box coordinates are within image boundaries
            min_y = min(image_size[0], max(row["y"], 0))
            min_x = min(image_size[1], max(row["x"], 0))
            max_y = min(image_size[0], max(row["y"] + row["height"], 0))
            max_x = min(image_size[1], max(row["x"] + row["width"], 0))

            # Crop the image
            image_cropped = image_np[min_y:max_y, min_x:max_x]

            # Update the entry with cropped image information
            cropped_entry["image"] = image_cropped.tobytes()
            cropped_entry["resized_size"] = (max_y - min_y, max_x - min_x)
            cropped_entry["hashsum_resized"] = hashlib.md5(
                cropped_entry["image"]
            ).hexdigest()
            cropped_entry["uuid"] = row["tol_uuid"]
            cropped_entry["source_id"] = row["bb_uuid"]

            # Validate that the image size matches the expected dimensions
            assert len(cropped_entry["image"]) == (
                cropped_entry["resized_size"][0] * cropped_entry["resized_size"][1] * 3
            ), f"Size mismatch for {row['tol_uuid']}"

            cropped_images.append(cropped_entry)

        self.is_enough_time()
        # Create a DataFrame with all the cropped images
        full_image = pd.DataFrame(cropped_images)
        # Prepare output path
        output_path = os.path.join(
            self.image_crop_path,
            f"server_name={server_name}",
            f"partition_id={partition_id}",
        )
        os.makedirs(output_path, exist_ok=True)
        # Save the cropped images as a parquet file
        full_image.to_parquet(
            os.path.join(output_path, "successes.parquet"),
            index=False,
            compression="zstd",
            compression_level=3,
        )
        # Copy other necessary files to maintain the same structure
        for file in ["errors.parquet", "completed"]:
            shutil.copyfile(
                os.path.join(input_path, file), os.path.join(output_path, file)
            )

        return len(full_image)