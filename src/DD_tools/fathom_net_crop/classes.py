import hashlib
import os
import shutil
from typing import List

import numpy as np
import pandas as pd

from DD_tools.main.config import Config
from DD_tools.main.filters import PythonFilterToolBase, FilterRegister
from DD_tools.main.runners import MPIRunnerTool, RunnerRegister
from DD_tools.main.schedulers import DefaultScheduler, SchedulerRegister


@FilterRegister("fathom_net_crop")
class FathomnetCropFilter(PythonFilterToolBase):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.filter_name: str = "fathom_net_crop"


@SchedulerRegister("fathom_net_crop")
class FathomnetCropScheduleCreation(DefaultScheduler):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.filter_name: str = "fathom_net_crop"


@RunnerRegister("fathom_net_crop")
class FathomnetCropRunner(MPIRunnerTool):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.filter_name: str = "fathom_net_crop"
        self.data_scheme: List[str] = ["server_name", "partition_id"]
        self.verification_scheme: List[str] = ["server_name", "partition_id"]
        self.total_time = 150
        self.bb_df = pd.read_csv(
            "/fs/scratch/PAS2136/gbif/processed/fathomNet/filtered_by_size.csv"
        )
        self.image_crop_path = os.path.join(
            cfg.get_folder("path_to_output_folder"), "image_crop"
        )

    def apply_filter(
            self, filtering_df: pd.DataFrame, server_name: str, partition_id: int
    ) -> int:
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

        full_image = pd.read_parquet(
            parquet_path, filters=[("source_id", "in", self.bb_df["image_uuid"])]
        )

        self.is_enough_time()

        columns = full_image.columns
        full_image = full_image.merge(
            self.bb_df,
            left_on="source_id",
            right_on="image_uuid",
            how="inner",
            validate="1:m",
        )
        cropped_images = []
        for _, row in full_image.iterrows():
            cropped_entry = row[columns].to_dict()
            image_binary = row["image"]
            image_size = row["resized_size"]
            image_np = np.frombuffer(image_binary, dtype=np.uint8).reshape(
                [image_size[0], image_size[1], 3]
            )
            # fix
            min_y = min(image_size[0], max(row["y"], 0))
            min_x = min(image_size[1], max(row["x"], 0))
            max_y = min(image_size[0], max(row["y"] + row["height"], 0))
            max_x = min(image_size[1], max(row["x"] + row["width"], 0))

            image_cropped = image_np[min_y:max_y, min_x:max_x]

            cropped_entry["image"] = image_cropped.tobytes()
            cropped_entry["resized_size"] = (max_y - min_y, max_x - min_x)
            cropped_entry["hashsum_resized"] = hashlib.md5(
                cropped_entry["image"]
            ).hexdigest()
            cropped_entry["uuid"] = row["tol_uuid"]
            cropped_entry["source_id"] = row["bb_uuid"]

            assert len(cropped_entry["image"]) == (
                    cropped_entry["resized_size"][0] * cropped_entry["resized_size"][1] * 3
            ), f"Size mismatch for {row['tol_uuid']}"

            cropped_images.append(cropped_entry)

        self.is_enough_time()
        full_image = pd.DataFrame(cropped_images)
        output_path = os.path.join(
            self.image_crop_path,
            f"server_name={server_name}",
            f"partition_id={partition_id}",
        )
        os.makedirs(output_path, exist_ok=True)
        full_image.to_parquet(
            os.path.join(output_path, "successes.parquet"),
            index=False,
            compression="zstd",
            compression_level=3,
        )
        for file in ["errors.parquet", "completed"]:
            shutil.copyfile(
                os.path.join(input_path, file), os.path.join(output_path, file)
            )

        return len(full_image)
