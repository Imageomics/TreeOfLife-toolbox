import hashlib
import os
import uuid
from typing import List

import numpy as np
import pandas as pd
import pyspark.sql.functions as func

from DD_tools.main.config import Config
from DD_tools.main.filters import FilterRegister, SparkFilterToolBase
from DD_tools.main.runners import MPIRunnerTool, RunnerRegister
from DD_tools.main.schedulers import DefaultScheduler, SchedulerRegister
from DD_tools.main.utils import load_dataframe

server_pattern = r"server=([^/]+)"


@FilterRegister("fathom_net_crop_fix")
class FathomnetCropFixFilter(SparkFilterToolBase):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.filter_name: str = "fathom_net_crop_fix"

    def run(self):
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
    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.filter_name: str = "fathom_net_crop_fix"
        self.scheme = ["server"]


@RunnerRegister("fathom_net_crop_fix")
class FathomnetCropFixRunner(MPIRunnerTool):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.filter_name: str = "fathom_net_crop_fix"
        self.data_scheme: List[str] = ["uuid", "server", "path"]
        self.verification_scheme: List[str] = ["server"]
        self.total_time = 150

        self.data_transfer_df = pd.read_csv(cfg["data_transfer_table"])
        self.bb_df = pd.read_csv(cfg["filtered_by_size"])
        self.image_crop_path = os.path.join(
            cfg.get_folder("path_to_output_folder"), "image_crop"
        )
        self.base_path = (
            "/fs/scratch/PAS2136/gbif/processed/fathomNet/images_full/source=fathomNet"
        )
        self.original_image_base_path = (
            "/fs/scratch/PAS2136/gbif/processed/fathomNet/images_full/downloaded_images"
        )

    def apply_filter(self, filtering_df: pd.DataFrame, server_name: str) -> int:
        self.is_enough_time()
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

            file_name = os.path.basename(full_path)
            original_image_path = (
                    self.original_image_base_path
                    + self.data_transfer_df[
                          self.data_transfer_df["dst_path"]
                          == os.path.join(self.base_path, f"server={server_name}", file_name)
                          ].iloc[0]["src_path"][67:]
            )

            if not os.path.exists(original_image_path):
                self.logger.info(f"Path doesn't exists: {original_image_path}")
                return 0

            full_image = pd.read_parquet(
                original_image_path,
                filters=[("source_id", "in", uuids_df["image_uuid"])],
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
                        cropped_entry["resized_size"][0]
                        * cropped_entry["resized_size"][1]
                        * 3
                ), f"Size mismatch for {row['tol_uuid']}"

                cropped_images.append(cropped_entry)

        self.is_enough_time()
        cropped_image = pd.DataFrame(cropped_images)
        output_path = os.path.join(self.image_crop_path, f"server={server_name}")
        os.makedirs(output_path, exist_ok=True)
        cropped_image.to_parquet(
            os.path.join(output_path, f"data_{uuid.uuid4()}.parquet"),
            index=False,
            compression="zstd",
            compression_level=3,
        )

        return len(cropped_image)

    def runner_fn(self, df_local: pd.DataFrame) -> int:
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
            print(f"{server_name}", end="\n", file=self.verification_IO)
            self.logger.debug(
                f"Completed filtering: {server_name} with {filtered_parquet_length}"
            )
            return 1
