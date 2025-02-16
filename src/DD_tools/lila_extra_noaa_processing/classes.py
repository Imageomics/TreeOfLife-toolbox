import hashlib
import os
import uuid
from typing import List

import cv2
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

from DD_tools.main.config import Config
from DD_tools.main.filters import FilterRegister, SparkFilterToolBase
from DD_tools.main.runners import MPIRunnerTool, RunnerRegister
from DD_tools.main.schedulers import DefaultScheduler, SchedulerRegister
from DD_tools.main.utils import load_dataframe

images_root = "/fs/scratch/PAS2136/gbif/processed/lilabc/extra/noaa-kotz/"


@udf(returnType=StringType())
def get_uuid():
    return str(uuid.uuid4())


@FilterRegister("lila_extra_noaa_processing")
class LilaExtraNoaaFilter(SparkFilterToolBase):
    def __init__(self, cfg: Config, spark: SparkSession = None):
        super().__init__(cfg, spark)
        self.filter_name: str = "lila_extra_noaa_processing"

    def run(self):
        multimedia_df = (
            load_dataframe(self.spark, self.config["path_to_input"])
            .repartition(20)
            .withColumnsRenamed(
                {
                    "detection_id": "source_id",
                    "detection_type": "life_stage",
                    "rgb_left": "left",
                    "rgb_right": "right",
                    "rgb_top": "top",
                    "rgb_bottom": "bottom",
                }
            )
        )

        multimedia_df_prep = multimedia_df.withColumn(
            "identifier", F.concat(F.lit(images_root), F.col("rgb_image_path"))
        )

        multimedia_df_prep = multimedia_df_prep.withColumn("server_name", F.lit("noaa"))
        multimedia_df_prep = multimedia_df_prep.withColumn("uuid", get_uuid())

        columns = multimedia_df_prep.columns

        self.logger.info("Starting batching")

        servers_grouped = (
            multimedia_df_prep.select("server_name")
            .groupBy("server_name")
            .count()
            .withColumn(
                "batch_count",
                F.floor(
                    F.col("count") / self.config["downloader_parameters"]["batch_size"]
                ),
            )
        )

        window_part = Window.partitionBy("server_name").orderBy("server_name")
        master_df_filtered = (
            multimedia_df_prep.withColumn(
                "row_number", F.row_number().over(window_part)
            )
            .join(servers_grouped, ["server_name"])
            .withColumn("partition_id", F.col("row_number") % F.col("batch_count"))
            .withColumn(
                "partition_id",
                (
                    F.when(F.col("partition_id").isNull(), 0).otherwise(
                        F.col("partition_id")
                    )
                ),
            )
            .select(*columns, "partition_id")
        )

        self.logger.info("Writing to parquet")

        (
            master_df_filtered.repartition("server_name", "partition_id")
            .write.partitionBy("server_name", "partition_id")
            .mode("overwrite")
            .format("parquet")
            .save(self.urls_path)
        )

        filtered_df = master_df_filtered.select(
            "uuid",
            "source_id",
            "identifier",
            "left",
            "right",
            "top",
            "bottom",
            "server_name",
            "partition_id",
        )

        self.save_filter(filtered_df)

        self.logger.info("Finished batching")
        self.logger.info(f"Too small images number: {master_df_filtered.count()}")


@SchedulerRegister("lila_extra_noaa_processing")
class LilaExtraNoaaScheduleCreation(DefaultScheduler):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.filter_name: str = "lila_extra_noaa_processing"


@RunnerRegister("lila_extra_noaa_processing")
class LilaExtraNoaaRunner(MPIRunnerTool):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.filter_name: str = "lila_extra_noaa_processing"
        self.data_scheme: List[str] = [
            "uuid",
            "source_id",
            "left",
            "right",
            "top",
            "bottom",
            "server_name",
            "partition_id",
        ]
        self.verification_scheme: List[str] = ["server_name", "partition_id"]
        self.total_time = 1000

    def apply_filter(
            self, filtering_df: pd.DataFrame, server_name: str, partition_id: str
    ) -> int:
        self.is_enough_time()

        parquet_folder_path = os.path.join(
            self.downloaded_images_path,
            f"server_name={server_name}",
            f"partition_id={partition_id}",
        )
        os.makedirs(parquet_folder_path)
        parquet_path = os.path.join(parquet_folder_path, "successes.parquet")

        images = []
        for _, row in filtering_df.iterrows():
            new_entry = {
                "uuid": row["uuid"],
                "source_id": row["source_id"],
                "identifier": row["identifier"],
                "is_license_full": False,
                "license": None,
                "source": None,
                "title": None,
                "original_size": "",
                "resized_size": "",
                "hashsum_original": "",
                "hashsum_resized": "",
                "image": "",
            }

            image = cv2.imread(row["identifier"])
            cropped = image[row["bottom"]: row["top"], row["left"]: row["right"]]
            cropped_binary = cropped.tobytes()

            new_entry["original_size"] = image.shape[:2]
            new_entry["resized_size"] = cropped.shape[:2]
            new_entry["hashsum_original"] = hashlib.md5(image.tobytes()).hexdigest()
            new_entry["hashsum_resized"] = hashlib.md5(cropped_binary).hexdigest()
            new_entry["image"] = cropped_binary

            images.append(new_entry)

        filtered_parquet = pd.DataFrame(images)

        self.is_enough_time()

        filtered_parquet.to_parquet(
            parquet_path, index=False, compression="zstd", compression_level=3
        )

        return len(filtered_parquet)
