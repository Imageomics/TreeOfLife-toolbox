import os
import shutil
from typing import List

import pandas as pd

from DD_tools.main.config import Config
from DD_tools.main.filters import PythonFilterToolBase, FilterRegister
from DD_tools.main.runners import MPIRunnerTool, RunnerRegister
from DD_tools.main.schedulers import DefaultScheduler, SchedulerRegister


@FilterRegister("lila_separation")
class LilaSeparationFilter(PythonFilterToolBase):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.filter_name: str = "lila_separation"
        self.data_path = "/users/PAS2119/andreykopanev/gbif/data/lila_separation_table/part-00000-6e425202-ecec-426d-9631-f2f52fd45c51-c000.csv"

    def run(self):
        filter_table_folder = os.path.join(
            self.tools_path, self.filter_name, "filter_table"
        )
        os.makedirs(filter_table_folder, exist_ok=True)
        filter_table_folder += "/table.csv"

        shutil.copyfile(self.data_path, filter_table_folder)


@SchedulerRegister("lila_separation")
class LilaSeparationScheduleCreation(DefaultScheduler):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.filter_name: str = "lila_separation"


@RunnerRegister("lila_separation")
class LilaSeparationRunner(MPIRunnerTool):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.filter_name: str = "lila_separation"

        self.data_scheme: List[str] = [
            "uuid",
            "source_id",
            "uuid_main",
            "source_id_main",
            "server_name",
            "old_partition_id",
            "partition_id",
        ]
        self.verification_scheme: List[str] = ["server_name", "partition_id"]
        self.new_path = "/fs/scratch/PAS2136/gbif/processed/lilabc/separated_multilabel_data/downloaded_image"
        self.total_time = 600

    def apply_filter(
            self, filtering_df: pd.DataFrame, server_name: str, partition_id: str
    ) -> int:
        self.is_enough_time()

        # self.downloaded_images_path
        filtering_df_grouped = filtering_df.groupby(["server_name", "old_partition_id"])
        separated_dict = []
        for name, group in filtering_df_grouped:
            parquet_path = os.path.join(
                self.downloaded_images_path,
                f"server_name={name[0]}",
                f"partition_id={name[1]}",
                "successes.parquet",
            )
            if not os.path.exists(parquet_path):
                self.logger.info(f"Path doesn't exists: {server_name}/{partition_id}")
                continue

            partial_df = pd.read_parquet(
                parquet_path, filters=[("uuid", "in", group["uuid_main"])]
            )
            partial_merged_df = pd.merge(
                partial_df,
                group,
                left_on="uuid",
                right_on="uuid_main",
                suffixes=("_x", "_y"),
                sort=False,
                how="right",
            )

            partial_merged_df = partial_merged_df[
                [
                    "uuid_y",
                    "source_id_y",
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
            ]
            separated_dict.extend(
                partial_merged_df.rename(
                    {"uuid_y": "uuid", "source_id_y": "source_id"}, inplace=True
                ).to_dict("records")
            )

        merged_df = pd.DataFrame.from_records(separated_dict)

        self.is_enough_time()

        save_path = os.path.join(
            self.new_path, f"server_name={server_name}", f"partition_id={partition_id}"
        )
        os.makedirs(save_path, exist_ok=True)

        if len(merged_df) == 0:
            self.logger.info(f"Empty: {server_name}/{partition_id}")

        merged_df.to_parquet(
            save_path + "/successes.parquet",
            index=False,
            compression="zstd",
            compression_level=3,
        )

        return len(merged_df)
