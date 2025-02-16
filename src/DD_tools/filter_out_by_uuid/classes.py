import os
from typing import List

import pandas as pd

from DD_tools.main.config import Config
from DD_tools.main.filters import FilterRegister, SparkFilterToolBase
from DD_tools.main.runners import MPIRunnerTool, RunnerRegister
from DD_tools.main.schedulers import DefaultScheduler, SchedulerRegister
from DD_tools.main.utils import load_dataframe


@FilterRegister("filter_out_by_uuid")
class FilterOutByUUIDFilter(SparkFilterToolBase):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.filter_name: str = "filter_out_by_uuid"

    def run(self):
        uuid_table_df = load_dataframe(
            self.spark, self.config["uuid_table_path"]
        ).repartition(20)
        lookup_table_df = load_dataframe(
            self.spark, self.config["look_up_table_path"]
        ).repartition(20)

        merged_df = uuid_table_df.join(lookup_table_df, on="uuid", how="left")

        (
            merged_df.repartition(1).write.csv(
                os.path.join(self.tools_path, self.filter_name, "filter_table"),
                header=True,
                mode="overwrite",
            )
        )


@SchedulerRegister("filter_out_by_uuid")
class FilterOutByUUIDScheduleCreation(DefaultScheduler):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.filter_name: str = "filter_out_by_uuid"
        self.scheme = ["path"]


@RunnerRegister("filter_out_by_uuid")
class FilterOutByUUIDRunner(MPIRunnerTool):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.filter_name: str = "filter_out_by_uuid"
        self.data_scheme: List[str] = ["uuid", "path"]
        self.verification_scheme: List[str] = ["path"]
        self.total_time = 150

    def apply_filter(self, filtering_df: pd.DataFrame, file_path: str) -> int:
        self.is_enough_time()

        if not os.path.exists(file_path):
            self.logger.info(f"Path doesn't exists: {file_path}")
            return 0

        filtered_parquet = pd.read_parquet(
            file_path, filters=[("uuid", "not in", filtering_df["uuid"])]
        )

        self.is_enough_time()

        if len(filtered_parquet) == 0:
            self.logger.info(f"Fully filtered out: {file_path}")

        filtered_parquet.to_parquet(
            file_path, index=False, compression="zstd", compression_level=3
        )

        return len(filtered_parquet)

    def runner_fn(self, df_local: pd.DataFrame) -> int:
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
