import os
import re
from typing import List

import pandas as pd

from DD_tools.main.config import Config
from DD_tools.main.filters import FilterRegister, SparkFilterToolBase
from DD_tools.main.runners import MPIRunnerTool, RunnerRegister
from DD_tools.main.schedulers import DefaultScheduler, SchedulerRegister


@FilterRegister("research_filtering")
class ResearchFilteringFilter(SparkFilterToolBase):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.filter_name: str = "research_filtering"
        self.string_to_remove = "file:/"

    def run(self):
        import pyspark.sql.functions as func

        occurrences_df = (
            self.spark.read.parquet(self.config["occurrences_path"])
            .select("gbifID", "basisOfRecord")
            .withColumnRenamed("gbifID", "source_id")
        )

        data_df = (
            self.spark.read.option("basePath", self.config["data_path"])
            .parquet(f"{self.config['data_path']}/source=*/server=*/data_*.parquet")
            .select("uuid", "source_id")
            .withColumn(
                "path",
                func.substring(
                    func.input_file_name(), len(self.string_to_remove), 2000000
                ),
            )
        )

        occurrences_df_filtered = occurrences_df.where(
            occurrences_df["basisOfRecord"].contains(self.config["basis_of_record"])
        )
        data_merged = data_df.join(occurrences_df_filtered, on="source_id", how="inner")

        (
            data_merged.repartition(1).write.csv(
                os.path.join(self.tools_path, self.filter_name, "filter_table"),
                header=True,
                mode="overwrite",
            )
        )


@SchedulerRegister("research_filtering")
class ResearchFilteringScheduleCreation(DefaultScheduler):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.filter_name: str = "research_filtering"
        self.scheme = ["path"]


@RunnerRegister("research_filtering")
class ResearchFilteringRunner(MPIRunnerTool):
    server_pattern = r"server=([^/]+)"
    source_pattern = r"source=([^/]+)"

    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.filter_name: str = "research_filtering"
        self.data_scheme: List[str] = ["uuid", "path"]
        self.verification_scheme: List[str] = ["path"]
        self.total_time = 150
        self.save_path_folder = "/fs/scratch/PAS2136/gbif/processed/research_filtering/"

    def apply_filter(self, filtering_df: pd.DataFrame, file_path: str) -> int:
        self.is_enough_time()

        if not os.path.exists(file_path):
            self.logger.info(f"Path doesn't exists: {file_path}")
            return 0

        server_name = re.findall(r"server=([^/]+)", file_path)[0]
        filename_path = os.path.basename(file_path)

        filtered_parquet = pd.read_parquet(
            file_path, filters=[("uuid", "not in", filtering_df["uuid"])]
        )

        self.is_enough_time()
        if len(filtered_parquet) == 0:
            self.logger.info(f"Fully filtered out: {server_name}/{filename_path}")

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
