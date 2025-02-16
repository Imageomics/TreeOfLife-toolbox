import pyspark.sql as ps
from pyspark.sql import SparkSession

from DD_tools.main.config import Config
from DD_tools.main.filters import FilterRegister, SparkFilterToolBase
from DD_tools.main.runners import RunnerRegister, FilterRunnerTool
from DD_tools.main.schedulers import DefaultScheduler, SchedulerRegister


@FilterRegister("lila_bc_filtering")
class LilaBCFilter(SparkFilterToolBase):
    path_to_excluding_labels = (
        "/fs/scratch/PAS2136/gbif/processed/lilabc/distinct_labels/labels_to_remove.csv"
    )

    def __init__(self, cfg: Config, spark: SparkSession = None):
        super().__init__(cfg, spark)
        self.filter_name: str = "lila_bc_filtering"

    def run(self):
        successes_df: ps.DataFrame = self.load_data_parquet()
        data_df = self.spark.read.parquet(self.urls_path).select(
            "uuid", "original_label"
        )
        labels_to_exclude_df = self.spark.read.csv(
            self.path_to_excluding_labels, header=True
        )

        merged_df = successes_df.join(data_df, on="uuid", how="inner")
        filtered_df = merged_df.join(
            labels_to_exclude_df, on="original_label", how="inner"
        ).select("uuid", "source_id", "server_name", "partition_id")

        self.save_filter(filtered_df)

        self.logger.info(f"Images to filter out: {filtered_df.count()}")


@SchedulerRegister("lila_bc_filtering")
class LilaBCScheduleCreation(DefaultScheduler):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.filter_name: str = "lila_bc_filtering"


@RunnerRegister("lila_bc_filtering")
class LilaBCRunner(FilterRunnerTool):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.filter_name: str = "lila_bc_filtering"
