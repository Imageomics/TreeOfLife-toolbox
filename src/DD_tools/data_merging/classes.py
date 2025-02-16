from DD_tools.main.config import Config
from DD_tools.main.filters import FilterRegister, SparkFilterToolBase
from DD_tools.main.runners import RunnerRegister, FilterRunnerTool
from DD_tools.main.schedulers import DefaultScheduler, SchedulerRegister


@FilterRegister("merged_duplicated_check")
class DataMergedDupCheckFilter(SparkFilterToolBase):
    merge_target = "/fs/ess/PAS2136/TreeOfLife/data"

    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.filter_name: str = "merged_duplicated_check"

    def run(self):
        from pyspark.sql import DataFrame
        import pyspark.sql.functions as func

        target_df: DataFrame = (
            self.spark.read.option("basePath", self.merge_target)
            .parquet(self.merge_target + "/source=*/server=*/data_*.parquet")
            .select("uuid", "source_id", "hashsum_original")
            .withColumnsRenamed({"uuid": "uuid_main", "source_id": "source_id_main"})
        )

        target_df = target_df.withColumn("file_main", func.input_file_name())

        object_df = (
            self.spark.read.schema(self.success_scheme)
            .option("basePath", self.downloaded_images_path)
            .parquet(self.downloaded_images_path + "/*/*/successes.parquet")
            .select(
                "uuid", "source_id", "server_name", "partition_id", "hashsum_original"
            )
        )

        duplicate_records = object_df.join(
            target_df, on=["hashsum_original"], how="inner"
        )

        self.save_filter(duplicate_records)

        self.logger.info(f"duplicated number: {duplicate_records.count()}")


@SchedulerRegister("merged_duplicated_check")
class DataMergedDupCheckScheduleCreation(DefaultScheduler):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.filter_name: str = "merged_duplicated_check"


@RunnerRegister("merged_duplicated_check")
class DataMergedDupCheckRunner(FilterRunnerTool):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.filter_name: str = "merged_duplicated_check"
