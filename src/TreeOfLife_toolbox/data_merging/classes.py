from TreeOfLife_toolbox.main.config import Config
from TreeOfLife_toolbox.main.filters import FilterRegister, SparkFilterToolBase
from TreeOfLife_toolbox.main.runners import RunnerRegister, FilterRunnerTool
from TreeOfLife_toolbox.main.schedulers import DefaultScheduler, SchedulerRegister


@FilterRegister("merged_duplicated_check")
class DataMergedDupCheckFilter(SparkFilterToolBase):
    """
    Filter that identifies duplicate records between a target dataset and a source dataset.
    
    This filter compares records based on their 'hashsum_original' column to identify
    entries that exist in both the target dataset (specified by 'merge_target' in config)
    and the source dataset (downloaded images).
    
    Attributes:
        filter_name (str): Name identifier for the filter, set to "merged_duplicated_check".
        merge_target (str): Path to the target dataset for duplicate checking.
    """

    def __init__(self, cfg: Config):
        """
        Initialize the duplicate check filter with configuration.
        
        Args:
            cfg (Config): Configuration object containing filter settings
                          including the 'merge_target' path.
        """
        super().__init__(cfg)

        self.filter_name: str = "merged_duplicated_check"
        self.merge_target: str = str(self.config["merge_target"])

    def run(self):
        """
        Execute the duplicate check process.
        
        This method:
        1. Reads the target dataset from the specified merge_target path
        2. Reads the source dataset from the downloaded_images_path
        3. Joins the datasets on 'hashsum_original' to find duplicates
        4. Saves the duplicate records and logs the count
        
        Returns:
            None
        """
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
    """
    Scheduler for the duplicate check process.
    
    Extends the DefaultScheduler to handle scheduling of the 
    duplicate checking tasks. This scheduler is registered with
    the "merged_duplicated_check" name.
    
    Attributes:
        filter_name (str): Name identifier for the filter, set to "merged_duplicated_check".
    """
    def __init__(self, cfg: Config):
        """
        Initialize the duplicate check scheduler with configuration.
        
        Args:
            cfg (Config): Configuration object for scheduling parameters.
        """
        super().__init__(cfg)

        self.filter_name: str = "merged_duplicated_check"


@RunnerRegister("merged_duplicated_check")
class DataMergedDupCheckRunner(FilterRunnerTool):
    """
    Runner for executing the duplicate check filter.
    
    Handles the execution flow of the duplicate checking process,
    extending the FilterRunnerTool to provide specific functionality
    for the merged_duplicated_check filter.
    
    Attributes:
        filter_name (str): Name identifier for the filter, set to "merged_duplicated_check".
    """
    def __init__(self, cfg: Config):
        """
        Initialize the duplicate check runner with configuration.
        
        Args:
            cfg (Config): Configuration object containing runner settings.
        """
        super().__init__(cfg)

        self.filter_name: str = "merged_duplicated_check"
