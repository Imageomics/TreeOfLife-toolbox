import pyspark.sql as ps
from pyspark.sql import SparkSession

from TreeOfLife_toolbox.main.config import Config
from TreeOfLife_toolbox.main.filters import FilterRegister, SparkFilterToolBase
from TreeOfLife_toolbox.main.runners import RunnerRegister, FilterRunnerTool
from TreeOfLife_toolbox.main.schedulers import DefaultScheduler, SchedulerRegister


@FilterRegister("lila_bc_filtering")
class LilaBCFilter(SparkFilterToolBase):
    """
    Filter class for filtering out images from LILA Biodiversity Catalog based on specified labels.
    
    This class identifies images that have labels matching those in the excluding labels file
    and creates a filter table containing UUIDs of those images. These images will later
    be removed from the dataset by the runner.
    
    Attributes:
        filter_name (str): Name of the filter tool, used for folder structure.
        path_to_excluding_labels (str): Path to CSV containing labels to be excluded.
    """

    def __init__(self, cfg: Config, spark: SparkSession = None):
        """
        Initialize the LILA BC filter with configuration.
        
        Args:
            cfg (Config): Configuration object containing paths and settings.
            spark (SparkSession, optional): Existing SparkSession. If None, a new one will be created.
        """
        super().__init__(cfg, spark)
        self.filter_name: str = "lila_bc_filtering"
        self.path_to_excluding_labels = cfg["path_to_excluding_labels"]

    def run(self):
        """
        Execute the filtering process.
        
        This method:
        1. Loads the image data from the parquet files
        2. Loads the original labels from URLs table
        3. Loads the labels to be excluded
        4. Joins the datasets to identify images with labels to exclude
        5. Saves the filter table for later processing by the runner
        
        Returns:
            None
        """
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
    """
    Scheduler for LILA BC filtering tool.
    
    This class creates a schedule for parallel processing of the filtering task.
    It inherits from DefaultScheduler which manages the distribution of work
    across available workers.
    
    Attributes:
        filter_name (str): Name of the filter tool, used for folder structure.
    """
    def __init__(self, cfg: Config):
        """
        Initialize the LILA BC scheduler.
        
        Args:
            cfg (Config): Configuration object containing paths and settings.
        """
        super().__init__(cfg)

        self.filter_name: str = "lila_bc_filtering"


@RunnerRegister("lila_bc_filtering")
class LilaBCRunner(FilterRunnerTool):
    """
    Runner for LILA BC filtering tool.
    
    This class executes the actual filtering operation by removing images 
    with specified labels. It uses MPI to distribute work across multiple nodes.
    Inherits from FilterRunnerTool which provides common functionality for
    filtering operations on downloaded images.
    
    Attributes:
        filter_name (str): Name of the filter tool, used for folder structure.
    """
    def __init__(self, cfg: Config):
        """
        Initialize the LILA BC runner.
        
        Args:
            cfg (Config): Configuration object containing paths and settings.
        """
        super().__init__(cfg)

        self.filter_name: str = "lila_bc_filtering"
