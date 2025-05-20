import os
import shutil
from typing import List

from TreeOfLife_toolbox.main.config import Config
from TreeOfLife_toolbox.main.filters import FilterRegister, SparkFilterToolBase
from TreeOfLife_toolbox.main.runners import FilterRunnerTool, RunnerRegister
from TreeOfLife_toolbox.main.schedulers import DefaultScheduler, SchedulerRegister


@FilterRegister("lila_separation_single_label_filtering")
class LilaSeparationSingleLabelFilteringFilter(SparkFilterToolBase):
    """
    Filter class for separating single-label images from a dataset.
    
    This class is responsible for the initial filtering step in the single-label
    filtering process. It copies a provided CSV file containing information about
    single-label images to the filter table directory. This CSV file will later be
    used by the runner to filter the dataset.
    
    Attributes:
        filter_name (str): The name of the filter used for registration and folder creation.
        data_path (str): Path to the input CSV file containing single-label image information.
    """
    def __init__(self, cfg: Config):
        """
        Initialize the filter with configuration settings.
        
        Args:
            cfg (Config): Configuration object containing necessary parameters,
                          including the data_path for the input CSV.
        """
        super().__init__(cfg)

        self.filter_name: str = "lila_separation_single_label_filtering"
        self.data_path = cfg["data_path"]

    def run(self):
        """
        Execute the filtering process by copying the input CSV to the filter table directory.
        
        The method creates the necessary directory structure and copies the CSV file
        containing UUIDs of single-label images to be used in the subsequent steps.
        """
        filter_table_folder = os.path.join(
            self.tools_path, self.filter_name, "filter_table"
        )
        os.makedirs(filter_table_folder, exist_ok=True)
        filter_table_folder += "/table.csv"

        shutil.copyfile(self.data_path, filter_table_folder)


@SchedulerRegister("lila_separation_single_label_filtering")
class LilaSeparationSingleLabelFilteringScheduleCreation(DefaultScheduler):
    """
    Scheduler class for the single-label filtering process.
    
    This class creates a schedule for distributing the filtering work across multiple
    workers. It inherits from DefaultScheduler, which handles the standard scheduling
    logic of partitioning the data by server_name and partition_id.
    
    Attributes:
        filter_name (str): The name of the filter used for registration and folder creation.
    """
    def __init__(self, cfg: Config):
        """
        Initialize the scheduler with configuration settings.
        
        Args:
            cfg (Config): Configuration object containing parameters needed for scheduling.
        """
        super().__init__(cfg)

        self.filter_name: str = "lila_separation_single_label_filtering"


@RunnerRegister("lila_separation_single_label_filtering")
class LilaSeparationSingleLabelFilteringRunner(FilterRunnerTool):
    """
    Runner class that performs the actual filtering of images based on their UUIDs.
    
    This class implements the execution logic for filtering out images that don't have
    a single label. It reads the schedule created by the scheduler and processes the
    dataset to keep only single-label images based on the UUIDs in the filter table.
    
    Attributes:
        data_scheme (List[str]): The column schema for the filter data.
        filter_name (str): The name of the filter used for registration and folder creation.
    """
    def __init__(self, cfg: Config):
        """
        Initialize the runner with configuration settings.
        
        Args:
            cfg (Config): Configuration object containing parameters needed for execution.
        """
        super().__init__(cfg)
        self.data_scheme: List[str] = ["uuid", "server_name", "partition_id"]

        self.filter_name: str = "lila_separation_single_label_filtering"
