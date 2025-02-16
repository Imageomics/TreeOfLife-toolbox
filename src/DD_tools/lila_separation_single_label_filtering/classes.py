import os
import shutil
from typing import List

from DD_tools.main.config import Config
from DD_tools.main.filters import FilterRegister, SparkFilterToolBase
from DD_tools.main.runners import FilterRunnerTool, RunnerRegister
from DD_tools.main.schedulers import DefaultScheduler, SchedulerRegister


@FilterRegister("lila_separation_single_label_filtering")
class LilaSeparationSingleLabelFilteringFilter(SparkFilterToolBase):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.filter_name: str = "lila_separation_single_label_filtering"
        self.data_path = "/fs/scratch/PAS2136/gbif/processed/lilabc/temp/tools/lila_separation_filtering/filter_table/table.csv"

    def run(self):
        filter_table_folder = os.path.join(
            self.tools_path, self.filter_name, "filter_table"
        )
        os.makedirs(filter_table_folder, exist_ok=True)
        filter_table_folder += "/table.csv"

        shutil.copyfile(self.data_path, filter_table_folder)


@SchedulerRegister("lila_separation_single_label_filtering")
class LilaSeparationSingleLabelFilteringScheduleCreation(DefaultScheduler):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.filter_name: str = "lila_separation_single_label_filtering"


@RunnerRegister("lila_separation_single_label_filtering")
class LilaSeparationSingleLabelFilteringRunner(FilterRunnerTool):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.data_scheme: List[str] = ["uuid", "server_name", "partition_id"]

        self.filter_name: str = "lila_separation_single_label_filtering"
