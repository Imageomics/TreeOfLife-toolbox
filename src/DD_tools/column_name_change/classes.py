import os
from typing import List

import pandas as pd

from DD_tools.main.config import Config
from DD_tools.main.filters import PythonFilterToolBase, FilterRegister
from DD_tools.main.runners import MPIRunnerTool, RunnerRegister
from DD_tools.main.schedulers import DefaultScheduler, SchedulerRegister


@FilterRegister("column_name_change")
class ColumnNameChangeFilter(PythonFilterToolBase):

    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.filter_name: str = "column_name_change"


@SchedulerRegister("column_name_change")
class ColumnNameChangeScheduleCreation(DefaultScheduler):

    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.filter_name: str = "column_name_change"


@RunnerRegister("column_name_change")
class ColumnNameChangeRunner(MPIRunnerTool):

    def __init__(self, cfg: Config, name_mapping=None):
        super().__init__(cfg)

        if name_mapping is None:
            name_mapping = {"gbif_id": "source_id"}

        self.filter_name: str = "column_name_change"
        self.data_scheme: List[str] = ["server_name", "partition_id"]
        self.verification_scheme: List[str] = ["server_name", "partition_id"]
        self.total_time = 150

        self.name_mapping = name_mapping

    def apply_filter(self, filtering_df: pd.DataFrame, server_name: str, partition_id: int) -> int:
        self.is_enough_time()

        parquet_path = os.path.join(
            self.downloaded_images_path,
            f"server_name={server_name}",
            f"partition_id={partition_id}",
            "successes.parquet"
        )

        if not os.path.exists(parquet_path):
            self.logger.info(f"Path doesn't exists: {parquet_path}")
            return 0

        renamed_parquet = pd.read_parquet(parquet_path)

        self.is_enough_time()

        renamed_parquet = renamed_parquet.rename(columns=self.name_mapping)

        renamed_parquet.to_parquet(parquet_path, index=False, compression="zstd", compression_level=3)

        return len(renamed_parquet)
