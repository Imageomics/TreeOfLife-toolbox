import os
from typing import List

import pandas as pd

from DD_tools.main.config import Config
from DD_tools.main.filters import PythonFilterToolBase, FilterRegister
from DD_tools.main.runners import MPIRunnerTool, RunnerRegister
from DD_tools.main.schedulers import DefaultScheduler, SchedulerRegister


@FilterRegister("eol_rename")
class EoLRenameFilter(PythonFilterToolBase):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.filter_name: str = "eol_rename"


@SchedulerRegister("eol_rename")
class EoLRenameScheduleCreation(DefaultScheduler):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.filter_name: str = "eol_rename"


@RunnerRegister("eol_rename")
class EoLRenameRunner(MPIRunnerTool):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.filter_name: str = "eol_rename"
        self.data_scheme: List[str] = ["server_name", "partition_id"]
        self.verification_scheme: List[str] = ["server_name", "partition_id"]
        self.total_time = 150

    def apply_filter(
            self, filtering_df: pd.DataFrame, server_name: str, partition_id: int
    ) -> int:
        self.is_enough_time()

        parquet_path = os.path.join(
            self.downloaded_images_path,
            f"server_name={server_name}",
            f"partition_id={partition_id}",
            "successes.parquet",
        )
        server_batch_path = os.path.join(
            self.config.get_folder("urls_folder"),
            f"server_name={server_name}",
            f"partition_id={partition_id}",
        )

        if not os.path.exists(parquet_path):
            self.logger.info(f"Path doesn't exists: {parquet_path}")
            return 0

        parquet = pd.read_parquet(parquet_path)
        server_batch = pd.read_parquet(
            server_batch_path, columns=["EOL content ID", "EOL page ID", "uuid"]
        )

        self.is_enough_time()

        parquet = parquet.merge(server_batch, on="uuid", how="left", validate="1:1")
        parquet["source_id"] = parquet["EOL content ID"] + "_" + parquet["EOL page ID"]

        parquet.to_parquet(
            parquet_path, index=False, compression="zstd", compression_level=3
        )

        return len(parquet)
