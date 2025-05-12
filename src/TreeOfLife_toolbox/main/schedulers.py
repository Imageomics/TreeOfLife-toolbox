import glob
import os
from functools import partial
from typing import List

import pandas as pd

from DD_tools.main.config import Config
from DD_tools.main.registry import ToolsBase, ToolsRegistryBase

SchedulerRegister = partial(ToolsRegistryBase.register, "scheduler")


class SchedulerToolBase(ToolsBase):

    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.filter_family = "scheduler"


class DefaultScheduler(SchedulerToolBase):

    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.scheme: List[str] = ["server_name", "partition_id"]

    def run(self):
        assert self.filter_name is not None, ValueError("filter name is not set")
        assert self.scheme is not None, ValueError("Scheme was not set")

        filter_folder = os.path.join(self.tools_path, self.filter_name)
        filter_table_folder = os.path.join(filter_folder, "filter_table")

        all_files = glob.glob(os.path.join(filter_table_folder, "*.csv"))
        df: pd.DataFrame = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
        df = df[self.scheme]
        df = df.drop_duplicates(subset=self.scheme).reset_index(drop=True)
        df["rank"] = df.index % self.total_workers

        df.to_csv(os.path.join(filter_folder, "schedule.csv"), header=True, index=False)
