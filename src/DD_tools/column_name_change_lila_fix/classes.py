import os
from typing import List

import pandas as pd

from DD_tools.main.config import Config
from DD_tools.main.filters import PythonFilterToolBase, FilterRegister
from DD_tools.main.runners import MPIRunnerTool, RunnerRegister
from DD_tools.main.schedulers import DefaultScheduler, SchedulerRegister


@FilterRegister("column_name_change_lila_fix")
class ColumnNameChangeLilaFixFilter(PythonFilterToolBase):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.filter_name: str = "column_name_change_lila_fix"

    def run(self):
        uuid_table_df = pd.read_csv(self.config["uuid_table_path"], low_memory=False)
        uuid_table_df = uuid_table_df[
            uuid_table_df["server"] == "storage.googleapis.com"
            ][["path"]].drop_duplicates()

        uuid_table_df.to_csv(
            os.path.join(
                self.tools_path, self.filter_name, "filter_table", "table.csv"
            ),
            index=False,
        )


@SchedulerRegister("column_name_change_lila_fix")
class ColumnNameChangeLilaFixScheduleCreation(DefaultScheduler):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.filter_name: str = "column_name_change_lila_fix"
        self.scheme = ["path"]


@RunnerRegister("column_name_change_lila_fix")
class ColumnNameChangeLilaFixRunner(MPIRunnerTool):
    def __init__(self, cfg: Config, name_mapping=None):
        super().__init__(cfg)

        if name_mapping is None:
            name_mapping = {"uuid_y": "uuid", "source_id_y": "source_id"}

        self.filter_name: str = "column_name_change_lila_fix"
        self.data_scheme: List[str] = ["path"]
        self.verification_scheme: List[str] = ["path"]
        self.total_time = 150
        self.save_path_folder = "/fs/scratch/PAS2136/gbif/processed/lilabc/name_fix/server=storage.googleapis.com"

        self.name_mapping = name_mapping

    def apply_filter(self, filtering_df: pd.DataFrame, file_path: str) -> int:
        self.is_enough_time()

        if not os.path.exists(file_path):
            self.logger.info(f"Path doesn't exists: {file_path}")
            return 0

        renamed_parquet = pd.read_parquet(file_path)

        self.is_enough_time()

        renamed_parquet = renamed_parquet.rename(columns=self.name_mapping)
        save_path = os.path.join(self.save_path_folder, os.path.basename(file_path))
        os.makedirs(self.save_path_folder, exist_ok=True)

        renamed_parquet.to_parquet(
            save_path, index=False, compression="zstd", compression_level=3
        )

        return len(renamed_parquet)

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
