import os
from typing import List

import pandas as pd

from DD_tools.main.config import Config
from DD_tools.main.filters import FilterRegister, SparkFilterToolBase
from DD_tools.main.runners import MPIRunnerTool, RunnerRegister
from DD_tools.main.schedulers import DefaultScheduler, SchedulerRegister


@FilterRegister("mam_ansp_fix")
class MamAnspFixFilter(SparkFilterToolBase):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.filter_name: str = "mam_ansp_fix"

    def run(self):
        uuid_table_df = pd.read_csv(self.config["uuid_table_path"], low_memory=False)
        uuid_table_df = uuid_table_df[uuid_table_df["server"] == "mam.ansp.org"][
            ["path"]
        ].drop_duplicates()

        uuid_table_df.to_csv(
            os.path.join(
                self.tools_path, self.filter_name, "filter_table", "table.csv"
            ),
            index=False,
        )


@SchedulerRegister("mam_ansp_fix")
class MamAnspFixScheduleCreation(DefaultScheduler):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.filter_name: str = "mam_ansp_fix"
        self.scheme = ["path"]


@RunnerRegister("mam_ansp_fix")
class MamAnspFixRunner(MPIRunnerTool):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.filter_name: str = "mam_ansp_fix"
        self.data_scheme: List[str] = ["path"]
        self.verification_scheme: List[str] = ["path"]
        self.total_time = 150
        self.save_path_folder = (
            "/fs/scratch/PAS2136/gbif/processed/mam_ansp_fix/server=mam.ansp.org"
        )

    def apply_filter(self, filtering_df: pd.DataFrame, file_path: str) -> int:
        self.is_enough_time()

        if not os.path.exists(file_path):
            self.logger.info(f"Path doesn't exists: {file_path}")
            return 0

        filtered_parquet = pd.read_parquet(file_path)

        self.is_enough_time()

        if len(filtered_parquet) == 0:
            self.logger.info(f"Fully filtered out: {file_path}")

        filtered_parquet = filtered_parquet.drop_duplicates("uuid")
        save_path = os.path.join(self.save_path_folder, os.path.basename(file_path))
        os.makedirs(self.save_path_folder, exist_ok=True)

        filtered_parquet.to_parquet(
            save_path, index=False, compression="zstd", compression_level=3
        )

        return len(filtered_parquet)

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
