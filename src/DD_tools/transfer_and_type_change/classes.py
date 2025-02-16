import glob
import os
from typing import List

import pandas as pd

from DD_tools.main.config import Config
from DD_tools.main.filters import PythonFilterToolBase
from DD_tools.main.runners import MPIRunnerTool
from DD_tools.main.schedulers import DefaultScheduler


class Filter(PythonFilterToolBase):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.filter_name: str = "transfer_and_type_change"

    def run(self):
        pass


class ScheduleCreation(DefaultScheduler):
    def __init__(self, cfg: Config, seq_id: int):
        super().__init__(cfg)

        self.filter_name: str = "transfer_and_type_change"
        self.seq_id = seq_id

    def run(self):
        assert self.filter_name is not None, ValueError("filter name is not set")

        filter_folder = os.path.join(
            self.tools_path, self.filter_name, str(self.seq_id).zfill(4)
        )
        filter_table_folder = os.path.join(filter_folder, "filter_table")

        all_files = glob.glob(os.path.join(filter_table_folder, "*.csv"))
        df: pd.DataFrame = pd.concat(
            (pd.read_csv(f) for f in all_files), ignore_index=True
        )
        df = df[["source", "server", "file_name"]]
        df = df.drop_duplicates(subset=["source", "server", "file_name"]).reset_index(
            drop=True
        )
        df["rank"] = df.index % self.total_workers

        df.to_csv(os.path.join(filter_folder, "schedule.csv"), header=True, index=False)


class Runner(MPIRunnerTool):
    def __init__(self, cfg: Config, seq_id: int):
        super().__init__(cfg)

        self.filter_name: str = "transfer_and_type_change"
        self.data_scheme: List[str] = ["source", "server", "file_name"]
        self.verification_scheme: List[str] = ["source", "server", "file_name"]
        self.total_time = 150
        self.seq_id = seq_id

        self.src_path = self.config["src_path"]
        self.dst_path = self.config["dst_path"]

    def ensure_folders_created(self):
        assert self.filter_name is not None, ValueError("filter name is not set")
        assert self.verification_scheme is not None, ValueError(
            "verification scheme is not set"
        )

        self.filter_folder = os.path.join(
            self.tools_path, self.filter_name, str(self.seq_id).zfill(4)
        )
        self.filter_table_folder = os.path.join(self.filter_folder, "filter_table")
        self.verification_folder = os.path.join(self.filter_folder, "verification")

        os.makedirs(self.verification_folder, exist_ok=True)

    def apply_filter_different(
            self, filtering_df: pd.DataFrame, source: str, server: str, file_name: str
    ) -> int:
        self.is_enough_time()

        src_path = os.path.join(
            self.src_path, f"source={source}", "data", f"server={server}", file_name
        )
        dst_path = os.path.join(
            self.dst_path, f"source={source}", f"server={server}", file_name
        )
        os.makedirs(
            os.path.join(
                self.dst_path,
                f"source={source}",
                f"server={server}",
            ),
            exist_ok=True,
        )

        if not os.path.exists(src_path):
            self.logger.info(f"Path doesn't exists: {src_path}")
            return 0

        renamed_parquet = pd.read_parquet(src_path)

        self.is_enough_time()

        renamed_parquet = renamed_parquet.astype({"source_id": "string"})

        renamed_parquet.to_parquet(
            dst_path, index=False, compression="zstd", compression_level=3
        )

        os.remove(src_path)

        return len(renamed_parquet)

    def runner_fn(self, df_local: pd.DataFrame) -> int:
        filtering_df = df_local.reset_index(drop=True)
        source = filtering_df.iloc[0]["source"]
        server = filtering_df.iloc[0]["server"]
        file_name = filtering_df.iloc[0]["file_name"]
        try:
            filtered_parquet_length = self.apply_filter_different(
                filtering_df, source, server, file_name
            )
        except NotImplementedError:
            raise NotImplementedError("Filter function wasn't implemented")
        except Exception as e:
            self.logger.exception(e)
            self.logger.error(f"Error occurred: {e}")
            return 0
        else:
            print(f"{source},{server},{file_name}", end="\n", file=self.verification_IO)
            self.logger.debug(
                f"Completed filtering: {source}/{server}/{file_name} with {filtered_parquet_length}"
            )
            return 1
