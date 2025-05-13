import glob
import os
import time
from functools import partial
from typing import List, TextIO, Optional

import pandas as pd

from TreeOfLife_toolbox.main.config import Config
from TreeOfLife_toolbox.main.registry import ToolsBase, ToolsRegistryBase

RunnerRegister = partial(ToolsRegistryBase.register, "runner")


class RunnerToolBase(ToolsBase):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.filter_family = "runner"


class MPIRunnerTool(RunnerToolBase):
    def __init__(self, cfg: Config):
        import mpi4py.MPI as MPI

        super().__init__(cfg)

        self.filter_folder: Optional[str] = None
        self.filter_table_folder: Optional[str] = None
        self.verification_folder: Optional[str] = None
        self.verification_IO: Optional[TextIO] = None

        self.data_scheme: Optional[List[str]] = None
        self.verification_scheme: Optional[List[str]] = None

        self.mpi_comm: MPI.Intracomm = MPI.COMM_WORLD
        self.mpi_rank: int = self.mpi_comm.rank
        self.total_time: Optional[int] = None

    def is_enough_time(self):
        assert self.total_time is not None, ValueError("total_time is not set")
        if time.time() > int(os.getenv("SLURM_JOB_END_TIME", 0)) - self.total_time:
            raise TimeoutError("Not enough time")

    @staticmethod
    def load_table(folder: str, columns: List[str] = None) -> pd.DataFrame:
        all_files = glob.glob(os.path.join(folder, "*.csv"))
        if len(all_files) == 0:
            assert columns is not None, ValueError(
                "No files found and columns are not defined"
            )

            return pd.DataFrame(columns=columns)
        return pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

    @staticmethod
    def get_csv_writer(path: str, scheme: List[str]) -> TextIO:
        if not os.path.exists(path):
            file = open(path, "w")
            print(",".join(scheme), file=file, flush=True)
        else:
            file = open(path, "a")
        return file

    def ensure_folders_created(self):
        assert self.filter_name is not None, ValueError("filter name is not set")
        assert self.verification_scheme is not None, ValueError(
            "verification scheme is not set"
        )

        self.filter_folder = os.path.join(self.tools_path, self.filter_name)
        self.filter_table_folder = os.path.join(self.filter_folder, "filter_table")
        self.verification_folder = os.path.join(
            self.tools_path, self.filter_name, "verification"
        )

        os.makedirs(self.verification_folder, exist_ok=True)

    def get_schedule(self):
        schedule_df = pd.read_csv(os.path.join(self.filter_folder, "schedule.csv"))
        schedule_df = schedule_df.query(f"rank == {self.mpi_rank}")
        verification_df = self.load_table(
            self.verification_folder, self.verification_scheme
        )
        outer_join = schedule_df.merge(
            verification_df, how="outer", indicator=True, on=self.verification_scheme
        )
        return outer_join[(outer_join["_merge"] == "left_only")].drop("_merge", axis=1)

    def get_remaining_table(
            self, schedule: pd.DataFrame
    ) -> pd.api.typing.DataFrameGroupBy:
        assert self.data_scheme is not None, ValueError("data scheme is not set")

        df = self.load_table(self.filter_table_folder)
        df = df.merge(schedule, how="right", on=self.verification_scheme)
        df = df[self.data_scheme]

        return df.groupby(self.verification_scheme, group_keys=True)

    def apply_filter(
            self, filtering_df: pd.DataFrame, server_name: str, partition_id: str
    ) -> int:
        raise NotImplementedError()

    def runner_fn(self, df_local: pd.DataFrame) -> int:
        filtering_df = df_local.reset_index(drop=True)
        server_name = filtering_df.iloc[0]["server_name"]
        partition_id = filtering_df.iloc[0]["partition_id"]
        try:
            filtered_parquet_length = self.apply_filter(
                filtering_df, server_name, partition_id
            )
        except NotImplementedError:
            raise NotImplementedError("Filter function wasn't implemented")
        except Exception as e:
            self.logger.exception(e)
            self.logger.error(f"Error occurred: {e}")
            return 0
        else:
            print(f"{server_name},{partition_id}", end="\n", file=self.verification_IO)
            self.logger.debug(
                f"Completed filtering: {server_name}/{partition_id} with {filtered_parquet_length}"
            )
            return 1

    def run(self):
        self.ensure_folders_created()

        schedule = self.get_schedule()
        self.mpi_comm.Barrier()
        if len(schedule) == 0:
            self.logger.error(f"Schedule not found or empty for rank {self.mpi_rank}")
            exit(0)

        self.verification_IO = self.get_csv_writer(
            f"{self.verification_folder}/{str(self.mpi_rank).zfill(4)}.csv",
            self.verification_scheme,
        )

        remaining_table = self.get_remaining_table(schedule)

        remaining_table.apply(self.runner_fn)

    def __del__(self):
        if self.verification_IO is not None:
            self.verification_IO.close()


class FilterRunnerTool(MPIRunnerTool):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.data_scheme: List[str] = [
            "uuid",
            "source_id",
            "server_name",
            "partition_id",
        ]
        self.verification_scheme: List[str] = ["server_name", "partition_id"]
        self.total_time = 150

    def apply_filter(
            self, filtering_df: pd.DataFrame, server_name: str, partition_id: str
    ) -> int:
        self.is_enough_time()

        parquet_path = os.path.join(
            self.downloaded_images_path,
            f"server_name={server_name}",
            f"partition_id={partition_id}",
            "successes.parquet",
        )

        if not os.path.exists(parquet_path):
            self.logger.info(f"Path doesn't exists: {server_name}/{partition_id}")
            return 0

        filtered_parquet = pd.read_parquet(
            parquet_path, filters=[("uuid", "not in", filtering_df["uuid"])]
        )

        self.is_enough_time()

        if len(filtered_parquet) == 0:
            self.logger.info(f"Fully filtered out: {server_name}/{partition_id}")

        filtered_parquet.to_parquet(
            parquet_path, index=False, compression="zstd", compression_level=3
        )

        return len(filtered_parquet)
