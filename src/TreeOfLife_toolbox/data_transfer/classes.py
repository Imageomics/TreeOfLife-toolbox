import glob
import hashlib
import os
import shutil
import uuid
from typing import List, Tuple, Sequence

import numpy as np
import pandas as pd

from TreeOfLife_toolbox.main.config import Config
from TreeOfLife_toolbox.main.filters import PythonFilterToolBase, FilterRegister
from TreeOfLife_toolbox.main.runners import MPIRunnerTool, RunnerRegister
from TreeOfLife_toolbox.main.schedulers import DefaultScheduler, SchedulerRegister
from TreeOfLife_toolbox.main.utils import ensure_created


@FilterRegister("data_transfer")
class DataTransferFilter(PythonFilterToolBase):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.filter_name: str = "data_transfer"

    def get_all_paths_to_merge(self) -> pd.DataFrame:
        glob_wildcard = self.downloaded_images_path + "/*/*"
        server_name_regex = (
            rf"{self.downloaded_images_path}/server_name=(.*)/partition_id=.*"
        )
        basename_regex = (
            rf"{self.downloaded_images_path}/server_name=.*/partition_id=.*/(.*)"
        )
        os.makedirs(self.config["dst_image_folder"], exist_ok=True)
        os.makedirs(self.config["dst_error_folder"], exist_ok=True)

        src_paths = []

        for path in glob.glob(glob_wildcard):
            if not os.path.exists(os.path.join(path, "completed")):
                continue
            src_paths.append(os.path.join(path, "successes.parquet"))
            src_paths.append(os.path.join(path, "errors.parquet"))

        src_paths_df = pd.DataFrame(src_paths, columns=["src_path"])
        src_paths_df["server_name"] = src_paths_df["src_path"].str.extract(
            server_name_regex
        )
        src_paths_df["corrected_server_name"] = src_paths_df["server_name"].str.replace(
            "%3A", "_"
        )
        src_paths_df["basename"] = src_paths_df["src_path"].str.extract(basename_regex)
        src_paths_df["dst_base_folder"] = np.where(
            src_paths_df["basename"] == "successes.parquet",
            self.config["dst_image_folder"],
            self.config["dst_error_folder"],
        )
        src_paths_df["dst_basename"] = np.where(
            src_paths_df["basename"] == "successes.parquet", "data_", "errors_"
        )
        src_paths_df["uuid"] = src_paths_df.apply(lambda _: str(uuid.uuid4()), axis=1)
        src_paths_df["dst_path"] = (
            src_paths_df["dst_base_folder"]
            + "/server="
            + src_paths_df["corrected_server_name"]
            + "/"
            + src_paths_df["dst_basename"]
            + src_paths_df["uuid"]
            + ".parquet"
        )
        return src_paths_df[["src_path", "dst_path"]]


@SchedulerRegister("data_transfer")
class DataTransferScheduleCreation(DefaultScheduler):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.filter_name: str = "data_transfer"

    def run(self):
        assert self.filter_name is not None, ValueError("filter name is not set")

        filter_folder = os.path.join(self.tools_path, self.filter_name)
        filter_table_folder = os.path.join(filter_folder, "filter_table")

        all_files = glob.glob(os.path.join(filter_table_folder, "*.csv"))
        df: pd.DataFrame = pd.concat(
            (pd.read_csv(f) for f in all_files), ignore_index=True
        )

        df.to_csv(os.path.join(filter_folder, "schedule.csv"), header=True, index=False)


@RunnerRegister("data_transfer")
class DataTransferRunner(MPIRunnerTool):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.mpi_comm = None
        self.mpi_rank = 0

        self.filter_name: str = "data_transfer"
        self.verification_scheme = [
            "src_path",
            "dst_path",
            "hashsum_src",
            "hashsum_dst",
        ]
        self.total_time = 60

    def get_schedule(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.filter_folder, "schedule.csv"))

    def get_remaining_table(self, schedule: pd.DataFrame) -> pd.DataFrame:
        verification_df = self.load_table(
            self.verification_folder, self.verification_scheme
        )
        outer_join = schedule.merge(
            verification_df, how="outer", indicator=True, on=["src_path", "dst_path"]
        )
        left_to_copy = outer_join[(outer_join["_merge"] == "left_only")].drop(
            "_merge", axis=1
        )[["src_path", "dst_path"]]
        return left_to_copy

    @staticmethod
    def correct_server_name(server_names: List[str]) -> List[str]:
        for i, server in enumerate(server_names):
            server_names[i] = server.replace("%3A", "_")

        return server_names

    def ensure_all_servers_exists(
        self, all_files_df: pd.DataFrame, dst_paths: Sequence[str]
    ) -> None:
        server_name_regex = (
            rf"{self.downloaded_images_path}/server_name=(.*)/partition_id=.*"
        )

        server_names_series: pd.Series = all_files_df["src_path"].str.extract(
            server_name_regex, expand=False
        )
        server_names = (
            server_names_series.drop_duplicates().reset_index(drop=True).to_list()
        )
        server_names = self.correct_server_name(server_names)
        for dst_path in dst_paths:
            ensure_created(
                [os.path.join(dst_path, f"server={server}") for server in server_names]
            )

    @staticmethod
    def compute_hashsum(file_path: str, hashsum_alg) -> str:
        with open(file_path, "rb") as f:
            while True:
                data = f.read(131_072)
                if not data:
                    break
                hashsum_alg.update(data)
        return hashsum_alg.hexdigest()

    def copy_file(
        self, row: Tuple[pd.Index, pd.Series]
    ) -> Tuple[bool, str, str, str, str]:
        src_path = row[1]["src_path"]
        dst_path = row[1]["dst_path"]
        try:
            self.is_enough_time()

            hs_src_alg = hashlib.md5()
            hs_src_local = self.compute_hashsum(src_path, hs_src_alg)

            self.is_enough_time()

            shutil.copy(src_path, dst_path)

            self.is_enough_time()

            hs_dest_alg = hashlib.md5()
            hs_dest_local = self.compute_hashsum(dst_path, hs_dest_alg)
            return False, src_path, dst_path, hs_src_local, hs_dest_local
        except Exception as e:
            return True, src_path, str(e), "", ""

    def run(self):
        from mpi4py.futures import MPIPoolExecutor

        self.ensure_folders_created()

        schedule = self.get_schedule()
        if len(schedule) == 0:
            self.logger.error(f"Schedule not found or empty for rank {self.mpi_rank}")
            exit(0)

        remaining_table = self.get_remaining_table(schedule)
        self.ensure_all_servers_exists(
            remaining_table,
            [self.config["dst_image_folder"], self.config["dst_error_folder"]],
        )

        self.logger.info("Started copying")
        self.logger.info(f"{len(remaining_table)} files left to copy")
        with self.get_csv_writer(
            f"{self.verification_folder}/verification.csv", self.verification_scheme
        ) as verification_file:
            with MPIPoolExecutor() as executor:
                for is_error, src, dst, hs_src, hs_dest in executor.map(
                    self.copy_file, remaining_table.iterrows()
                ):
                    if is_error:
                        self.logger.error(f"Error {dst} for {src}")
                    else:
                        print(
                            src,
                            dst,
                            hs_src,
                            hs_dest,
                            sep=",",
                            file=verification_file,
                            flush=True,
                        )
                        self.logger.debug(f"Copied file {src} to {dst}")

        self.logger.info("Finished copying")
