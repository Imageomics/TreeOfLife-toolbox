import argparse
import os
from logging import Logger
from typing import Dict, List, Optional, TextIO, Tuple

import pandas as pd
from attr import Factory, define, field

from src.tol2webdataset.checkpoint import Checkpoint
from src.tol2webdataset.config import Config
from src.tol2webdataset.utils import (
    ensure_created,
    init_logger,
    preprocess_dep_ids,
    submit_job,
    truncate_paths, logger_folder_count,
)


@define
class Tol2WebdatasetConverter:
    config: Config

    logger: Logger = field(default=Factory(lambda: init_logger(__name__)))

    t2w_output_folder: Optional[str] = None
    t2w_job_history_path: Optional[str] = None
    t2w_checkpoint_path: Optional[str] = None
    checkpoint_scheme = {
        "filtering_scheduled": False,
        "filtering_completed": False,
        "scheduling_scheduled": False,
        "scheduling_completed": False,
        "completed": False
    }

    t2w_checkpoint: Optional[Checkpoint] = None
    _checkpoint_override: Optional[Dict[str, bool]] = None
    t2w_job_history: Optional[List[int]] = None
    t2w_job_history_io: Optional[TextIO] = None

    @classmethod
    def from_path(cls, path: str,
                  checkpoint_override: Optional[Dict[str, bool]] = None) -> "Tol2WebdatasetConverter":
        return cls(config=Config.from_path(path, "tol2webdataset"),
                   checkpoint_override=checkpoint_override)

    def __attrs_post_init__(self):
        # noinspection PyTypeChecker
        self.t2w_output_folder: str = self.config.get_folder("path_to_output_folder")
        self.t2w_job_history_path: str = os.path.join(self.t2w_output_folder, "job_history.csv")
        self.t2w_checkpoint_path: str = os.path.join(self.t2w_output_folder, "t2w_checkpoint.yaml")

        self.__init_environment()
        self.__init_file_structure()

    def __init_environment(self) -> None:
        os.environ["CONFIG_PATH"] = self.config.config_path

        os.environ["ACCOUNT"] = self.config["account"]
        os.environ["PATH_TO_SOURCE_TAXA"] = self.config["path_to_source_taxa"]
        os.environ["PATH_TO_IMAGE_DATA"] = self.config["path_to_image_data"]
        os.environ["PATH_TO_IMAGE_LOOKUP_TABLE"] = self.config["path_to_image_lookup_table"]

        os.environ["PATH_TO_OUTPUT"] = self.config["path_to_output_folder"]
        for output_folder, output_path in self.config.folder_structure.items():
            os.environ["OUTPUT_" + output_folder.upper()] = output_path
        os.environ["OUTPUT_T2W_LOGS_FOLDER"] = os.path.join(self.t2w_output_folder,
                                                            "logs")

        for downloader_var, downloader_value in self.config["t2w_parameters"].items():
            os.environ["T2W_" + downloader_var.upper()] = str(downloader_value)

        self.logger.info("Environment initialized")

    def __init_file_structure(self):
        ensure_created([
            self.t2w_output_folder,
            self.config.get_folder("metadata_folder"),
            self.config.get_folder("logs_folder"),
            self.config.get_folder("verification_folder"),
            self.config.get_folder("dataset_output_folder")
        ])

        self.t2w_checkpoint = Checkpoint.from_path(self.t2w_checkpoint_path, self.checkpoint_scheme)
        if self._checkpoint_override is not None:
            for key, value in self._checkpoint_override.items():
                if key == "verification":
                    truncate_paths([self.config.get_folder("verification_folder")])
                    self.t2w_checkpoint["completed"] = False
                    continue
                if key not in self.checkpoint_scheme.keys():
                    raise KeyError("Unknown key for override in checkpoint")

                self.t2w_checkpoint[key] = value

        self.t2w_job_history, self.t2w_job_history_io = self.__load_job_history()

    def __load_job_history(self) -> Tuple[List[int], TextIO]:
        job_ids = []

        if os.path.exists(self.t2w_job_history_path):
            df = pd.read_csv(self.t2w_job_history_path)
            job_ids = df["job_ids"].to_list()
        else:
            with open(self.t2w_job_history_path, "w") as f:
                print("job_ids", file=f)

        job_io = open(self.t2w_job_history_path, "a")

        return job_ids, job_io

    def __update_job_history(self, new_id: int) -> None:
        self.t2w_job_history.append(new_id)
        print(new_id, file=self.t2w_job_history_io)

    def __schedule_filtering(self) -> None:
        self.logger.info("Scheduling filtering script")
        job_id = submit_job(self.config.get_script("t2w_submitter"),
                            self.config.get_script("t2w_filter_script"),
                            0,
                            *preprocess_dep_ids(
                                [self.t2w_job_history[-1] if len(self.t2w_job_history) != 0 else None]),
                            "--spark")
        self.__update_job_history(job_id)
        self.t2w_checkpoint["filtering_scheduled"] = True
        self.logger.info("Scheduled filtering script")

    def __schedule_schedule_creation(self) -> None:
        self.logger.info("Scheduling schedule creation script")
        job_id = submit_job(self.config.get_script("t2w_submitter"),
                            self.config.get_script("t2w_scheduling_script"),
                            0,
                            *preprocess_dep_ids([self.t2w_job_history[-1]]))
        self.__update_job_history(job_id)
        self.t2w_checkpoint["scheduling_scheduled"] = True
        self.logger.info("Scheduled schedule creation script")

    def __schedule_workers(self) -> None:
        self.logger.info("Scheduling workers script")

        job_index = logger_folder_count(self.config.get_folder("logs_folder"))

        for _ in range(self.config["t2w_parameters"]["num_runners"]):
            job_id = submit_job(self.config.get_script("t2w_submitter"),
                                self.config.get_script("t2w_worker_script"),
                                str(job_index).zfill(4),
                                *preprocess_dep_ids([self.t2w_job_history[-1]]))
            self.__update_job_history(job_id)
            job_index += 1

        job_id = submit_job(self.config.get_script("t2w_submitter"),
                            self.config.get_script("t2w_verification_script"),
                            str(job_index).zfill(4),
                            *preprocess_dep_ids([self.t2w_job_history[-1]]))
        self.__update_job_history(job_id)

        self.logger.info("Scheduled workers script")

    def apply_tool(self):
        if not self.t2w_checkpoint.get("filtering_scheduled", False):
            self.__schedule_filtering()
        else:
            self.logger.info("Skipping filtering script: table already created")

        if not self.t2w_checkpoint.get("scheduling_scheduled", False):
            self.__schedule_schedule_creation()
        else:
            self.logger.info("Skipping schedule creation script: schedule already created")

        if not self.t2w_checkpoint.get("completed", False):
            self.__schedule_workers()
        else:
            self.logger.error("Tool completed its job")

    def __del__(self):
        if self.t2w_job_history_io is not None:
            self.t2w_job_history_io.close()


def main():
    parser = argparse.ArgumentParser(description='Tools')
    parser.add_argument("config_path", metavar="config_path", type=str,
                        help="the name of the tool that is intended to be used")
    parser.add_argument("--reset_filtering", action="store_true", help="Will reset filtering and scheduling steps")
    parser.add_argument("--reset_scheduling", action="store_true", help="Will reset scheduling step")
    parser.add_argument("--reset_runners", action="store_true", help="Will reset runners, making them to start over")

    _args = parser.parse_args()

    config_path = _args.config_path
    state_override = None
    if _args.reset_filtering:
        state_override = {
            "filtering_scheduled": False,
            "filtering_completed": False,
            "scheduling_scheduled": False,
            "scheduling_completed": False,
            "verification": False,
            "completed": False,
        }
    elif _args.reset_scheduling:
        state_override = {
            "scheduling_scheduled": False,
            "scheduling_completed": False,
            "completed": False,
        }
    if _args.reset_runners:
        state_override = {
            "verification": False,
            "completed": False
        }

    dd = Tol2WebdatasetConverter.from_path(config_path,
                                           state_override)
    dd.apply_tool()


if __name__ == "__main__":
    main()
