import argparse
import os
from logging import Logger
from typing import Dict, List, Optional, TextIO, Tuple

import pandas as pd
from attr import Factory, define, field

from DD_tools.main.checkpoint import Checkpoint
from DD_tools.main.config import Config
from DD_tools.main.registry import ToolsRegistryBase
from DD_tools.main.utils import (
    init_logger,
    ensure_created,
    truncate_paths,
    submit_job,
    preprocess_dep_ids,
)


@define
class Tools:
    config: Config
    tool_name: str

    logger: Logger = field(default=Factory(lambda: init_logger(__name__)))

    tool_folder: Optional[str] = None
    tool_job_history_path: Optional[str] = None
    tool_checkpoint_path: Optional[str] = None
    checkpoint_scheme = {
        "filtering_scheduled": False,
        "filtering_completed": False,
        "scheduling_scheduled": False,
        "scheduling_completed": False,
        "completed": False,
    }

    tool_checkpoint: Optional[Checkpoint] = None
    _checkpoint_override: Optional[Dict[str, bool]] = None
    tool_job_history: Optional[List[int]] = None
    tool_job_history_io: Optional[TextIO] = None

    @classmethod
    def from_path(
            cls,
            path: str,
            tool_name: str,
            checkpoint_override: Optional[Dict[str, bool]] = None,
            tool_name_override: Optional[bool] = False,
    ) -> "Tools":
        if (
                not tool_name_override
                and tool_name not in ToolsRegistryBase.TOOLS_REGISTRY.keys()
        ):
            raise ValueError("unknown tool name")

        return cls(
            config=Config.from_path(path, "tools"),
            tool_name=tool_name,
            checkpoint_override=checkpoint_override,
        )

    def __attrs_post_init__(self):
        # noinspection PyTypeChecker
        self.tool_folder: str = os.path.join(
            self.config.get_folder("tools_folder"), self.tool_name
        )
        self.tool_job_history_path: str = os.path.join(
            self.tool_folder, "job_history.csv"
        )
        self.tool_checkpoint_path: str = os.path.join(
            self.tool_folder, "tool_checkpoint.yaml"
        )

        self.__init_environment()
        self.__init_file_structure()

    def __init_environment(self) -> None:
        os.environ["CONFIG_PATH"] = self.config.config_path

        os.environ["ACCOUNT"] = self.config["account"]
        os.environ["PATH_TO_INPUT"] = self.config["path_to_input"]

        os.environ["PATH_TO_OUTPUT"] = self.config["path_to_output_folder"]
        for output_folder, output_path in self.config.folder_structure.items():
            os.environ["OUTPUT_" + output_folder.upper()] = output_path
        os.environ["OUTPUT_TOOLS_LOGS_FOLDER"] = os.path.join(self.tool_folder, "logs")

        for downloader_var, downloader_value in self.config["tools_parameters"].items():
            os.environ["TOOLS_" + downloader_var.upper()] = str(downloader_value)

        self.logger.info("Environment initialized")

    def __init_file_structure(self):
        ensure_created(
            [
                self.tool_folder,
                os.path.join(self.tool_folder, "filter_table"),
                os.path.join(self.tool_folder, "verification"),
                os.path.join(self.tool_folder, "logs"),
            ]
        )

        self.tool_checkpoint = Checkpoint.from_path(
            self.tool_checkpoint_path, self.checkpoint_scheme
        )
        if self._checkpoint_override is not None:
            for key, value in self._checkpoint_override.items():
                if key == "verification":
                    truncate_paths([os.path.join(self.tool_folder, "verification")])
                    self.tool_checkpoint["completed"] = False
                    continue
                if key not in self.checkpoint_scheme.keys():
                    raise KeyError("Unknown key for override in checkpoint")

                self.tool_checkpoint[key] = value

        self.tool_job_history, self.tool_job_history_io = self.__load_job_history()

    def __load_job_history(self) -> Tuple[List[int], TextIO]:
        job_ids = []

        if os.path.exists(self.tool_job_history_path):
            df = pd.read_csv(self.tool_job_history_path)
            job_ids = df["job_ids"].to_list()
        else:
            with open(self.tool_job_history_path, "w") as f:
                print("job_ids", file=f)

        job_io = open(self.tool_job_history_path, "a")

        return job_ids, job_io

    def __update_job_history(self, new_id: int) -> None:
        self.tool_job_history.append(new_id)
        print(new_id, file=self.tool_job_history_io)

    def __schedule_filtering(self) -> None:
        self.logger.info("Scheduling filtering script")
        job_id = submit_job(
            self.config.get_script("tools_submitter"),
            self.config.get_script("tools_filter_script"),
            self.tool_name,
            *preprocess_dep_ids(
                [self.tool_job_history[-1] if len(self.tool_job_history) != 0 else None]
            ),
            "--spark",
        )
        self.__update_job_history(job_id)
        self.tool_checkpoint["filtered"] = True
        self.logger.info("Scheduled filtering script")

    def __schedule_schedule_creation(self) -> None:
        self.logger.info("Scheduling schedule creation script")
        job_id = submit_job(
            self.config.get_script("tools_submitter"),
            self.config.get_script("tools_scheduling_script"),
            self.tool_name,
            *preprocess_dep_ids([self.tool_job_history[-1]]),
        )
        self.__update_job_history(job_id)
        self.tool_checkpoint["schedule_created"] = True
        self.logger.info("Scheduled schedule creation script")

    def __schedule_workers(self) -> None:
        self.logger.info("Scheduling workers script")

        for _ in range(self.config["tools_parameters"]["num_workers"]):
            job_id = submit_job(
                self.config.get_script("tools_submitter"),
                self.config.get_script("tools_worker_script"),
                self.tool_name,
                *preprocess_dep_ids([self.tool_job_history[-1]]),
            )
            self.__update_job_history(job_id)

        job_id = submit_job(
            self.config.get_script("tools_submitter"),
            self.config.get_script("tools_verification_script"),
            self.tool_name,
            *preprocess_dep_ids([self.tool_job_history[-1]]),
        )
        self.__update_job_history(job_id)

        self.logger.info("Scheduled workers script")

    def apply_tool(self):
        if not (
                self.tool_checkpoint.get("filtering_scheduled", False)
                or self.tool_checkpoint.get("filtering_completed", False)
        ):
            self.__schedule_filtering()
        else:
            self.logger.info(
                "Skipping filtering script: job is already scheduled or table has been already created"
            )

        if not (
                self.tool_checkpoint.get("schedule_scheduled", False)
                or self.tool_checkpoint.get("schedule_completed", False)
        ):
            self.__schedule_schedule_creation()
        else:
            self.logger.info(
                "Skipping schedule creation script: job is already scheduled or schedule has been already created"
            )

        if not self.tool_checkpoint.get("completed", False):
            self.__schedule_workers()
        else:
            self.logger.error("Tool completed its job")

    def __del__(self):
        if self.tool_job_history_io is not None:
            self.tool_job_history_io.close()


def main():
    parser = argparse.ArgumentParser(description="Tools")
    parser.add_argument(
        "config_path",
        metavar="config_path",
        type=str,
        help="the name of the tool that is intended to be used",
    )
    parser.add_argument(
        "tool_name",
        metavar="tool_name",
        type=str,
        help="the name of the tool that is intended to be used",
    )
    parser.add_argument(
        "--reset_filtering",
        action="store_true",
        help="Will reset filtering and scheduling steps",
    )
    parser.add_argument(
        "--reset_scheduling", action="store_true", help="Will reset scheduling step"
    )
    parser.add_argument(
        "--reset_runners",
        action="store_true",
        help="Will reset runners, making them to start over",
    )
    parser.add_argument(
        "--tool_name_override",
        action="store_true",
        help="Will override tool name check (allows for custom tool run)",
    )
    _args = parser.parse_args()

    config_path = _args.config_path
    tool_name = _args.tool_name
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
        state_override = {"verification": False, "completed": False}

    dd = Tools.from_path(
        config_path, tool_name, state_override, _args.tool_name_override
    )
    dd.apply_tool()


if __name__ == "__main__":
    main()
