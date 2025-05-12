import argparse
import os

import pandas as pd

from TreeOfLife_toolbox.main.checkpoint import Checkpoint
from TreeOfLife_toolbox.main.config import Config
from TreeOfLife_toolbox.main.registry import ToolsRegistryBase
from TreeOfLife_toolbox.main.runners import MPIRunnerTool
from TreeOfLife_toolbox.main.utils import init_logger

if __name__ == "__main__":
    config_path = os.environ.get("CONFIG_PATH")
    if config_path is None:
        raise ValueError("CONFIG_PATH not set")

    config = Config.from_path(config_path, "tools")
    logger = init_logger(__name__)

    parser = argparse.ArgumentParser(description="Running step of the Tool")
    parser.add_argument(
        "runner_name",
        metavar="runner_name",
        type=str,
        help="the name of the tool that is intended to be used",
    )
    _args = parser.parse_args()
    tool_name = _args.runner_name

    assert tool_name in ToolsRegistryBase.TOOLS_REGISTRY.keys(), ValueError(
        "unknown runner"
    )

    tool_folder = os.path.join(config.get_folder("tools_folder"), tool_name)
    checkpoint = Checkpoint.from_path(
        os.path.join(tool_folder, "tool_checkpoint.yaml"), {"completed": False}
    )
    schedule_df = pd.read_csv(os.path.join(tool_folder, "schedule.csv"))
    runner = ToolsRegistryBase.TOOLS_REGISTRY[tool_name]["runner"](config)
    verification_columns = runner.verification_scheme
    verification_df = MPIRunnerTool.load_table(
        os.path.join(tool_folder, "verification"), verification_columns
    )

    outer_join = schedule_df.merge(
        verification_df, how="outer", indicator=True, on=verification_columns
    )
    left = outer_join[(outer_join["_merge"] == "left_only")].drop("_merge", axis=1)

    if len(left) == 0:
        checkpoint["completed"] = True

        logger.info("Tool completed its job")
    else:
        logger.info(f"Tool needs more time, left to complete: {len(left)}")
