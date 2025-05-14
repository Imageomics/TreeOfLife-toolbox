import argparse
import os

from TreeOfLife_toolbox.main.checkpoint import Checkpoint
from TreeOfLife_toolbox.main.config import Config
from TreeOfLife_toolbox.main.registry import ToolsRegistryBase
from TreeOfLife_toolbox.main.utils import init_logger

if __name__ == "__main__":
    config_path = os.environ.get("CONFIG_PATH")
    if config_path is None:
        raise ValueError("CONFIG_PATH not set")

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

    config = Config.from_path(config_path, "tools")
    checkpoint = Checkpoint.from_path(
        os.path.join(
            config.get_folder("tools_folder"), tool_name, "tool_checkpoint.yaml"
        ),
        {"scheduling_completed": False},
    )
    logger = init_logger(__name__)

    if not checkpoint.get("scheduling_completed", False):
        logger.error("Scheduling wasn't complete, can't perform work")
        exit(1)

    tool_runner = ToolsRegistryBase.TOOLS_REGISTRY[tool_name]["runner"](config)

    logger.info("Starting runner")
    tool_runner.run()

    logger.info("completed runner")
