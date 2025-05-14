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

    parser = argparse.ArgumentParser(description="Filtering step of the Tool")
    parser.add_argument(
        "filter_name",
        metavar="filter_name",
        type=str,
        help="the name of the tool that is intended to be used",
    )
    _args = parser.parse_args()
    tool_name = _args.filter_name

    assert tool_name in ToolsRegistryBase.TOOLS_REGISTRY.keys(), ValueError(
        "unknown filter"
    )

    config = Config.from_path(config_path, "tools")
    checkpoint = Checkpoint.from_path(
        os.path.join(
            config.get_folder("tools_folder"), tool_name, "tool_checkpoint.yaml"
        ),
        {"filtering_scheduled": True, "filtering_completed": False},
    )
    logger = init_logger(__name__)
    checkpoint["filtering_scheduled"] = False

    tool_filter = ToolsRegistryBase.TOOLS_REGISTRY[tool_name]["filter"](config)

    if not checkpoint.get("filtering_completed", False):
        logger.info("Starting filter")
        tool_filter.run()

        logger.info("completed filtering")

        checkpoint["filtering_completed"] = True
    else:
        logger.info("Filtering was already completed")
