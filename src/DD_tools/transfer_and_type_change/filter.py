import os

from DD_tools.main.config import Config
from DD_tools.main.utils import init_logger
from DD_tools.transfer_and_type_change.classes import Filter

if __name__ == "__main__":
    config_path = os.environ.get("CONFIG_PATH")
    if config_path is None:
        raise ValueError("CONFIG_PATH not set")

    config = Config.from_path(config_path, "tools")
    logger = init_logger(__name__)

    tool_filter = Filter(config)

    logger.info("Starting filter")
    tool_filter.run()

    logger.info("completed filtering")

    tool_filter = None
