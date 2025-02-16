import argparse
import os

from DD_tools.main.config import Config
from DD_tools.main.utils import init_logger
from DD_tools.transfer_and_type_change.classes import Runner

if __name__ == "__main__":
    config_path = os.environ.get("CONFIG_PATH")
    if config_path is None:
        raise ValueError("CONFIG_PATH not set")

    config = Config.from_path(config_path, "tools")
    logger = init_logger(__name__)

    parser = argparse.ArgumentParser(description='Running step of the Tool')
    parser.add_argument("seq_id", metavar="seq_id", type=int,
                        help="the name of the tool that is intended to be used")
    _args = parser.parse_args()
    seq_id = _args.seq_id

    tool_filter = Runner(config, seq_id)

    logger.info("Starting runner")
    tool_filter.run()

    logger.info("completed runner")
