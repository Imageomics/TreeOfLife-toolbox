import argparse
import os
import pprint

from TreeOfLife_toolbox.main.config import Config
from TreeOfLife_toolbox.main.utils import init_logger
from TreeOfLife_toolbox.transfer_and_type_change.classes import ScheduleCreation

if __name__ == "__main__":
    config_path = os.environ.get("CONFIG_PATH")
    if config_path is None:
        raise ValueError("CONFIG_PATH not set")

    config = Config.from_path(config_path, "tools")
    logger = init_logger(__name__)

    parser = argparse.ArgumentParser(description='Running step of the Tool')
    parser.add_argument(
        "scheduler_name",
        metavar="scheduler_name",
        type=str,
        help="the name of the tool that is intended to be used",
    )
    parser.add_argument("seq_id", metavar="seq_id", type=int,
                        help="the name of the tool that is intended to be used")
    _args = parser.parse_args()
    logger.info(pprint.pformat(_args))
    seq_id = _args.seq_id

    tool_filter = ScheduleCreation(config, seq_id)

    logger.info("Starting scheduler")
    tool_filter.run()

    logger.info("completed scheduler")
