import os

import pandas as pd

from src.tol2webdataset.checkpoint import Checkpoint
from src.tol2webdataset.config import Config
from src.tol2webdataset.runner import Runner
from src.tol2webdataset.utils import init_logger

if __name__ == "__main__":
    config_path = os.environ.get("CONFIG_PATH")
    if config_path is None:
        raise ValueError("CONFIG_PATH not set")

    config = Config.from_path(config_path, "tol2webdataset")
    logger = init_logger(__name__)

    tool_folder = config.get_folder("path_to_output_folder")
    checkpoint = Checkpoint.from_path(os.path.join(tool_folder, "t2w_checkpoint.yaml"),
                                      {"completed": False, "scheduling_completed": False})

    if not checkpoint.get("scheduling_completed", False):
        logger.error("Scheduling wasn't complete, can't perform work")
        exit(1)

    schedule_df = pd.read_csv(os.path.join(tool_folder, "schedule.csv"))
    verification_df = Runner.load_table(config.get_folder("verification_folder"),
                                        ["shard_id"])

    outer_join = schedule_df.merge(verification_df, how='outer', indicator=True, on=["shard_id"])
    left = outer_join[(outer_join["_merge"] == 'left_only')].drop('_merge', axis=1)

    if len(left) == 0:
        checkpoint["completed"] = True

        logger.info("Tool completed its job")
    else:
        logger.info(f"Tool needs more time, left to complete: {len(left)}")
