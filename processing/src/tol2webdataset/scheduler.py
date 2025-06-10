import glob
import os
import re

import pandas as pd

from src.tol2webdataset.utils import init_logger
from src.tol2webdataset.config import Config
from src.tol2webdataset.checkpoint import Checkpoint


class Scheduler:
    def __init__(self,
                 metadata_path,
                 schedule_file_path,
                 num_nodes,
                 num_workers_per_node,
                 ):
        self.logger = init_logger(__name__)

        self.metadata_path = metadata_path
        self.schedule_file_path = schedule_file_path
        self.num_workers_per_node = num_workers_per_node
        self.num_nodes = num_nodes

    def run(self):
        all_shard_ids = [int(re.search(r".*/shard_id=(\d+)", folder).group(1))
                         for folder in glob.glob(self.metadata_path + "/shard_id=*")]

        schedule_df = pd.DataFrame(all_shard_ids, columns=["shard_id"])
        schedule_df["rank"] = schedule_df.index % (self.num_nodes * self.num_workers_per_node)

        schedule_df.to_csv(self.schedule_file_path, header=True, index=False)


if __name__ == "__main__":
    config_path = os.environ.get("CONFIG_PATH")
    if config_path is None:
        raise ValueError("CONFIG_PATH not set")

    config = Config.from_path(config_path, "tol2webdataset")
    checkpoint = Checkpoint.from_path(os.path.join(config.get_folder("path_to_output_folder"), "t2w_checkpoint.yaml"),
                                      {"filtering_completed": False, "scheduling_completed": False})
    logger = init_logger(__name__)

    if not checkpoint.get("filtering_completed", False):
        logger.error("Filtering wasn't complete, can't create schedule")
        exit(1)

    WD_scheduler = Scheduler(config.get_folder("metadata_folder"),
                             os.path.join(config.get_folder("path_to_output_folder"), "schedule.csv"),
                             config["t2w_parameters"]["max_nodes_per_runner"],
                             config["t2w_parameters"]["workers_per_node"],
                             )

    logger.info("Starting scheduler")
    WD_scheduler.run()

    logger.info("Completed scheduler")
    checkpoint["scheduling_completed"] = True
