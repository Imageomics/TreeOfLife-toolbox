import glob
import os
import time
from io import BytesIO
from typing import Optional, TextIO, List

import numpy as np
import pandas as pd
from mpi4py import MPI
import webdataset as wds
from PIL import Image

from src.tol2webdataset.utils import init_logger, init_shard, resize_image, determine_most_specific_known_rank, \
    create_taxon_tag_text, generate_text_files
from src.tol2webdataset.config import Config
from src.tol2webdataset.checkpoint import Checkpoint

temp_data_path = "/fs/ess/PAS2136/TreeOfLife/data"
temp_filter_folder: str = "/fs/scratch/PAS2136/gbif/webdataset/test"
temp_metadata_folder_path: str = "/fs/scratch/PAS2136/gbif/webdataset/test/metadata"
temp_resize_size: int = 720


class Runner:
    def __init__(self,
                 data_path,
                 filter_folder,
                 metadata_folder_path,
                 resize_size=0,
                 ):
        self.logger = init_logger(__name__)

        self.data_path = data_path
        self.filter_folder: str = filter_folder
        self.metadata_folder_path: str = metadata_folder_path
        self.resize_size: int = resize_size

        self.tar_folder: Optional[str] = None
        self.verification_folder: Optional[str] = None
        self.verification_scheme: List[str] = ["shard_id"]

        self.mpi_comm: MPI.Intracomm = MPI.COMM_WORLD
        self.mpi_rank: int = self.mpi_comm.rank
        self.total_time: Optional[int] = 300

    def is_enough_time(self):
        assert self.total_time is not None, ValueError("total_time is not set")
        if time.time() > int(os.getenv("SLURM_JOB_END_TIME", 0)) - self.total_time:
            raise TimeoutError("Not enough time")

    @staticmethod
    def load_table(folder: str, columns: List[str] = None) -> pd.DataFrame:
        all_files = glob.glob(os.path.join(folder, "*.csv"))
        if len(all_files) == 0:
            assert columns is not None, ValueError("No files found and columns are not defined")

            return pd.DataFrame(columns=columns)
        return pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

    @staticmethod
    def get_csv_writer(path: str, scheme: List[str]) -> TextIO:
        if not os.path.exists(path):
            file = open(path, "w")
            print(",".join(scheme), file=file, flush=True)
        else:
            file = open(path, "a")
        return file

    def ensure_folders_created(self):
        self.tar_folder = os.path.join(self.filter_folder, "tar_dataset")
        self.verification_folder = os.path.join(self.filter_folder, "verification")

        os.makedirs(self.verification_folder, exist_ok=True)
        os.makedirs(self.tar_folder, exist_ok=True)

    def get_schedule(self):
        schedule_df = pd.read_csv(os.path.join(self.filter_folder, "schedule.csv"))
        schedule_df = schedule_df.query(f"rank == {self.mpi_rank}")
        verification_df = self.load_table(self.verification_folder, self.verification_scheme)
        outer_join = schedule_df.merge(verification_df, how='outer', indicator=True, on=self.verification_scheme)
        return outer_join[(outer_join["_merge"] == 'left_only')].drop('_merge', axis=1)

    def convert_data_file_to_shard(self, data_df, tar_writer: wds.TarWriter):
        self.is_enough_time()

        filtering_df = data_df.reset_index(drop=True)
        source = filtering_df.iloc[0]["source"]
        server = filtering_df.iloc[0]["server"]
        filename = filtering_df.iloc[0]["filename"]

        metadata_df = filtering_df[['uuid', 'scientific_name', 'provided_common_name',
                                    'kingdom', 'phylum', 'class', 'order',
                                    'family', 'genus', 'species']]
        data_df = (pd.read_parquet(
            os.path.join(
                self.data_path,
                f"source={source}",
                f"server={server}",
                filename
            ),
            filters=[("uuid", "in", filtering_df["uuid"])],
            columns=['uuid', 'image', 'resized_size'])
                   .drop_duplicates(subset=["uuid"])
                   .merge(metadata_df, on="uuid", how="inner", validate="1:1"))

        for _, row in data_df.iterrows():
            try:
                sample = {
                    "__key__": row['uuid'],
                    "jpg": row['image']
                }
                resized_size = row['resized_size']

                if resized_size is None or len(resized_size) < 2:
                    self.logger.error(f"Invalid resized_size for {sample['__key__']}. Skipping.")
                    continue

                # Correctly assign height and width
                height, width = resized_size[:2]

                # Ensure the image bytes match the expected size
                expected_length = height * width * 3  # BGR
                actual_length = len(sample["jpg"])
                if actual_length != expected_length:
                    self.logger.error(
                        f"Byte length mismatch for UUID {sample['__key__']}: expected {expected_length}, got {actual_length}. Skipping.")
                    continue

                if self.resize_size != 0:
                    sample["jpg"] = resize_image(sample["jpg"], height, width, self.resize_size)
                else:
                    # Convert raw BGR bytes to JPEG using Pillow
                    img_array = np.frombuffer(sample["jpg"], dtype=np.uint8).reshape((height, width, 3))  # BGR
                    img_array = img_array[..., ::-1]  # Convert BGR to RGB
                    image = Image.fromarray(img_array, 'RGB')
                    with BytesIO() as img_buffer:
                        image.save(img_buffer, format='JPEG')
                        jpeg_bytes = img_buffer.getvalue()
                    sample["jpg"] = jpeg_bytes

                taxon_dict = {
                    'scientific_name': row['scientific_name'],
                    'original_common_name': row['provided_common_name'],
                    'kingdom': row['kingdom'],
                    'phylum': row['phylum'],
                    'class': row['class'],
                    'order': row['order'],
                    'family': row['family'],
                    'genus': row['genus'],
                    'species': row['species']
                }

                # Determine the most specific known rank
                most_specific_rank = determine_most_specific_known_rank(taxon_dict)

                # Generate all required text files
                text_files = generate_text_files(taxon_dict, most_specific_rank)

                # Add additional text files to the sample without UUID prefix
                for ext, content in text_files.items():
                    sample[f"{ext}"] = content.encode('utf-8')

                # Write the sample to the shard
                tar_writer.write(sample)
            except Exception as img_e:
                self.logger.error(f"Failed to process image for UUID {row['uuid']}: {img_e}")
                continue

    def process_shard(self, shard_id):
        tar_writer = init_shard(self.tar_folder, shard_id)
        try:
            shard_data_df = pd.read_parquet(self.metadata_folder_path + f"/shard_id={shard_id}")

            grouped_shard_df = shard_data_df.groupby(["source", "server", "filename"], group_keys=True)
            grouped_shard_df.apply(self.convert_data_file_to_shard, tar_writer)
        except Exception as e:
            tar_writer.close()
            raise e
        else:
            tar_writer.close()

    def run(self):
        self.ensure_folders_created()

        schedule = self.get_schedule()
        if len(schedule) == 0:
            self.logger.error(f"Schedule not found or empty for rank {self.mpi_rank}")
            exit(0)

        verification_io = self.get_csv_writer(f"{self.verification_folder}/{str(self.mpi_rank).zfill(4)}.csv",
                                              self.verification_scheme)

        for _, row in schedule.iterrows():
            if isinstance(row["shard_id"], float):
                self.logger.warning(f"Shard id is float: {row['shard_id']}")
            shard_id = int(row["shard_id"])
            try:
                self.process_shard(shard_id)
            except Exception as e:
                self.logger.exception(e)
                self.logger.error(f"Error occurred: {e}")
            else:
                print(f"{shard_id}", end="\n", file=verification_io)
                self.logger.debug(f"Completed filtering: {shard_id}")


if __name__ == "__main__":
    config_path = os.environ.get("CONFIG_PATH")
    if config_path is None:
        raise ValueError("CONFIG_PATH not set")

    config = Config.from_path(config_path, "tol2webdataset")
    checkpoint = Checkpoint.from_path(os.path.join(config.get_folder("path_to_output_folder"), "t2w_checkpoint.yaml"),
                                      {"scheduling_completed": False})

    logger = init_logger(__name__)

    if not checkpoint.get("scheduling_completed", False):
        logger.error("Scheduling wasn't complete, can't perform work")
        exit(1)

    WD_runner = Runner(
        config.get_folder("path_to_image_data"),
        config.get_folder("path_to_output_folder"),
        config.get_folder("metadata_folder"),
        config["t2w_parameters"].get("resize_size", 0),
    )

    logger.info("Starting worker")
    WD_runner.run()

    logger.info("Completed worker")
