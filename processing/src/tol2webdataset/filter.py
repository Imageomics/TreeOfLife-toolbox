import os

from src.tol2webdataset.utils import init_logger
import pyspark.sql.functions as func
from pyspark.sql import SparkSession, Window
from pyspark.sql.types import IntegerType

from src.tol2webdataset.config import Config
from src.tol2webdataset.checkpoint import Checkpoint

source_pattern = r"source=([^/]+)"  # Extracts the value of source (e.g., "alpha")
server_pattern = r"server=([^/]+)"  # Extracts the value of server (e.g., "beta")
filename_pattern = r"([^/]+\.parquet)$"  # Extracts the file name (e.g., "data_123.parquet")


class Filter:
    def __init__(self,
                 source_taxa_path,
                 data_path,
                 metadata_save_path,
                 shard_size,
                 shard_limit,
                 included_sources=None,
                 using_lookup_table=False
                 ):
        self.logger = init_logger(__name__)
        self.spark: SparkSession = SparkSession.builder.appName("Filtering").getOrCreate()
        self.spark.conf.set("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")
        self.spark.conf.set("spark.sql.parquet.int96RebaseModeInWrite", "CORRECTED")

        self.source_taxa_path = source_taxa_path
        self.data_path = data_path
        self.using_lookup_table = using_lookup_table

        self.metadata_save_path = metadata_save_path

        self.shard_size = shard_size
        self.shard_limit = shard_limit
        self.included_sources = included_sources if included_sources else []

    def run(self):
        if self.using_lookup_table:
            data_df = self.spark.read.csv(self.data_path, header=True).repartition(20)
        else:
            data_df = (
                self.spark.read.option("basePath", self.data_path)
                .parquet(f"{self.data_path}/source=*/server=*/data_*.parquet")
                .select("uuid")
                .withColumn(
                    "path",
                    func.substring(func.input_file_name(), len("file:/"), 2000000),
                )
                .select("uuid", "path")
            )

        data_df = (data_df
                   .withColumn("source", func.regexp_extract("path", source_pattern, 1))
                   .withColumn("server", func.regexp_extract("path", server_pattern, 1))
                   .withColumn("filename", func.regexp_extract("path", filename_pattern, 1))
                   .drop("path")
                   .select("uuid", "source", "server", "filename"))
        metadata_df = self.spark.read.parquet(self.source_taxa_path)

        if "source" in metadata_df.columns:
            metadata_df = metadata_df.drop("source")

        if len(self.included_sources) != 0:
            data_df = data_df.filter(data_df["source"].isin(self.included_sources))

        data_df = data_df.join(metadata_df, on="uuid", how="inner")
        print(data_df.count())

        print("DEDUP")
        data_df = data_df.drop_duplicates(["uuid"])
        print(data_df.count())

        window = Window.orderBy("filename")
        data_df_with_shard = (data_df
                              .withColumn("row_number", func.row_number().over(window))
                              .withColumn("shard_id",
                                          func.floor(func.col("row_number") / self.shard_size).cast(IntegerType()))
                              .drop("row_number")
                              )

        if self.shard_limit > 0:
            data_df_with_shard = data_df_with_shard.filter(func.col("shard_id") >= self.shard_limit)

        (data_df_with_shard
         .repartition("shard_id")
         .write
         .partitionBy("shard_id")
         .mode("overwrite")
         .format("parquet")
         .save(self.metadata_save_path))


if __name__ == "__main__":
    config_path = os.environ.get("CONFIG_PATH")
    if config_path is None:
        raise ValueError("CONFIG_PATH not set")

    config = Config.from_path(config_path, "tol2webdataset")
    checkpoint = Checkpoint.from_path(os.path.join(config.get_folder("path_to_output_folder"), "t2w_checkpoint.yaml"),
                                      {"filtering_completed": False})
    logger = init_logger(__name__)

    lookup_table_presence = bool(config.get_folder("path_to_image_lookup_table"))
    WD_filter = Filter(
        config.get_folder("path_to_source_taxa"),
        config.get_folder("path_to_image_lookup_table") if lookup_table_presence else config.get_folder(
            "path_to_image_data"),
        config.get_folder("metadata_folder"),
        config["t2w_parameters"]["shard_size"],
        config["t2w_parameters"]["shard_count_limit"],
        included_sources=config["t2w_parameters"].get("included_sources"),
        using_lookup_table=lookup_table_presence
    )

    logger.info("Starting filter")
    WD_filter.run()

    logger.info("completed filter")
    checkpoint["filtering_completed"] = True
