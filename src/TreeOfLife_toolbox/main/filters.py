import os.path
from functools import partial
from typing import Optional

import pandas as pd
import pyspark.sql as ps
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType

from TreeOfLife_toolbox.main.config import Config
from TreeOfLife_toolbox.main.registry import ToolsBase
from TreeOfLife_toolbox.main.registry import ToolsRegistryBase
from TreeOfLife_toolbox.main.utils import SuccessEntry

FilterRegister = partial(ToolsRegistryBase.register, "filter")


class FilterToolBase(ToolsBase):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.filter_family = "filter"


class SparkFilterToolBase(FilterToolBase):
    success_scheme = SuccessEntry.get_success_spark_scheme()

    def __init__(self, cfg: Config, spark: SparkSession = None):
        super().__init__(cfg)
        self.spark: SparkSession = (
            spark
            if spark is not None
            else SparkSession.builder.appName("Filtering").getOrCreate()
        )
        self.spark.conf.set("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")
        self.spark.conf.set("spark.sql.parquet.int96RebaseModeInWrite", "CORRECTED")

    def run(self):
        raise NotImplementedError()

    def load_data_parquet(self, scheme: Optional[StructType] = None):
        if scheme is None:
            scheme = self.success_scheme
        return (
            self.spark.read.schema(scheme)
            .option("basePath", self.downloaded_images_path)
            .parquet(
                self.downloaded_images_path
                + "/server_name=*/partition_id=*/successes.parquet"
            )
        )

    def save_filter(self, df: ps.DataFrame):
        if self.filter_name is None:
            raise ValueError("filter name was not defined")
        (
            df.repartition(10).write.csv(
                os.path.join(self.tools_path, self.filter_name, "filter_table"),
                header=True,
                mode="overwrite",
            )
        )

    def __del__(self):
        if self.spark is not None:
            self.spark.stop()


class PythonFilterToolBase(FilterToolBase):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

    def get_all_paths_to_merge(self) -> pd.DataFrame:
        all_schedules = []
        path = self.downloaded_images_path
        for folder in os.listdir(path):
            server_name = folder.split("=")[1]
            for partition in os.listdir(f"{path}/{folder}"):
                partition_path = f"{path}/{folder}/{partition}"
                if not os.path.exists(
                        f"{partition_path}/successes.parquet"
                ) or not os.path.exists(f"{partition_path}/completed"):
                    continue
                all_schedules.append([server_name, partition.split("=")[1]])
        return pd.DataFrame(all_schedules, columns=["server_name", "partition_id"])

    def run(self):
        filter_table = self.get_all_paths_to_merge()

        filter_table_folder = os.path.join(
            self.tools_path, self.filter_name, "filter_table"
        )
        os.makedirs(filter_table_folder, exist_ok=True)

        filter_table.to_csv(
            filter_table_folder + "/table.csv", header=True, index=False
        )
