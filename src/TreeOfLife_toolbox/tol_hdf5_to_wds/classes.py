"""
Toolbox integration for converting HDF5-backed TreeOfLife data into WebDatasets.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Dict, List

import h5py
import pandas as pd
import polars as pl
import pyspark.sql.functions as func
from pyspark.sql.window import Window

from TreeOfLife_toolbox.main.config import Config
from TreeOfLife_toolbox.main.filters import FilterRegister, SparkFilterToolBase
from TreeOfLife_toolbox.main.runners import MPIRunnerTool, RunnerRegister
from TreeOfLife_toolbox.main.schedulers import DefaultScheduler, SchedulerRegister
from TreeOfLife_toolbox.tol_hdf5_to_wds.utils import (
    convert_webp_to_jpeg,
    generate_text_files,
    init_shard_writer,
)


def _sanitize_path_column(column):
    return func.regexp_replace(column, "^file:", "")


@FilterRegister("tol_hdf5_to_wds")
class TolHDF5ToWDSFilter(SparkFilterToolBase):
    """
    Builds shard-level metadata derived from the parquet_to_hdf5 outputs.
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.filter_name = "tol_hdf5_to_wds"
        tool_cfg = self.config.get("tol_hdf5_to_wds", {}) or {}
        self.shard_size = int(tool_cfg.get("shard_size", 10000))
        self.shard_limit = int(tool_cfg.get("shard_limit", 0))
        self.metadata_glob = tool_cfg.get("metadata_glob", "**/*_metadata.parquet")
        self.include_sources = tool_cfg.get("include_sources") or []
        self.include_servers = tool_cfg.get("include_servers") or []
        self.taxa_glob = tool_cfg.get("taxa_glob")
        self.lookup_table_path = tool_cfg.get("lookup_table_path")
        self.lookup_path_column = tool_cfg.get("lookup_path_column", "path")
        self.shard_metadata_dir = os.path.join(
            self.tools_path, self.filter_name, "shard_metadata"
        )

        if self.shard_size <= 0:
            raise ValueError("tol_hdf5_to_wds.shard_size must be greater than zero")

        os.makedirs(self.shard_metadata_dir, exist_ok=True)

    def _discover_metadata_files(self) -> List[str]:
        dataset_root = self.config.get("path_to_input")
        if not dataset_root or not os.path.exists(dataset_root):
            raise ValueError("path_to_input must point to converted HDF5 dataset")

        pattern = os.path.join(dataset_root, self.metadata_glob)
        files = glob.glob(pattern, recursive=True)
        return sorted(files)

    def run(self):
        metadata_files = self._discover_metadata_files()
        if not metadata_files:
            raise ValueError(
                f"No metadata parquet files found using pattern: {self.metadata_glob}"
            )

        self.logger.info("Discovered %d metadata files", len(metadata_files))

        # Read only the UUID column to avoid schema conflicts from unused fields
        metadata_df = self.spark.read.parquet(*metadata_files).select("uuid")
        metadata_df = metadata_df.withColumn(
            "metadata_file", _sanitize_path_column(func.input_file_name())
        )
        metadata_df = metadata_df.withColumn(
            "hdf5_path",
            func.regexp_replace("metadata_file", "_metadata\\.parquet$", "_images.h5"),
        )
        metadata_df = metadata_df.withColumn(
            "base_name",
            func.regexp_extract("metadata_file", r"([^/]+)_metadata\.parquet$", 1),
        )

        if "source" not in metadata_df.columns:
            metadata_df = metadata_df.withColumn(
                "source", func.regexp_extract("metadata_file", r"source=([^/]+)", 1)
            )
        if "server" not in metadata_df.columns:
            metadata_df = metadata_df.withColumn(
                "server", func.regexp_extract("metadata_file", r"server=([^/]+)", 1)
            )

        if self.taxa_glob:
            taxa_files = glob.glob(self.taxa_glob)
            if not taxa_files:
                raise ValueError(
                    f"No taxonomy parquet files found for pattern: {self.taxa_glob}"
                )
            taxa_df = self.spark.read.parquet(*taxa_files)
            taxa_columns = [
                col_name
                for col_name in [
                    "uuid",
                    "scientific_name",
                    "provided_common_name",
                    "common_name",
                    "kingdom",
                    "phylum",
                    "class",
                    "order",
                    "family",
                    "genus",
                    "species",
                ]
                if col_name in taxa_df.columns
            ]
            taxa_df = taxa_df.select(*taxa_columns)
            metadata_df = metadata_df.join(taxa_df, on="uuid", how="left")

        if self.include_sources:
            metadata_df = metadata_df.filter(func.col("source").isin(self.include_sources))
        if self.include_servers:
            metadata_df = metadata_df.filter(func.col("server").isin(self.include_servers))

        if self.lookup_table_path:
            lookup_path_lower = self.lookup_table_path.lower()
            if lookup_path_lower.endswith(".parquet"):
                lookup_df = self.spark.read.parquet(self.lookup_table_path)
            else:
                lookup_df = (
                    self.spark.read.option("header", True).csv(self.lookup_table_path)
                )

            if "uuid" not in lookup_df.columns:
                raise ValueError("lookup table must contain a 'uuid' column")

            has_lookup_path = (
                self.lookup_path_column in lookup_df.columns
                if self.lookup_path_column
                else False
            )
            has_hdf5_override = "hdf5_path" in lookup_df.columns
            if not has_hdf5_override and "h5_file" in lookup_df.columns:
                lookup_df = lookup_df.withColumnRenamed("h5_file", "hdf5_path")
                has_hdf5_override = True

            if has_lookup_path:
                lookup_df = lookup_df.withColumn(
                    "lookup_base_name",
                    func.regexp_replace(
                        func.regexp_extract(
                            func.col(self.lookup_path_column), r"([^/]+)$", 1
                        ),
                        r"\.parquet$",
                        "",
                    ),
                )
            else:
                lookup_df = lookup_df.withColumn("lookup_base_name", func.lit(None))

            if has_hdf5_override:
                lookup_df = lookup_df.withColumnRenamed("hdf5_path", "lookup_hdf5_path")

            select_cols = ["uuid", "lookup_base_name"]
            if has_hdf5_override:
                select_cols.append("lookup_hdf5_path")

            lookup_df = lookup_df.select(*select_cols).dropDuplicates(["uuid"])
            metadata_df = metadata_df.join(lookup_df, on="uuid", how="inner")

            if has_lookup_path:
                metadata_df = metadata_df.filter(
                    func.col("lookup_base_name").isNull()
                    | (func.col("lookup_base_name") == func.col("base_name"))
                )

            if has_hdf5_override:
                if "hdf5_path" in metadata_df.columns:
                    metadata_df = metadata_df.drop("hdf5_path")
                metadata_df = metadata_df.withColumnRenamed(
                    "lookup_hdf5_path", "hdf5_path"
                )

            metadata_df = metadata_df.drop("lookup_base_name")

        # Drop duplicate UUIDs to avoid double writes
        metadata_df = metadata_df.dropDuplicates(["uuid"])

        required_columns = [
            "uuid",
            "hdf5_path",
            "source",
            "server",
            "scientific_name",
            "common_name",
            "provided_common_name",
            "kingdom",
            "phylum",
            "class",
            "order",
            "family",
            "genus",
            "species",
        ]

        string_columns = set(required_columns)

        for col_name in required_columns:
            if col_name not in metadata_df.columns:
                default_value = func.lit(None)
                if col_name in string_columns:
                    default_value = default_value.cast("string")
                metadata_df = metadata_df.withColumn(col_name, default_value)

        ordering_window = Window.orderBy("metadata_file", "uuid")
        metadata_df = (
            metadata_df.withColumn(
                "row_number", func.row_number().over(ordering_window)
            )
            .withColumn(
                "shard_id",
                func.floor((func.col("row_number") - 1) / self.shard_size).cast("int"),
            )
            .drop("row_number")
        )

        if self.shard_limit > 0:
            metadata_df = metadata_df.filter(func.col("shard_id") < self.shard_limit)

        if metadata_df.count() == 0:
            raise ValueError("No metadata rows left after filtering/shard limiting")

        selected_columns = required_columns + ["shard_id"]
        sharded_df = metadata_df.select(*selected_columns)

        (
            sharded_df.repartition("shard_id")
            .write.partitionBy("shard_id")
            .mode("overwrite")
            .parquet(self.shard_metadata_dir)
        )
        self.logger.info("Wrote shard metadata to %s", self.shard_metadata_dir)

        schedule_df = (
            sharded_df.select("shard_id")
            .dropDuplicates()
            .withColumn(
                "metadata_path",
                func.concat(
                    func.lit(self.shard_metadata_dir),
                    func.lit("/shard_id="),
                    func.col("shard_id"),
                ),
            )
            .withColumn("server_name", func.lit("tol_hdf5_to_wds"))
            .withColumn("partition_id", func.col("shard_id"))
        )

        self.save_filter(schedule_df)
        total_shards = schedule_df.count()
        self.logger.info("Prepared %d shards", total_shards)


@SchedulerRegister("tol_hdf5_to_wds")
class TolHDF5ToWDSScheduler(DefaultScheduler):
    """
    Scheduler that treats each metadata shard as an independent work unit.
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.filter_name = "tol_hdf5_to_wds"
        # Scheduling keys only. The data columns (metadata_path, shard_id) ride along in the filter table and are picked up by the runner's data_scheme; including them here duplicates them into schedule.csv and collides on the runner's merge.
        self.scheme = ["server_name", "partition_id"]


@RunnerRegister("tol_hdf5_to_wds")
class TolHDF5ToWDSRunner(MPIRunnerTool):
    """
    Converts shard metadata into WebDataset tar archives.
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.filter_name = "tol_hdf5_to_wds"
        self.data_scheme = ["metadata_path", "shard_id", "server_name", "partition_id"]
        self.verification_scheme = ["server_name", "partition_id"]

        params = self.config.get("tol_hdf5_to_wds", {}) or {}
        self.tar_output_root = params.get("tar_output_root") or os.path.join(
            self.tools_path, self.filter_name, "tar_dataset"
        )
        self.resize_size = int(params.get("resize_size", 224))
        self.shard_prefix = params.get("shard_prefix", "shard")
        self.total_time = int(params.get("runner_timeout_seconds", 3600))

        os.makedirs(self.tar_output_root, exist_ok=True)

    def _load_shard_metadata(self, metadata_path: str) -> pl.DataFrame:
        metadata_dir = Path(metadata_path)
        if not metadata_dir.exists():
            raise FileNotFoundError(f"Shard metadata directory missing: {metadata_path}")

        files = sorted(metadata_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No Parquet files found under {metadata_path}")

        return pl.read_parquet([str(path) for path in files])

    def _iter_records(self, shard_df: pl.DataFrame):
        for row in shard_df.iter_rows(named=True):
            yield row

    def _open_hdf5(self, path: str, cache: Dict[str, h5py.File]) -> h5py.File:
        if path not in cache:
            if not os.path.exists(path):
                raise FileNotFoundError(f"HDF5 file missing: {path}")
            cache[path] = h5py.File(path, "r")
        return cache[path]

    def apply_filter(
        self, filtering_df: pd.DataFrame, server_name: str, partition_id: int
    ) -> int:
        shard_id = int(filtering_df.iloc[0]["shard_id"])
        metadata_path = filtering_df.iloc[0]["metadata_path"]
        return self.process_shard(shard_id, metadata_path)

    def process_shard(self, shard_id: int, metadata_path: str) -> int:
        self.is_enough_time()
        shard_df = self._load_shard_metadata(metadata_path)
        if shard_df.height == 0:
            self.logger.warning("Shard %s is empty, skipping", shard_id)
            return 0

        tar_writer, tar_path = init_shard_writer(
            self.tar_output_root, shard_id, self.shard_prefix
        )
        hdf5_cache: Dict[str, h5py.File] = {}
        written = 0
        failed = 0

        try:
            for row in self._iter_records(shard_df):
                uuid = row.get("uuid")
                hdf5_path = row.get("hdf5_path")
                if not uuid or not hdf5_path:
                    failed += 1
                    continue

                try:
                    h5_file = self._open_hdf5(hdf5_path, hdf5_cache)
                    images_group = h5_file.get("images")
                    if images_group is None or uuid not in images_group:
                        failed += 1
                        continue

                    dataset = images_group[uuid][:]
                    webp_bytes = dataset.tobytes()
                    jpeg_bytes = convert_webp_to_jpeg(webp_bytes, self.resize_size)

                    taxon_dict = {
                        "scientific_name": row.get("scientific_name"),
                        "common_name": row.get("common_name"),
                        "kingdom": row.get("kingdom"),
                        "phylum": row.get("phylum"),
                        "class": row.get("class"),
                        "order": row.get("order"),
                        "family": row.get("family"),
                        "genus": row.get("genus"),
                        "species": row.get("species"),
                    }

                    sample = {"__key__": uuid, "jpg": jpeg_bytes}
                    for ext, content in generate_text_files(taxon_dict).items():
                        sample[ext] = content.encode("utf-8")

                    tar_writer.write(sample)
                    written += 1
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    self.logger.error(
                        "Failed to process uuid %s in shard %s: %s", uuid, shard_id, exc
                    )
        finally:
            tar_writer.close()
            for handle in hdf5_cache.values():
                handle.close()

        self.logger.info(
            "Shard %s written to %s (%s records, %s failures)",
            shard_id,
            tar_path,
            written,
            failed,
        )
        return written
