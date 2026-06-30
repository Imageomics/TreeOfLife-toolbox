#!/usr/bin/env python
"""
Single-node converter from the hybrid Tree of Life HDF5/metadata dataset to WebDataset shards.

Example:
    python tol_hdf5_to_wds.py \
        --input-root /fs/ess/PAS2136/TreeOfLife/data \
        --lookup /fs/scratch/PAS2136/TreeOfLife_test-wds/lookup-tables/all/hdf5_lookup.parquet \
        --taxa-glob "/fs/ess/PAS2136/TreeOfLife/annotations/resolved_taxa/*/*.parquet" \
        --output-dir /fs/scratch/PAS2136/TreeOfLife_test-wds/wds/manual \
        --metadata-glob "**/*_metadata.parquet"
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import polars as pl
import webdataset as wds

try:
    from TreeOfLife_toolbox.tol_hdf5_to_wds.utils import (
        convert_webp_to_jpeg,
        generate_text_files,
    )
except ImportError as exc:  # pragma: no cover - script usage only
    raise SystemExit(
        "Unable to import TreeOfLife_toolbox.tol_hdf5_to_wds.utils. "
        "Please install the toolbox package (pip install -e .) before running this script."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create WebDataset shards from HDF5 dataset")
    parser.add_argument("--input-root", required=True, help="Root folder containing *_metadata.parquet + *_images.h5")
    parser.add_argument(
        "--metadata-glob",
        default="**/*_metadata.parquet",
        help="Glob pattern under --input-root to find metadata files (default: %(default)s)",
    )
    parser.add_argument(
        "--lookup",
        required=True,
        help="Lookup table (CSV or Parquet) listing UUIDs to include. "
        "Must contain a 'uuid' column; optional 'hdf5_path' overrides file locations.",
    )
    parser.add_argument(
        "--taxa-glob",
        help="Optional glob pointing to taxonomy parquet files (e.g. /path/source=*/*.parquet).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where shard-XXXXX.tar files will be written.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=10000,
        help="Number of samples per shard (default: %(default)s)",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=224,
        help="Output JPEG square size in pixels (0 keeps original WebP dimensions).",
    )
    parser.add_argument(
        "--shard-prefix",
        default="shard",
        help="Prefix for tar files (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: %(default)s)",
    )
    return parser.parse_args()


def setup_logger(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("tol_hdf5_to_wds")


def discover_metadata_files(root: str, pattern: str) -> List[str]:
    search_pattern = os.path.join(root, pattern)
    files = sorted(glob.glob(search_pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No metadata parquet files found using pattern: {search_pattern}")
    return files


def build_metadata_lazy(metadata_files: List[str]) -> pl.LazyFrame:
    # Include file paths to derive HDF5 locations.
    metadata_lazy = pl.scan_parquet(metadata_files, include_file_paths="metadata_file")
    metadata_lazy = metadata_lazy.with_columns(
        pl.col("metadata_file")
        .str.replace(r"^file:", "", literal=True)
        .alias("metadata_file_clean")
    )
    metadata_lazy = metadata_lazy.with_columns(
        pl.col("metadata_file_clean")
        .str.replace("_metadata.parquet$", "_images.h5")
        .alias("hdf5_path"),
        pl.col("metadata_file_clean")
        .str.extract(r"([^/]+)_metadata\.parquet$", group_index=1)
        .alias("base_name"),
    )
    schema_names = metadata_lazy.collect_schema().names()
    drop_cols = [c for c in ("source", "server") if c in schema_names]
    if drop_cols:
        metadata_lazy = metadata_lazy.drop(drop_cols)
    metadata_lazy = metadata_lazy.with_columns(
        pl.col("metadata_file_clean").str.extract(r"source=([^/]+)", group_index=1).alias("source"),
        pl.col("metadata_file_clean").str.extract(r"server=([^/]+)", group_index=1).alias("server"),
    )
    desired_cols = [
        "uuid",
        "hdf5_path",
        "base_name",
        "source",
        "server",
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
    cols_present = [c for c in desired_cols if c in metadata_lazy.collect_schema().names()]
    metadata_lazy = metadata_lazy.select(cols_present + ["metadata_file_clean"])
    return metadata_lazy


def load_lookup_table(path: str) -> pl.LazyFrame:
    ext = Path(path).suffix.lower()
    if ext == ".parquet":
        lookup = pl.scan_parquet(path)
    else:
        lookup = pl.scan_csv(path)
    schema = lookup.collect_schema()
    schema_names = schema.names()
    if "uuid" not in schema:
        raise ValueError(f"Lookup table {path} must contain a 'uuid' column")
    keep_cols = ["uuid"]
    if "hdf5_path" in schema_names:
        keep_cols.append("hdf5_path")
    if "path" in schema_names:
        keep_cols.append("path")
    return lookup.select(keep_cols)


def load_taxa_lazy(glob_pattern: str) -> pl.LazyFrame:
    taxa_files = glob.glob(glob_pattern)
    if not taxa_files:
        raise FileNotFoundError(f"No taxonomy files match {glob_pattern}")
    taxa_lazy = pl.scan_parquet(taxa_files, extra_columns="ignore")
    schema_names = taxa_lazy.collect_schema().names()
    cols = [col for col in [
        "uuid",
        "common_name",
        "provided_common_name",
        "kingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
    ] if col in schema_names]
    return taxa_lazy.select(cols)


def merge_metadata(
    metadata_lazy: pl.LazyFrame,
    lookup_lazy: pl.LazyFrame,
    taxa_lazy: pl.LazyFrame | None,
    logger: logging.Logger,
) -> pl.DataFrame:
    joined = metadata_lazy.join(lookup_lazy, on="uuid", how="inner")
    # Ensure column types align before sorting/collecting
    schema_names = joined.collect_schema().names()
    cast_exprs = []
    if "common_name" in schema_names:
        cast_exprs.append(pl.col("common_name").cast(pl.Utf8, strict=False))
    if "provided_common_name" in schema_names:
        cast_exprs.append(pl.col("provided_common_name").cast(pl.Utf8, strict=False))
    if "source" in schema_names:
        cast_exprs.append(pl.col("source").cast(pl.Utf8, strict=False))
    if "server" in schema_names:
        cast_exprs.append(pl.col("server").cast(pl.Utf8, strict=False))
    if cast_exprs:
        joined = joined.with_columns(cast_exprs)
    joined_schema = joined.collect_schema().names()
    if "hdf5_path_right" in joined_schema:
        joined = joined.with_columns(
            pl.when(pl.col("hdf5_path_right").is_not_null())
            .then(pl.col("hdf5_path_right"))
            .otherwise(pl.col("hdf5_path"))
            .alias("hdf5_path")
        ).drop("hdf5_path_right")
    if taxa_lazy is not None:
        joined = joined.join(taxa_lazy, on="uuid", how="left")
    joined = joined.unique(subset=["uuid"])
    result = joined.sort(["hdf5_path", "uuid"]).collect(streaming=True)
    logger.info("Collected %s metadata rows", result.height)
    return result


def chunk_records(df: pl.DataFrame, shard_size: int) -> Iterable[Tuple[int, List[dict]]]:
    buffer: List[dict] = []
    shard_id = 0
    for row in df.iter_rows(named=True):
        buffer.append(row)
        if len(buffer) >= shard_size:
            yield shard_id, buffer
            buffer = []
            shard_id += 1
    if buffer:
        yield shard_id, buffer


def process_shard(
    shard_id: int,
    records: List[dict],
    output_dir: Path,
    resize: int,
    shard_prefix: str,
    logger: logging.Logger,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = output_dir / f"{shard_prefix}-{shard_id:05d}.tar"
    writer = wds.TarWriter(str(shard_path))
    hdf5_cache: Dict[str, h5py.File] = {}
    written = failed = 0
    try:
        for record in records:
            uuid = record.get("uuid")
            hdf5_path = record.get("hdf5_path")
            if not uuid or not hdf5_path:
                failed += 1
                continue
            try:
                h5_file = _open_hdf5(hdf5_path, hdf5_cache)
                dataset = h5_file["images"][uuid][:]
                webp_bytes = dataset.tobytes()
                jpeg_bytes = convert_webp_to_jpeg(webp_bytes, resize)

                taxon_dict = {
                    "scientific_name": record.get("scientific_name"),
                    "common_name": record.get("common_name"),
                    "kingdom": record.get("kingdom"),
                    "phylum": record.get("phylum"),
                    "class": record.get("class"),
                    "order": record.get("order"),
                    "family": record.get("family"),
                    "genus": record.get("genus"),
                    "species": record.get("species"),
                }
                sample = {"__key__": uuid, "jpg": jpeg_bytes}
                for ext, content in generate_text_files(taxon_dict).items():
                    sample[ext] = content.encode("utf-8")
                writer.write(sample)
                written += 1
            except Exception as exc:  # pragma: no cover
                failed += 1
                logger.error("Shard %s: failed uuid %s (%s)", shard_id, uuid, exc)
        logger.info("Shard %s written with %s samples (%s failures)", shard_id, written, failed)
    finally:
        writer.close()
        for handle in hdf5_cache.values():
            with contextlib.suppress(Exception):
                handle.close()


def _open_hdf5(path: str, cache: Dict[str, h5py.File]) -> h5py.File:
    if path not in cache:
        cache[path] = h5py.File(path, "r")
    return cache[path]


def main():
    args = parse_args()
    logger = setup_logger(args.log_level)

    metadata_files = discover_metadata_files(args.input_root, args.metadata_glob)
    logger.info("Found %s metadata parquet files", len(metadata_files))

    metadata_lazy = build_metadata_lazy(metadata_files)
    lookup_lazy = load_lookup_table(args.lookup)
    taxa_lazy = load_taxa_lazy(args.taxa_glob) if args.taxa_glob else None

    metadata = merge_metadata(metadata_lazy, lookup_lazy, taxa_lazy, logger)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    last_shard = -1
    for shard_id, records in chunk_records(metadata, args.shard_size):
        last_shard = shard_id
        process_shard(shard_id, records, output_dir, args.resize, args.shard_prefix, logger)

    total_shards = (last_shard + 1) if metadata.height > 0 else 0
    logger.info("Completed WebDataset creation: %s shards written", total_shards)


if __name__ == "__main__":
    main()
