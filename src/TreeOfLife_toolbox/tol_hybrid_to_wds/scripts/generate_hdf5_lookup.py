#!/usr/bin/env python
"""
Generate a UUID-to-HDF5 lookup table from a converted Tree of Life dataset.

Usage:
    python generate_hdf5_lookup.py \
        --data-root /fs/scratch/PAS2136/TreeOfLife_test-wds/data \
        --output /fs/scratch/PAS2136/TreeOfLife_test-wds/lookup-tables/all/hdf5_lookup.parquet \
        --sample 100
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Iterable, List

import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a lookup Parquet of UUIDs to HDF5/metadata paths."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Root directory containing source=*/server=*/data_*_{metadata,images} files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination Parquet file (e.g. /.../lookup-tables/all/lookup.parquet).",
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=100.0,
        help="Percentage of UUIDs to keep (0 < sample <= 100). Default: 100.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when down-sampling.",
    )
    return parser.parse_args()


def discover_metadata_files(data_root: str) -> List[str]:
    pattern = os.path.join(data_root, "source=*", "server=*", "data_*_metadata.parquet")
    return sorted(glob.glob(pattern))


def build_lazy_frames(metadata_files: Iterable[str]) -> List[pl.LazyFrame]:
    lazy_frames: List[pl.LazyFrame] = []
    for meta_path in metadata_files:
        source = _extract_between(meta_path, "source=", "/server=")
        server = _extract_between(meta_path, "server=", "/data_")
        base_name = os.path.basename(meta_path).replace("_metadata.parquet", "")
        hdf5_path = meta_path.replace("_metadata.parquet", "_images.h5")

        lazy = (
            pl.scan_parquet(meta_path)
            .select("uuid")
            .with_columns(
                pl.lit(meta_path).alias("metadata_path"),
                pl.lit(hdf5_path).alias("hdf5_path"),
                pl.lit(source).alias("source"),
                pl.lit(server).alias("server"),
                pl.lit(base_name).alias("base_name"),
            )
        )
        lazy_frames.append(lazy)
    return lazy_frames


def _extract_between(value: str, start_token: str, end_token: str) -> str:
    start_idx = value.find(start_token)
    if start_idx == -1:
        return ""
    start_idx += len(start_token)
    end_idx = value.find(end_token, start_idx)
    if end_idx == -1:
        return value[start_idx:]
    return value[start_idx:end_idx]


def main() -> None:
    args = parse_args()
    data_root = os.path.abspath(args.data_root)
    metadata_files = discover_metadata_files(data_root)

    if not metadata_files:
        raise SystemExit(
            f"No metadata parquet files found under {data_root}. "
            "Expected pattern source=*/server=*/data_*_metadata.parquet"
        )

    lazy_frames = build_lazy_frames(metadata_files)
    lookup_lazy = pl.concat(lazy_frames)

    if args.sample <= 0 or args.sample > 100:
        raise SystemExit("--sample must be in the interval (0, 100].")

    if args.sample < 100:
        frac = args.sample / 100.0
        lookup_lazy = lookup_lazy.sample(
            fraction=frac, with_replacement=False, shuffle=True, seed=args.seed
        )

    lookup_df = lookup_lazy.collect(streaming=True)

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lookup_df.write_parquet(output_path, compression="zstd")
    print(f"Wrote {lookup_df.height} records to {output_path}")


if __name__ == "__main__":
    main()
