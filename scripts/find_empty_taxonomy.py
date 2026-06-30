#!/usr/bin/env python
"""
Scan shard_metadata for UUIDs lacking any taxonomy fields and emit a CSV map.

Usage:
    python scripts/find_empty_taxonomy.py /path/to/shard_metadata output.csv
"""

import argparse
import csv
from pathlib import Path

import polars as pl


def gather_null_taxonomy(shard_metadata_root: Path) -> pl.DataFrame:
    dataset = pl.scan_parquet(
        str(shard_metadata_root / "shard_id=*" / "*.parquet"), hive_partitioning=True
    )
    taxonomy_cols = [
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
    filter_expr = None
    for col in taxonomy_cols:
        expr = pl.col(col).is_null() | (pl.col(col).str.strip_chars().eq(""))
        filter_expr = expr if filter_expr is None else (filter_expr & expr)

    empty_taxa_df = (
        dataset.filter(filter_expr)
        .select("uuid", "shard_id")
        .collect()
        .sort("shard_id")
    )
    return empty_taxa_df


def main():
    parser = argparse.ArgumentParser(
        description="Find shard metadata rows with empty taxonomy."
    )
    parser.add_argument("shard_metadata_root", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()

    df = gather_null_taxonomy(args.shard_metadata_root)
    if df.is_empty():
        args.output_csv.write_text("uuid,shard_id\n")
        return

    with args.output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["uuid", "shard_id"])
        for row in df.iter_rows():
            writer.writerow(row)


if __name__ == "__main__":
    main()
