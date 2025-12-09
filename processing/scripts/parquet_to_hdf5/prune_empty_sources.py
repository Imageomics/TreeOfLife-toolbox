#!/usr/bin/env python3
"""
Identify and optionally delete empty source data_<uuid>.parquet files.

The script is intended to run in two phases:

    1) --dry-run
         Scan the provided source tree, record every legacy parquet whose
         row count is zero, and write CSV/JSON reports beneath --logs.

    2) --yes-delete
         Read the previously generated CSV plan and delete the files,
         writing an updated summary describing what was removed.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pyarrow.parquet as pq


@dataclass(frozen=True)
class EmptyParquet:
    uuid: str
    server: str
    path: Path
    size_bytes: int
    row_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan TreeOfLife source directories for empty legacy data_*.parquet files."
    )
    parser.add_argument(
        "--source-root",
        required=True,
        type=Path,
        help="Root directory that contains server=* folders with legacy data_*.parquet files.",
    )
    parser.add_argument(
        "--logs",
        required=True,
        type=Path,
        help="Directory for dry-run CSV summaries and deletion logs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan for empty files and record results without deleting anything.",
    )
    parser.add_argument(
        "--yes-delete",
        action="store_true",
        help="Delete files identified in the dry-run CSV plan.",
    )
    parser.add_argument(
        "--plan-file",
        type=Path,
        help="Optional explicit path to a dry-run CSV plan (defaults to <logs>/<basename>_empty_sources.csv).",
    )
    return parser.parse_args()


def error(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    raise SystemExit(f"ERROR: {msg}")


def find_server_name(path: Path) -> str:
    for part in path.parts:
        if part.startswith("server="):
            return part
    return "server=?"


def iter_source_parquets(source_root: Path) -> Iterable[Path]:
    glob_pattern = "data_*.parquet"
    for path in sorted(source_root.rglob(glob_pattern)):
        name = path.name
        if name.endswith("_metadata.parquet"):
            continue
        if name.endswith("_images.parquet"):
            continue
        yield path


def count_rows(parquet_path: Path) -> Optional[int]:
    try:
        meta = pq.ParquetFile(parquet_path)
        return meta.metadata.num_rows
    except Exception:
        return None


def detect_empty_files(source_root: Path) -> List[EmptyParquet]:
    records: List[EmptyParquet] = []
    for parquet_path in iter_source_parquets(source_root):
        rows = count_rows(parquet_path)
        if rows is None:
            continue
        if rows > 0:
            continue
        uuid = parquet_path.stem.replace("data_", "", 1)
        records.append(
            EmptyParquet(
                uuid=uuid,
                server=find_server_name(parquet_path),
                path=parquet_path,
                size_bytes=parquet_path.stat().st_size,
                row_count=rows,
            )
        )
    return records


def default_plan_paths(log_dir: Path, source_root: Path) -> tuple[Path, Path]:
    base = source_root.resolve().name
    csv_path = log_dir / f"{base}_empty_sources.csv"
    summary_path = log_dir / f"{base}_empty_sources_summary.json"
    return csv_path, summary_path


def write_csv(records: List[EmptyParquet], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["uuid", "server", "path", "size_bytes", "row_count"])
        for rec in records:
            writer.writerow([rec.uuid, rec.server, str(rec.path), rec.size_bytes, rec.row_count])


def write_summary(path: Path, *, scanned: int, empty: int, deleted: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "total_files_scanned": scanned,
        "empty_files": empty,
        "deleted_files": deleted,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def load_plan(csv_path: Path) -> List[EmptyParquet]:
    if not csv_path.exists():
        error(f"Plan file not found: {csv_path}")
    records: List[EmptyParquet] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            path = Path(row["path"])
            records.append(
                EmptyParquet(
                    uuid=row["uuid"],
                    server=row["server"],
                    path=path,
                    size_bytes=int(row["size_bytes"]),
                    row_count=int(row["row_count"]),
                )
            )
    return records


def delete_files(records: List[EmptyParquet]) -> List[EmptyParquet]:
    removed: List[EmptyParquet] = []
    for rec in records:
        try:
            os.remove(rec.path)
            removed.append(rec)
        except FileNotFoundError:
            continue
    return removed


def main() -> int:
    args = parse_args()
    source_root: Path = args.source_root.expanduser().resolve()
    log_dir: Path = args.logs.expanduser().resolve()

    if not source_root.is_dir():
        error(f"Source root is not a directory: {source_root}")
    if not args.dry_run and not args.yes_delete:
        error("Specify at least one of --dry-run or --yes-delete")

    if args.plan_file:
        csv_path = args.plan_file.expanduser().resolve()
        summary_path = log_dir / f"{csv_path.stem}_summary.json"
    else:
        csv_path, summary_path = default_plan_paths(log_dir, source_root)

    if args.dry_run:
        parquets = list(iter_source_parquets(source_root))
        empty_records = detect_empty_files(source_root)
        write_csv(empty_records, csv_path)
        write_summary(summary_path, scanned=len(parquets), empty=len(empty_records), deleted=0)
        print(f"Dry-run complete. Empty files: {len(empty_records)}")
        print(f"CSV plan:     {csv_path}")
        print(f"Summary JSON: {summary_path}")
        if not args.yes_delete:
            return 0

    if args.yes_delete:
        plan_records = load_plan(csv_path)
        removed = delete_files(plan_records)
        write_summary(summary_path, scanned=len(plan_records), empty=len(plan_records), deleted=len(removed))
        print(f"Deleted {len(removed)} files (plan entries: {len(plan_records)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
