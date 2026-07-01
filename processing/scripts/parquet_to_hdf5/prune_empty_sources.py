#!/usr/bin/env python3
"""
Identify and optionally delete empty source data_<uuid>.parquet files.

Typical workflow:
  1. Run with --dry-run to record every zero-row parquet and review the CSV plan.
  2. Once satisfied, rerun with --yes-delete. Add --prune-empty-dirs to record and
     remove any server=* directories that would become empty once those files are gone.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

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
    parser.add_argument(
        "--prune-empty-dirs",
        action="store_true",
        help="Consider server=* directories for removal when they become empty.",
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
    for path in sorted(source_root.rglob("data_*.parquet")):
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


def detect_empty_files(parquet_paths: Iterable[Path]) -> List[EmptyParquet]:
    records: List[EmptyParquet] = []
    for parquet_path in parquet_paths:
        rows = count_rows(parquet_path)
        if rows is None or rows > 0:
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


def resolve_plan_paths(
    log_dir: Path, source_root: Path, plan_file: Optional[Path]
) -> Tuple[Path, Path, Path, Path]:
    if plan_file:
        csv_path = plan_file.expanduser().resolve()
        stem = csv_path.stem
    else:
        source_base = source_root.resolve().name
        stem = f"{source_base}_empty_sources"
        csv_path = (log_dir / f"{stem}.csv").resolve()
    summary_path = log_dir / f"{stem}_summary.json"
    dir_plan_path = log_dir / f"{stem}_dirs_planned.csv"
    dir_deleted_path = log_dir / f"{stem}_dirs_deleted.csv"
    return csv_path, summary_path, dir_plan_path, dir_deleted_path


def write_csv(records: List[EmptyParquet], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["uuid", "server", "path", "size_bytes", "row_count"])
        for rec in records:
            writer.writerow([rec.uuid, rec.server, str(rec.path), rec.size_bytes, rec.row_count])


def write_dir_plan(paths: List[Path], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["server", "path"])
        for path in paths:
            writer.writerow([find_server_name(path), str(path)])


def write_summary(
    path: Path,
    *,
    scanned: int,
    empty: int,
    deleted: int = 0,
    pruned_dirs: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "total_files_scanned": scanned,
        "empty_files": empty,
        "deleted_files": deleted,
        "empty_directories_removed": pruned_dirs,
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


def _normalize_paths(paths: Iterable[Path]) -> Set[Path]:
    return {p.resolve() for p in paths}


def find_empty_server_dirs(
    source_root: Path, pretend_removed: Optional[Set[Path]] = None
) -> List[Path]:
    pretend = pretend_removed or set()
    empty_dirs: List[Path] = []
    for server_dir in sorted(source_root.glob("server=*")):
        if not server_dir.is_dir():
            continue
        try:
            entries = list(server_dir.iterdir())
        except FileNotFoundError:
            continue
        remaining = [entry for entry in entries if entry.resolve() not in pretend]
        if remaining:
            continue
        empty_dirs.append(server_dir)
    return empty_dirs


def prune_empty_server_dirs(source_root: Path) -> List[Path]:
    removed: List[Path] = []
    for server_dir in find_empty_server_dirs(source_root):
        try:
            server_dir.rmdir()
            removed.append(server_dir)
        except OSError:
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
    if args.prune_empty_dirs and not args.yes_delete:
        print("NOTE: Directory pruning is preview-only because --yes-delete was omitted.")

    csv_path, summary_path, dir_plan_path, dir_deleted_path = resolve_plan_paths(
        log_dir, source_root, args.plan_file
    )

    planned_dir_count = 0
    if args.dry_run:
        parquet_paths = list(iter_source_parquets(source_root))
        empty_records = detect_empty_files(parquet_paths)
        write_csv(empty_records, csv_path)
        if args.prune_empty_dirs:
            pretend_removed = _normalize_paths(rec.path for rec in empty_records)
            prunable_dirs = find_empty_server_dirs(source_root, pretend_removed)
            planned_dir_count = len(prunable_dirs)
            write_dir_plan(prunable_dirs, dir_plan_path)
            print(f"Dry-run would prune {planned_dir_count} empty server directories.")
            print(f"Directory plan: {dir_plan_path}")
        write_summary(
            summary_path,
            scanned=len(parquet_paths),
            empty=len(empty_records),
            deleted=0,
            pruned_dirs=planned_dir_count,
        )
        print(f"Dry-run complete. Empty files: {len(empty_records)}")
        print(f"CSV plan:     {csv_path}")
        print(f"Summary JSON: {summary_path}")
        if not args.yes_delete:
            return 0

    if args.yes_delete:
        plan_records = load_plan(csv_path)
        removed = delete_files(plan_records)
        pruned_dir_paths: List[Path] = []
        if args.prune_empty_dirs:
            pruned_dir_paths = prune_empty_server_dirs(source_root)
            write_dir_plan(pruned_dir_paths, dir_deleted_path)
            print(f"Removed {len(pruned_dir_paths)} empty server directories.")
            print(f"Directory removal log: {dir_deleted_path}")
        write_summary(
            summary_path,
            scanned=len(plan_records),
            empty=len(plan_records),
            deleted=len(removed),
            pruned_dirs=len(pruned_dir_paths),
        )
        print(f"Deleted {len(removed)} files (plan entries: {len(plan_records)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
