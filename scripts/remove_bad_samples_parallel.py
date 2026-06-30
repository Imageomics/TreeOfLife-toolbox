#!/usr/bin/env python3
"""
Remove specified UUIDs (and sidecars) from shard tar files using parallel workers.

Incremental behavior:
- Reads from --shards (input directory)
- Writes scrubbed shards to --shards-scrubbed (required output directory)
- If scrubbed output already exists for a shard, that shard is skipped
- Missing input shards are skipped

Usage:
  python scripts/remove_bad_samples_parallel.py \
      --shards /path/to/shards_to_scrub \
      --shards-scrubbed /path/to/shards_scrubbed \
      --map missing-taxa.csv \
      --workers 8 \
      --max-inflight 32
"""

from __future__ import annotations

import argparse
import csv
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

SIDE_EXTS = {
    "jpg",
    "com.txt",
    "common_name.txt",
    "sci.txt",
    "sci_com.txt",
    "scientific_name.txt",
    "taxon.txt",
    "taxonTag.txt",
    "taxonTag_com.txt",
    "taxon_com.txt",
    "taxonomic_name.txt",
}


def iter_shard_groups(csv_path: Path) -> Iterator[Tuple[int, List[str]]]:
    """
    Stream (shard_id, [uuid...]) groups from a CSV sorted by shard_id.
    """
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        current_sid: Optional[int] = None
        uuids: List[str] = []

        for row in reader:
            sid = int(row["shard_id"])
            u = row["uuid"]

            if current_sid is None:
                current_sid = sid

            if sid != current_sid:
                yield current_sid, uuids
                current_sid = sid
                uuids = [u]
            else:
                uuids.append(u)

        if current_sid is not None and uuids:
            yield current_sid, uuids


def scrub_shard(
    src_shard_path: Path,
    dst_shard_path: Path,
    uuids: Iterable[str],
) -> Tuple[str, int, int]:
    """
    Rewrite shard tar, dropping members whose basename matches <uuid>.<side_ext>.
    Writes scrubbed output to dst_shard_path using temp + atomic rename.
    Returns (shard_name, removed_count, kept_count).
    """
    if not src_shard_path.exists():
        return (src_shard_path.name, 0, 0)

    dst_shard_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = dst_shard_path.with_name(dst_shard_path.name + ".tmp")

    drop_prefixes = set(uuids)
    removed = 0
    kept = 0

    with tarfile.open(src_shard_path, "r") as src, tarfile.open(tmp_out, "w") as dst:
        for member in src:
            filename = os.path.basename(member.name)

            dot = filename.find(".")
            if dot != -1:
                base = filename[:dot]
                if base in drop_prefixes:
                    suffix = filename[dot + 1 :]
                    if suffix in SIDE_EXTS:
                        removed += 1
                        continue

            data = src.extractfile(member) if member.isfile() else None
            dst.addfile(member, data)
            kept += 1

    os.replace(tmp_out, dst_shard_path)
    return (src_shard_path.name, removed, kept)


def _worker(args: Tuple[Path, Path, List[str]]) -> Tuple[str, int, int]:
    src_shard_path, dst_shard_path, uuids = args
    return scrub_shard(src_shard_path, dst_shard_path, uuids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove bad UUIDs from shard tar files (parallel, incremental).")
    parser.add_argument("--shards", required=True, type=Path, help="Input directory containing shard-XXXXX.tar files.")
    parser.add_argument("--shards-scrubbed", required=True, type=Path, help="Output directory for scrubbed shards.")
    parser.add_argument("--map", required=True, type=Path, help="CSV with columns uuid,shard_id (sorted by shard_id recommended).")
    parser.add_argument("--workers", type=int, default=8, help="Parallel shard workers (I/O heavy; tune to cluster limits).")
    parser.add_argument("--max-inflight", type=int, default=32, help="Bound queued shard jobs to limit FS pressure.")
    args = parser.parse_args()

    shards_dir: Path = args.shards
    scrubbed_dir: Path = args.shards_scrubbed
    workers = max(1, args.workers)
    max_inflight = max(workers, args.max_inflight)

    scrubbed_dir.mkdir(parents=True, exist_ok=True)

    inflight = []
    total_removed = 0
    shards_processed = 0
    shards_skipped_missing = 0
    shards_skipped_already = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for shard_id, uuids in iter_shard_groups(args.map):
            shard_name = f"shard-{shard_id:05d}.tar"
            src_shard_path = shards_dir / shard_name
            dst_shard_path = scrubbed_dir / shard_name

            if not src_shard_path.exists():
                shards_skipped_missing += 1
                continue

            # Incremental: if output already exists, skip
            if dst_shard_path.exists():
                shards_skipped_already += 1
                continue

            fut = executor.submit(_worker, (src_shard_path, dst_shard_path, uuids))
            inflight.append(fut)

            if len(inflight) >= max_inflight:
                name, removed, kept = inflight.pop(0).result()
                total_removed += removed
                shards_processed += 1
                print(f"[INFO] {name}: removed={removed} kept={kept}")

        for fut in as_completed(inflight):
            name, removed, kept = fut.result()
            total_removed += removed
            shards_processed += 1
            print(f"[INFO] {name}: removed={removed} kept={kept}")

    print(
        "[DONE] "
        f"shards_processed={shards_processed} "
        f"shards_skipped_missing={shards_skipped_missing} "
        f"shards_skipped_already={shards_skipped_already} "
        f"members_removed={total_removed}"
    )


if __name__ == "__main__":
    main()
