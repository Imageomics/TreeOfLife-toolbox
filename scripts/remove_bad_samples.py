#!/usr/bin/env python
"""
Remove specified UUIDs from shard tar files.

Usage:
    python scripts/remove_bad_samples.py \
        --shards /path/to/shards \
        --map missing-taxa.csv \
        [--backup-dir /path/to/backups]
"""

from __future__ import annotations

import argparse
import csv
import os
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


SIDE_EXTS = [
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
]


def load_uuid_map(csv_path: Path) -> Dict[int, List[str]]:
    shard_map: Dict[int, List[str]] = defaultdict(list)
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            shard_map[int(row["shard_id"])].append(row["uuid"])
    return shard_map


def scrub_shard(shard_path: Path, uuids: Iterable[str], backup_dir: Path | None) -> None:
    if not shard_path.exists():
        print(f"[WARN] Shard not found: {shard_path}")
        return

    tmp_path = shard_path.with_suffix(".tmp")
    drop_prefixes = set(uuids)
    drop_members = {f"{u}.{ext}" for u in drop_prefixes for ext in SIDE_EXTS}

    with tarfile.open(shard_path, "r") as src, tarfile.open(tmp_path, "w") as dst:
        for member in src:
            base = member.name.split(".", 1)[0]
            if base in drop_prefixes and member.name in drop_members:
                continue
            data = src.extractfile(member) if member.isfile() else None
            dst.addfile(member, data)

    if backup_dir:
        backup_dir.mkdir(parents=True, exist_ok=True)
        shard_backup = backup_dir / shard_path.name
        os.replace(shard_path, shard_backup)
    else:
        shard_path.unlink()
    os.replace(tmp_path, shard_path)


def main():
    parser = argparse.ArgumentParser(description="Remove bad UUIDs from shard tar files.")
    parser.add_argument("--shards", required=True, type=Path, help="Directory containing shard-XXXXX.tar files.")
    parser.add_argument("--map", required=True, type=Path, help="CSV with columns uuid,shard_id.")
    parser.add_argument("--backup-dir", type=Path, help="Optional dir to store original tar backups.")
    args = parser.parse_args()

    shard_map = load_uuid_map(args.map)
    for shard_id, uuids in shard_map.items():
        shard_name = f"shard-{shard_id:05d}.tar"
        shard_path = args.shards / shard_name
        print(f"[INFO] Scrubbing {len(uuids)} samples from {shard_name}")
        scrub_shard(shard_path, uuids, args.backup_dir)


if __name__ == "__main__":
    main()
