#!/usr/bin/env python3
"""
Copy shard tar files implicated by missing-taxa.csv into a staging directory,
but ONLY when the source shard appears complete.

Completeness criteria (intrinsic):
- `tar tf` succeeds (tar structure readable)
- exactly 10,000 *.jpg entries
- exactly 10,000 entries for each required TXT sidecar suffix

Staging/update semantics:
- If a scrubbed output already exists (in --scrubbed-dir), skip entirely.
- If destination shard exists AND same byte size as source, skip.
- Otherwise, (re)copy the source shard into destination (atomic temp + rename),
  but only if the source passes completeness.

Usage:
  python scripts/copy_implicated_shards.py \
    --map missing-taxa.csv \
    --src-dir /path/to/shards \
    --dst-dir /path/to/shards_to_scrub \
    --scrubbed-dir /path/to/shards_scrubbed
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Set

from tqdm import tqdm


SIDE_TXT_SUFFIXES: List[str] = [
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

EXPECTED_PER_TYPE = 10_000


def shard_name(shard_id: int) -> str:
    return f"shard-{shard_id:05d}.tar"


def count_data_rows(path: Path) -> int:
    with path.open("rb") as f:
        return max(0, sum(1 for _ in f) - 1)


def load_unique_shard_ids(csv_path: Path, total_rows: int | None = None) -> Set[int]:
    shard_ids: Set[int] = set()
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in tqdm(reader, total=total_rows, desc="Scanning missing-taxa.csv", unit="rows"):
            shard_ids.add(int(row["shard_id"]))
    return shard_ids


def tar_counts_via_tar_tf(tar_path: Path) -> Dict[str, int] | None:
    counts: Dict[str, int] = {sfx: 0 for sfx in SIDE_TXT_SUFFIXES}
    counts["jpg"] = 0
    counts["total"] = 0

    proc = subprocess.Popen(
        ["tar", "tf", str(tar_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            name = line.strip()
            if not name:
                continue
            counts["total"] += 1

            if name.endswith(".jpg"):
                counts["jpg"] += 1
                continue

            for sfx in SIDE_TXT_SUFFIXES:
                if name.endswith("." + sfx):
                    counts[sfx] += 1
                    break
    finally:
        proc.stdout.close()
        rc = proc.wait()

    if rc != 0:
        return None
    return counts


def is_complete_shard(tar_path: Path) -> bool:
    counts = tar_counts_via_tar_tf(tar_path)
    if counts is None:
        return False

    if counts["jpg"] != EXPECTED_PER_TYPE:
        return False

    for sfx in SIDE_TXT_SUFFIXES:
        if counts[sfx] != EXPECTED_PER_TYPE:
            return False

    return True


def copy_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copy2(src, tmp)      # preserve mtime for staging comparisons if you care later
    os.replace(tmp, dst)


def same_size(src: Path, dst: Path) -> bool:
    try:
        return src.stat().st_size == dst.stat().st_size
    except FileNotFoundError:
        return False


def main() -> None:
    p = argparse.ArgumentParser(
        description="Copy implicated shard tars into a staging directory (only when complete)."
    )
    p.add_argument("--map", required=True, type=Path, help="missing-taxa.csv with columns uuid,shard_id.")
    p.add_argument("--src-dir", required=True, type=Path, help="Directory containing shard-XXXXX.tar files.")
    p.add_argument("--dst-dir", required=True, type=Path, help="Destination directory for shards to scrub.")
    p.add_argument(
        "--scrubbed-dir",
        required=True,
        type=Path,
        help="Directory containing scrubbed shard outputs; if shard exists here, it is skipped.",
    )
    p.add_argument("--limit", type=int, default=0, help="Optional: stop after copying N shards (0 = no limit).")
    args = p.parse_args()

    total_rows = count_data_rows(args.map)
    shard_ids = load_unique_shard_ids(args.map, total_rows=total_rows)
    shard_id_list = sorted(shard_ids)

    copied = 0
    updated = 0

    skipped_missing = 0
    skipped_incomplete = 0
    skipped_same_size = 0
    skipped_already_scrubbed = 0

    for sid in tqdm(shard_id_list, desc="Checking/copying shards", unit="shards"):
        src = args.src_dir / shard_name(sid)
        dst = args.dst_dir / src.name
        scrubbed = args.scrubbed_dir / src.name

        # Robust: if already scrubbed, never stage again (prevents noise if scrubbed published back to src-dir)
        if scrubbed.exists():
            skipped_already_scrubbed += 1
            continue

        if not src.exists():
            skipped_missing += 1
            continue

        # If destination exists and size matches, skip.
        if dst.exists() and same_size(src, dst):
            skipped_same_size += 1
            continue

        # Only copy/update if source is intrinsically complete.
        if not is_complete_shard(src):
            skipped_incomplete += 1
            continue

        was_update = dst.exists()
        copy_atomic(src, dst)
        if was_update:
            updated += 1
        else:
            copied += 1

        if (copied + updated) % 25 == 0:
            print(
                f"[INFO] copied={copied} updated={updated} "
                f"missing={skipped_missing} incomplete={skipped_incomplete} "
                f"same_size_skip={skipped_same_size} already_scrubbed_skip={skipped_already_scrubbed}"
            )

        if args.limit and (copied + updated) >= args.limit:
            break

    print(
        "[DONE] "
        f"unique_shards={len(shard_id_list)} "
        f"copied={copied} "
        f"updated={updated} "
        f"skipped_missing={skipped_missing} "
        f"skipped_incomplete={skipped_incomplete} "
        f"skipped_same_size={skipped_same_size} "
        f"skipped_already_scrubbed={skipped_already_scrubbed}"
    )


if __name__ == "__main__":
    main()

