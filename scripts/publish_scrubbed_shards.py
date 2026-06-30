#!/usr/bin/env python3
"""
Publish scrubbed shards back into the primary shards directory.

Design goals:
- Only overwrite shards/ with scrubbed versions from shards_scrubbed/.
- Do NOT touch shards_to_scrub/ (your backup copies).
- Avoid clobbering a shard that is still being generated: only publish onto a "complete" source shard.
- Publish via temp + atomic rename so Globus never sees a partial file.
- Do NOT preserve mtime (important for Globus sync level 2: mtime-based).

Usage:
  python scripts/publish_scrubbed_shards.py \
    --shards /path/to/shards \
    --shards-scrubbed /path/to/shards_scrubbed
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


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


def tar_counts_via_tar_tf(tar_path: Path) -> Optional[Dict[str, int]]:
    """
    Return counts of jpg + each sidecar suffix for tar_path by streaming `tar tf`.
    Returns None if `tar tf` fails.
    """
    counts: Dict[str, int] = {sfx: 0 for sfx in SIDE_TXT_SUFFIXES}
    counts["jpg"] = 0

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

    return None if rc != 0 else counts


def is_complete_source_shard(tar_path: Path) -> bool:
    """
    Only publish onto a shard in shards/ if it still looks like a complete original shard.
    (Prevents racing the generator / avoids clobbering partials.)
    """
    counts = tar_counts_via_tar_tf(tar_path)
    if counts is None:
        return False

    if counts["jpg"] != EXPECTED_PER_TYPE:
        return False

    for sfx in SIDE_TXT_SUFFIXES:
        if counts[sfx] != EXPECTED_PER_TYPE:
            return False

    return True


def copy_atomic_no_mtime(src: Path, dst: Path) -> None:
    """
    Copy src -> dst through a temp file in dst's directory, then atomic rename.
    Does NOT preserve metadata/mtime (good for Globus sync level 2).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)   # intentionally not copy2()
    os.replace(tmp, dst)


def main() -> None:
    p = argparse.ArgumentParser(description="Publish scrubbed shard tars back into shards/ safely.")
    p.add_argument("--shards", required=True, type=Path, help="Primary shards/ directory to overwrite.")
    p.add_argument("--shards-scrubbed", required=True, type=Path, help="Directory containing scrubbed shard outputs.")
    p.add_argument("--limit", type=int, default=0, help="Optional: stop after publishing N shards (0 = no limit).")
    args = p.parse_args()

    shards_dir: Path = args.shards
    scrubbed_dir: Path = args.shards_scrubbed

    published = 0
    skipped_missing_src = 0
    skipped_incomplete_src = 0
    skipped_no_change = 0

    for scrubbed_tar in sorted(scrubbed_dir.glob("shard-*.tar")):
        name = scrubbed_tar.name
        dst_tar = shards_dir / name

        if not dst_tar.exists():
            # If the generator hasn't created it yet, don't create it here.
            skipped_missing_src += 1
            continue

        # Never clobber a shard that doesn't look "complete" yet.
        if not is_complete_source_shard(dst_tar):
            skipped_incomplete_src += 1
            continue

        # If shards/ already matches scrubbed by size, assume already published.
        # (Cheap heuristic; avoids unnecessary rewrite/mtime churn.)
        try:
            if dst_tar.stat().st_size == scrubbed_tar.stat().st_size:
                skipped_no_change += 1
                continue
        except FileNotFoundError:
            # race: try again next loop
            continue

        copy_atomic_no_mtime(scrubbed_tar, dst_tar)
        published += 1

        if published % 25 == 0:
            print(
                f"[INFO] published={published} no_change={skipped_no_change} "
                f"missing_src={skipped_missing_src} incomplete_src={skipped_incomplete_src}"
            )

        if args.limit and published >= args.limit:
            break

    print(
        "[DONE] "
        f"published={published} "
        f"skipped_no_change={skipped_no_change} "
        f"skipped_missing_src={skipped_missing_src} "
        f"skipped_incomplete_src={skipped_incomplete_src}"
    )


if __name__ == "__main__":
    main()

