#!/usr/bin/env python3
"""Convert a split of TreeOfLife webdataset tar shards into a Lance dataset for BioCLIP 2 training.

Each sample becomes one row and each tar member one column, bytes unchanged (sci.txt becomes column sci),
plus uuid (the sample key), source_shard and source_index. Shards are converted in parallel, one Lance
fragment each, and committed in tar order. <output>.manifest.json, which the bioclip-2 loader reads, is
written next to the dataset.
"""
import argparse
import glob
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta

from lance_writer import Shard, commit, write_fragment, write_manifest

EXAMPLE = """example:
  python wds_to_lance.py /path/to/webdataset/train /path/to/lance/train.lance --workers 32"""


def main() -> None:
    args = parse_args()
    output = args.output.rstrip("/")
    if os.path.exists(output):
        raise SystemExit(f"error: {output} already exists; delete it to convert again")
    tars = sorted(glob.glob(os.path.join(args.split_dir, "*.tar")))
    if not tars:
        raise SystemExit(f"error: no .tar files in {args.split_dir}; pass the directory that holds the split's shards")

    print(f"Converting {len(tars)} tar files from {args.split_dir} into {output} with {args.workers} workers",
          flush=True)
    start = time.monotonic()
    try:
        shards = convert_shards(tars, output, args.workers)
    except Exception:
        traceback.print_exc()
        raise SystemExit(f"error: the conversion stopped (see above), so nothing was committed; "
                         f"delete {output} if it exists, then run again")
    if not shards:
        raise SystemExit(f"error: all {len(tars)} tar files in {args.split_dir} are empty; nothing to convert")
    commit(output, shards)
    manifest = write_manifest(output, args.split_dir, shards)

    rows = sum(shard.rows for shard in shards)
    elapsed = timedelta(seconds=round(time.monotonic() - start))
    print(f"Done in {elapsed}: {rows:,} rows from {len(tars)} tar files ({len(tars) - len(shards)} empty)")
    print(f"  dataset:  {output}")
    print(f"  manifest: {manifest} (the training loader reads it; keep it next to the dataset)")


def convert_shards(tars: list[str], output: str, workers: int) -> list[Shard]:
    """Write one fragment per tar in parallel, printing a line per tar in tar order as results come in.

    Returns the non-empty shards in tar order, which is the order they are committed in.
    """
    shards = []
    # spawn, not fork: lance is not fork-safe
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
        # map yields results in tar order, and cancels the shards still queued if one fails
        for done, shard in enumerate(pool.map(write_fragment, tars, [output] * len(tars)), start=1):
            status = f"{shard.rows:,} rows" if shard.rows else "empty, skipped"
            print(f"[{done}/{len(tars)}] {shard.name}: {status}", flush=True)
            if shard.rows:
                shards.append(shard)
    return shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, epilog=EXAMPLE,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("split_dir", help="directory holding the split's *.tar shards")
    parser.add_argument("output", help="Lance dataset to create, e.g. <out-dir>/train.lance; must not exist yet")
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count()), metavar="N",
                        help="shards to convert at the same time (default: one per CPU, at most 32; here %(default)s)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
