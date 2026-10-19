import argparse
import glob
import logging
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta

from lance_writer import Shard, commit, write_fragment, write_manifest

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def convert_shards(tars: list[str], output: str, workers: int) -> list[Shard]:
    shards = []
    # spawn, not fork: lance is not fork-safe
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
        for done, shard in enumerate(pool.map(write_fragment, tars, [output] * len(tars)), start=1):
            status = f"{shard.rows:,} rows" if shard.rows else "empty, skipped"
            logging.info(f"[{done}/{len(tars)}] {shard.name}: {status}")
            if shard.rows:
                shards.append(shard)
    return shards


def main():
    parser = argparse.ArgumentParser(
        description="Convert a split of TreeOfLife webdataset tar shards into a Lance dataset.",
        epilog="example: python wds_to_lance.py /path/to/wds/train /path/to/lance/train.lance",
    )
    parser.add_argument("split_dir", help="Directory holding the split's *.tar shards.")
    parser.add_argument("output", help="Lance dataset to create, e.g. <out-dir>/train.lance.")
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count()),
                        help="Shards to convert at the same time (default: %(default)s).")
    args = parser.parse_args()

    output = args.output.rstrip("/")
    if os.path.exists(output):
        raise SystemExit(f"{output} already exists; delete it to convert again")
    tars = sorted(glob.glob(os.path.join(args.split_dir, "*.tar")))
    if not tars:
        raise SystemExit(f"No .tar files in {args.split_dir}")

    logging.info(f"Converting {len(tars)} tar files from {args.split_dir} into {output} "
                 f"with {args.workers} workers")
    start = time.monotonic()
    try:
        shards = convert_shards(tars, output, args.workers)
    except Exception:
        logging.exception(f"Conversion stopped and nothing was committed; "
                          f"delete {output} if it exists and run again")
        raise SystemExit(1)
    if not shards:
        raise SystemExit(f"All {len(tars)} tar files in {args.split_dir} are empty")
    commit(output, shards)
    manifest = write_manifest(output, args.split_dir, shards)

    rows = sum(shard.rows for shard in shards)
    elapsed = timedelta(seconds=round(time.monotonic() - start))
    logging.info(f"Done in {elapsed}: {rows:,} rows from {len(tars)} tar files "
                 f"({len(tars) - len(shards)} empty)")
    logging.info(f"Dataset: {output}")
    logging.info(f"Manifest: {manifest} (read by the training loader; keep it next to the dataset)")


if __name__ == "__main__":
    main()
