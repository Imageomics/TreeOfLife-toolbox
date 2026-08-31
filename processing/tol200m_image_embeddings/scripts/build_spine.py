#!/usr/bin/env python
"""Build the metadata SPINE: the published bioclip-2_float16 config minus `emb`,
plus a global `_pos` row index.

The spine defines the row set, row order, and 15 metadata columns shared by ALL
embedding configs (it equals the TreeOfLife-200M catalog restricted to its
233,055,986 rows, in the published global taxonomic sort). Deriving it from the
published files (instead of re-sorting the catalog) preserves row order exactly:
the sort keys have large tie groups, so a fresh sort would not reproduce the
published order within ties.

One output file per published file, physical row order preserved (pyarrow
read_table/write_table are order-preserving). `_pos` is the global row index
computed from cumulative per-file row counts.

Usage:  python build_spine.py <PUBLISHED_DIR> <SPINE_DIR> [num_procs]
        PUBLISHED_DIR: local copy of the published bioclip-2_float16/ files
        SPINE_DIR:     output dir (train-NNNNN-of-NNNNN.parquet, uuid + 15 meta + _pos)
Run via a SLURM cpu job.
"""
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

EXPECT_ROWS = 233_055_986


def build_one(args):
    src, dst, offset = args
    t = pq.read_table(src).drop_columns(["emb"])
    pos = pa.array(np.arange(offset, offset + t.num_rows, dtype=np.int64))
    t = t.append_column("_pos", pos)
    pq.write_table(t, dst, compression="zstd", row_group_size=50_000)
    return t.num_rows


def main():
    published_dir, spine_dir = sys.argv[1], sys.argv[2]
    nprocs = int(sys.argv[3]) if len(sys.argv) > 3 else int(os.environ.get("SLURM_CPUS_PER_TASK", 16))
    os.makedirs(spine_dir, exist_ok=True)

    srcs = sorted(glob.glob(os.path.join(published_dir, "train-*.parquet")))
    if not srcs:
        sys.exit(f"no train-*.parquet under {published_dir}")

    # cumulative offsets from actual per-file row counts (footer reads, fast)
    counts = [pq.read_metadata(f).num_rows for f in srcs]
    offsets = np.concatenate(([0], np.cumsum(counts)[:-1]))
    jobs = [(src, os.path.join(spine_dir, os.path.basename(src)), int(off))
            for src, off in zip(srcs, offsets)]

    with ProcessPoolExecutor(max_workers=nprocs) as ex:
        total = sum(ex.map(build_one, jobs))

    status = "OK" if total == EXPECT_ROWS else "MISMATCH"
    print(f"spine rows: {total:,} across {len(srcs)} files (expect {EXPECT_ROWS:,}) -> {status}")
    if total != EXPECT_ROWS:
        sys.exit(1)


if __name__ == "__main__":
    main()
