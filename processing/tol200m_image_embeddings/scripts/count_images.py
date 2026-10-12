#!/usr/bin/env python
"""Count images per h5 file for a source, in parallel, and write a parquet table.

Work (embedding time) is per-image, so this per-file image count is what the array
driver balances chunks on. Reads the paired *_metadata.parquet footer (fast, no data
read); falls back to the h5 'images' group size if the parquet is missing.

Usage:  python count_images.py <target_dir> <out_parquet> [num_procs]
Output: <out_parquet> with columns [h5_path: str, n_images: int64], sorted by h5_path.
Run via a SLURM cpu job.
"""
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq


def count_one(h5_path: str) -> tuple:
    parquet = h5_path.replace("_images.h5", "_metadata.parquet")
    try:
        return h5_path, pq.read_metadata(parquet).num_rows
    except Exception:
        try:
            import h5py
            with h5py.File(h5_path, "r") as f:
                return h5_path, len(f["images"].keys())
        except Exception as e:
            sys.stderr.write(f"WARN could not count {h5_path}: {e}\n")
            return h5_path, -1


def main() -> None:
    target = sys.argv[1]
    out_parquet = sys.argv[2]
    nprocs = int(sys.argv[3]) if len(sys.argv) > 3 else (os.cpu_count() or 8)

    h5s = sorted(
        glob.glob(os.path.join(target, "**", "*.h5"), recursive=True)
        + glob.glob(os.path.join(target, "**", "*.hdf5"), recursive=True)
    )
    sys.stderr.write(f"Counting {len(h5s)} h5 files with {nprocs} procs...\n")

    with ProcessPoolExecutor(max_workers=nprocs) as ex:
        rows = list(ex.map(count_one, h5s, chunksize=16))  # kept in sorted(h5s) order

    paths = [h for h, _ in rows]
    counts = [n for _, n in rows]
    bad = sum(1 for n in counts if n < 0)
    total = sum(n for n in counts if n >= 0)

    table = pa.table(
        {"h5_path": pa.array(paths, pa.string()),
         "n_images": pa.array(counts, pa.int64())}
    )
    tmp = out_parquet + ".tmp"
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, out_parquet)
    sys.stderr.write(
        f"Wrote {out_parquet}: {len(rows)} files, {total:,} images"
        + (f", {bad} FAILED\n" if bad else "\n")
    )
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
