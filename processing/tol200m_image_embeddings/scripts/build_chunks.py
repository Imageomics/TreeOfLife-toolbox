#!/usr/bin/env python
"""Build size-balanced array chunks from a per-file image-count parquet.

Greedy-packs the h5 files into N chunks each holding ~target_images, so every array
task does ~the same amount of work (embedding time is per-image) regardless of the
wide per-file variance in gbif. Deterministic: same counts + target -> same chunks,
so resume keeps the same task<->file mapping. Idempotent: refuses to overwrite an
existing chunk set (delete chunks_manifest.parquet to rebuild).

Usage:  python build_chunks.py <counts_parquet> <out_root> <target_images>
Writes: <out_root>/task_<N>/filelist.txt   and   <out_root>/chunks_manifest.parquet
Prints: NTASKS (number of chunks) to stdout.
"""
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    counts_parquet, out_root, target = sys.argv[1], sys.argv[2], int(sys.argv[3])
    manifest = os.path.join(out_root, "chunks_manifest.parquet")

    if os.path.exists(manifest):  # already built -> reuse (keeps task<->file mapping stable)
        print(pq.read_metadata(manifest).num_rows)
        return

    t = pq.read_table(counts_parquet).to_pydict()
    files = [(p, n) for p, n in zip(t["h5_path"], t["n_images"]) if n > 0]
    total = sum(n for _, n in files)
    ntasks = max(1, round(total / target))
    per = total / ntasks  # even target per chunk

    chunks, cur, acc = [[]], 0, 0
    for path, n in files:
        chunks[cur].append((path, n))
        acc += n
        if acc >= per and cur < ntasks - 1:
            chunks.append([])
            cur += 1
            acc = 0
    chunks = [c for c in chunks if c]  # drop any trailing empty

    rows_id, rows_nf, rows_ni = [], [], []
    for i, ch in enumerate(chunks):
        d = os.path.join(out_root, f"task_{i}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "filelist.txt"), "w") as fh:
            fh.write("\n".join(p for p, _ in ch) + "\n")
        rows_id.append(i)
        rows_nf.append(len(ch))
        rows_ni.append(sum(n for _, n in ch))

    table = pa.table({"task_id": rows_id, "n_files": rows_nf, "n_images": rows_ni})
    pq.write_table(table, manifest, compression="zstd")
    sys.stderr.write(
        f"Built {len(chunks)} chunks from {len(files)} files ({total:,} images); "
        f"per-chunk images min={min(rows_ni):,} max={max(rows_ni):,} target={int(per):,}\n"
    )
    print(len(chunks))


if __name__ == "__main__":
    main()
