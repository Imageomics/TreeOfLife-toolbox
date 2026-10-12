#!/usr/bin/env python
"""Standardize raw embeddings into one published config: spine metadata + one
model's embeddings, in spine order, in the published parquet contract.

Pipeline: DuckDB inner-joins the spine (uuid + 15 metadata + _pos) to the raw
per-rank embedding parquets by uuid (raw rows not in the catalog are dropped),
orders by _pos, and streams 50K-row batches. A thread pool of writers then, per
output file: optionally L2-normalizes each embedding in float32 (--normalize;
used for bioclip-2.5, NOT for bioclip-2), casts to fixed_size_list<float16>[dim]
(lossless for normalized values), and writes the contract parquet:
ZSTD-3, 50K-row row groups, statistics + page indexes, 9 sorting_columns
(nulls_first=False), train-NNNNN-of-MMMMM.parquet with rows_per_file rows.

rows_per_file targets ~500 MB/file: 500_000 for 512/768-dim, 250_000 for 1024-dim.

Usage:
  python standardize_config.py <model_col> <config_name> <rows_per_file> --dim D \
      --spine "<SPINE_DIR>/train-*.parquet" \
      --emb-glob "<EMB_OUT>/model=<m>/source=*/embeddings/rank_*/*.parquet" \
      [--emb-glob "<EMB_OUT>/model=<m>/source=gbif/task_*/embeddings/rank_*/*.parquet"] \
      --out <STAGE_DIR> [--normalize] [--threads T] [--write-threads W] [--mem-limit G]

Run via a SLURM cpu job; the global join+sort spills through DuckDB temp space
(~470 GB peak for the 1024-dim config -> hugemem-class node; ~340 GB for 512-dim).
"""
import argparse
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BATCH = 50_000  # == published row group size

SORT_KEYS = ["source_dataset", "kingdom", "phylum", "class", "order", "family",
             "genus", "species", "common_name"]
META = ["source_dataset", "source_id", "kingdom", "phylum", "class", "order", "family",
        "genus", "species", "scientific_name", "common_name", "publisher",
        "basisOfRecord", "identifier", "img_type"]


def transform_emb(chunked, dim, normalize, emb_type):
    """list<float32> column -> (optionally L2-normalized) fixed_size_list<float16>[dim]."""
    arr = chunked.combine_chunks() if isinstance(chunked, pa.ChunkedArray) else chunked
    fsl = arr.cast(pa.list_(pa.float32(), dim))
    mat = np.reshape(fsl.values.to_numpy(zero_copy_only=False), (len(arr), dim))
    if normalize:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # keep all-zero vectors as-is
        mat = mat / norms
    flat = pa.array(mat.astype(np.float16).ravel())
    return pa.FixedSizeListArray.from_arrays(flat, dim).cast(emb_type)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_col", help="raw embedding column, e.g. emb_bioclip_2_5")
    ap.add_argument("config_name", help="published config dir name")
    ap.add_argument("rows_per_file", type=int)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--spine", required=True, help="spine glob (from build_spine.py)")
    ap.add_argument("--emb-glob", action="append", required=True,
                    help="raw embedding parquet glob(s); repeatable")
    ap.add_argument("--out", required=True, help="staging root; writes <out>/<config_name>/")
    ap.add_argument("--normalize", action="store_true",
                    help="L2-normalize embeddings in float32 before the float16 cast")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 32)))
    ap.add_argument("--write-threads", type=int, default=0, help="0 -> max(4, threads-4)")
    ap.add_argument("--mem-limit", default="400GB")
    ap.add_argument("--tmp", default=os.environ.get("TMPDIR", "/tmp"), help="DuckDB spill dir")
    args = ap.parse_args()

    assert args.rows_per_file % BATCH == 0, f"rows_per_file must be a multiple of {BATCH}"
    batches_per_file = args.rows_per_file // BATCH
    write_threads = args.write_threads or max(4, args.threads - 4)
    out_dir = os.path.join(args.out, args.config_name)
    os.makedirs(out_dir, exist_ok=True)

    emb_type = pa.list_(pa.field("element", pa.float16()), args.dim)  # published emb type

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{args.mem_limit}'; SET threads={args.threads}; "
                f"SET temp_directory='{args.tmp}'; SET preserve_insertion_order=true;")

    glist = "['" + "','".join(args.emb_glob) + "']"
    meta_sel = ", ".join(f's."{c}"' for c in META)
    base = f"""
      WITH sp AS (SELECT * FROM read_parquet('{args.spine}')),
           emb AS (SELECT uuid, "{args.model_col}" AS emb FROM read_parquet({glist}))
    """
    total_expected = con.execute(
        base + "SELECT COUNT(*) FROM sp s JOIN emb e USING (uuid)").fetchone()[0]
    n_files = (total_expected + args.rows_per_file - 1) // args.rows_per_file
    print(f"expected {total_expected:,} rows -> {n_files} files @ {args.rows_per_file:,}; "
          f"normalize={args.normalize}; duckdb threads={args.threads}, "
          f"write threads={write_threads}", flush=True)

    reader = con.execute(base + f"""
      SELECT s.uuid, e.emb AS emb, {meta_sel}
      FROM sp s JOIN emb e USING (uuid) ORDER BY s._pos
    """).fetch_record_batch(rows_per_batch=BATCH)

    raw_schema = None
    sc = None

    def write_file(idx, raw_table):
        # transform runs here so normalization + cast parallelize across the pool
        ei = raw_table.schema.get_field_index("emb")
        emb = transform_emb(raw_table.column(ei), args.dim, args.normalize, emb_type)
        table = raw_table.set_column(ei, pa.field("emb", emb_type), emb)
        path = os.path.join(out_dir, f"train-{idx:05d}-of-{n_files:05d}.parquet")
        pq.write_table(table, path, compression="zstd", compression_level=3,
                       write_statistics=True, write_page_index=True,
                       sorting_columns=sc, row_group_size=BATCH)

    pool = ThreadPoolExecutor(max_workers=write_threads)
    pending = deque()               # in-flight writes (each holds a ~500 MB table)
    max_inflight = write_threads + 4
    buf = []
    fi = total = 0

    def flush():
        nonlocal buf, fi
        if not buf:
            return
        table = pa.Table.from_batches(buf, schema=raw_schema)
        while len(pending) >= max_inflight:   # backpressure vs the DuckDB stream
            pending.popleft().result()
        pending.append(pool.submit(write_file, fi, table))
        fi += 1
        buf = []

    for batch in reader:
        if raw_schema is None:
            raw_schema = batch.schema
            ei = raw_schema.get_field_index("emb")
            out_schema = raw_schema.set(ei, pa.field("emb", emb_type))
            sc = pq.SortingColumn.from_ordering(out_schema, [(k, "ascending") for k in SORT_KEYS])
        buf.append(batch)
        total += batch.num_rows
        if len(buf) == batches_per_file:
            flush()
    flush()  # trailing partial file

    for f in pending:
        f.result()
    pool.shutdown()

    print(f"wrote {total:,} rows across {fi} files -> {out_dir}", flush=True)
    assert total == total_expected, f"ROW MISMATCH {total} != {total_expected}"
    assert fi == n_files, f"FILE COUNT MISMATCH {fi} != {n_files}"


if __name__ == "__main__":
    main()
