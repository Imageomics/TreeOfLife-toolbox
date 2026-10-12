#!/usr/bin/env python
"""Validate a standardized config before upload.

Checks:
  1. file / row / distinct-uuid counts (rows == distinct == spine rows, 0 dups)
  2. parquet contract on first/middle/last file: emb type fixed_size_list<float16>[dim],
     50K row groups, 9 sorting_columns, page indexes, file size
  3. FULL uuid-at-position order match against the spine
  4. embedding recompute on a random row-group sample: normalize (if --normalize)
     + float16-cast the RAW embeddings for the sampled uuids and byte-compare
     against the stored values

Usage:
  python validate_config.py <config_dir> <model_col> --dim D \
      --spine "<SPINE_DIR>/train-*.parquet" \
      --emb-glob "<EMB_OUT>/.../rank_*/*.parquet" [--emb-glob ...] \
      [--normalize] [--sample-files 3] [--threads T] [--mem-limit G]

Run via a SLURM cpu job (step 3 sorts 233M rows twice; step 4 joins a sample
against the raw embeddings).
"""
import argparse
import glob
import os
import random

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config_dir")
    ap.add_argument("model_col")
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--spine", required=True)
    ap.add_argument("--emb-glob", action="append", required=True)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--sample-files", type=int, default=3)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 32)))
    ap.add_argument("--mem-limit", default="200GB")
    ap.add_argument("--tmp", default=os.environ.get("TMPDIR", "/tmp"))
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.config_dir, "*.parquet")))
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{args.mem_limit}'; SET threads={args.threads}; "
                f"SET temp_directory='{args.tmp}';")
    ok = True

    # 1. counts
    n = con.sql(f"SELECT COUNT(*) FROM read_parquet({files})").fetchone()[0]
    d = con.sql(f"SELECT COUNT(DISTINCT uuid) FROM read_parquet({files})").fetchone()[0]
    sp_n = con.sql(f"SELECT COUNT(*) FROM read_parquet('{args.spine}')").fetchone()[0]
    print(f"1. files={len(files)} rows={n:,} distinct_uuid={d:,} dups={n-d:,} spine={sp_n:,}",
          flush=True)
    ok &= (n == d == sp_n)

    # 2. contract spot-check
    for f in (files[0], files[len(files) // 2], files[-1]):
        m = pq.ParquetFile(f).metadata
        sch = pq.ParquetFile(f).schema_arrow
        rg = m.row_group(0)
        emb_t = sch.field("emb").type
        print(f"2. {os.path.basename(f)}: emb={emb_t} rows={m.num_rows:,} "
              f"n_rg={m.num_row_groups} rg0={rg.num_rows} sort={len(rg.sorting_columns)} "
              f"pageidx={rg.column(0).has_offset_index} {os.path.getsize(f)/1e6:.0f}MB",
              flush=True)
        ok &= (str(emb_t) == f"fixed_size_list<element: halffloat>[{args.dim}]"
               and rg.num_rows == 50_000 and len(rg.sorting_columns) == 9)

    # 3. full order vs spine
    mism = con.sql(f"""
      WITH o AS (SELECT row_number() OVER () rn, uuid FROM read_parquet({files})),
           s AS (SELECT row_number() OVER (ORDER BY _pos) rn, uuid
                 FROM read_parquet('{args.spine}'))
      SELECT COUNT(*) FROM o JOIN s USING(rn) WHERE o.uuid <> s.uuid""").fetchone()[0]
    print(f"3. FULL order: uuid-at-position mismatches vs spine = {mism:,}", flush=True)
    ok &= (mism == 0)

    # 4. recompute check on sampled files (first row group of each)
    glist = "['" + "','".join(args.emb_glob) + "']"
    rng = random.Random(0)
    checked = bad = 0
    for f in rng.sample(files, min(args.sample_files, len(files))):
        stored = pq.ParquetFile(f).read_row_group(0, columns=["uuid", "emb"])
        uuids = stored.column("uuid").to_pylist()
        con.execute("CREATE OR REPLACE TEMP TABLE want(uuid VARCHAR)")
        con.executemany("INSERT INTO want VALUES (?)", [(u,) for u in uuids])
        raw = con.sql(f"""
            SELECT r.uuid, r."{args.model_col}" AS emb
            FROM read_parquet({glist}) r JOIN want USING (uuid)""").arrow()
        raw_map = dict(zip(raw.column("uuid").to_pylist(),
                           np.reshape(raw.column("emb").combine_chunks()
                                      .cast(pa.list_(pa.float32(), args.dim))
                                      .values.to_numpy(zero_copy_only=False),
                                      (raw.num_rows, args.dim))))
        got = np.reshape(stored.column("emb").combine_chunks()
                         .cast(pa.list_(pa.float16(), args.dim))
                         .values.to_numpy(zero_copy_only=False),
                         (stored.num_rows, args.dim))
        for i, u in enumerate(uuids):
            if u not in raw_map:
                bad += 1
                continue
            v = raw_map[u]
            if args.normalize:
                nrm = np.linalg.norm(v)
                v = v / nrm if nrm != 0 else v
            bad += not np.array_equal(v.astype(np.float16), got[i])
            checked += 1
    print(f"4. recompute check: {checked:,} rows sampled, mismatches={bad:,}", flush=True)
    ok &= (bad == 0)

    print("\nVERDICT:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
