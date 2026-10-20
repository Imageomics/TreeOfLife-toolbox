# WebDataset to Lance Converter

Converts a split of TreeOfLife webdataset tar shards into a [Lance](https://lancedb.github.io/lance/) dataset for
training BioCLIP 2. Each sample becomes one row and each tar member one column, with the bytes unchanged.

## Input

A directory of `*.tar` shards. Every sample must have the 11 members of the TreeOfLife-10M layout: `jpg` plus
`scientific_name`, `common_name`, `taxonomic_name`, `sci`, `com`, `taxon`, `sci_com`, `taxon_com`, `taxonTag` and
`taxonTag_com` (`.txt`). Otherwise the conversion stops with an error naming the shard, the sample key and the
missing or unexpected members. Empty tar files are skipped. Shards written by [`tol2webdataset`](../tol2webdataset)
have only 6 of these members and are not supported.

## Output

- `<output>.lance`: one column per member (`sci.txt` becomes `sci`), plus `uuid` (the sample key), `source_shard`
  and `source_index`. Rows are in tar order.
- `<output>.lance.manifest.json`: the row count and each shard's rows and offset. The training loader needs it, so
  keep it next to the dataset.

To train BioCLIP 2 from it, point `--train-data` and `--val-data` at the `.lance` paths and keep `--dataset-type auto`
(the default), so the LAION replay data in `--continual-data` stays on tar shards.

## Run

Install `processing/requirements_wds2lance.txt`, then:

```bash
python wds_to_lance.py <split-dir> <out-dir>/train.lance --workers 32
```

Shards are converted in parallel, `--workers` at a time (default: one per available CPU, at most 32). The
converter logs a line per shard, in tar order, and a summary at the end:

```text
2026-10-19 10:00:00,001 - INFO - Converting 3 tar files from wds/train into lance/train.lance with 32 workers
2026-10-19 10:00:01,402 - INFO - [1/3] shard-000000.tar: 1,000 rows
2026-10-19 10:00:01,403 - INFO - [2/3] shard-000001.tar: empty, skipped
2026-10-19 10:00:01,405 - INFO - [3/3] shard-000002.tar: 1,000 rows
2026-10-19 10:00:01,622 - INFO - Done in 0:00:02: 2,000 rows from 3 tar files (1 empty)
2026-10-19 10:00:01,622 - INFO - Dataset: lance/train.lance
2026-10-19 10:00:01,622 - INFO - Manifest: lance/train.lance.manifest.json (read by the training loader; keep it next to the dataset)
```

If a run is interrupted or stops with an error, delete the output and run it again. `python wds_to_lance.py --help`
shows the options and an example.

On Slurm, set `PATH/TO/ENVIRONMENT`, `BASE_DIR` and the data paths in
[`scripts/wds2lance/wds2lance.slurm`](../../scripts/wds2lance/wds2lance.slurm), then submit it from `processing/` with
your account and the number of CPUs to use:

```bash
REPO_ROOT=$PWD sbatch --account=<account> --cpus-per-task=32 scripts/wds2lance/wds2lance.slurm
```

## Code

- [`wds_to_lance.py`](wds_to_lance.py): the command line. It finds the shards, converts them in parallel and reports
  progress, then commits the dataset and writes the manifest.
- [`lance_writer.py`](lance_writer.py): the schema, writing one shard as one Lance fragment, the commit and the
  manifest.
- [`tar_reader.py`](tar_reader.py): reads the samples out of one tar shard.

`wds_to_lance.py` imports `lance_writer`, which imports `tar_reader`, by plain module name, so keep all three files
in the same directory.
