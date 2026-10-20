# WebDataset To Lance Converter

This script converts `TreeOfLife` webdataset shards into a `Lance` dataset for BioCLIP 2 training, using a pool of worker processes.

## How to run

1. Install the packages in [requirements_wds2lance.txt](../../requirements_wds2lance.txt).
2. Run [wds_to_lance.py](wds_to_lance.py) with the directory of the split's `*.tar` shards and the Lance dataset to create (e.g. `<out-dir>/train.lance`) as arguments. On Slurm, set the environment and data paths in [wds2lance.slurm](../../scripts/wds2lance/wds2lance.slurm) and submit it from `processing/` with `REPO_ROOT=$PWD sbatch --account=<account> --cpus-per-task=32 scripts/wds2lance/wds2lance.slurm`.
3. Wait for the conversion to be completed (it logs a line per shard and a summary at the end). If it stops with an error, delete the output and do step 2 again.
4. Train BioCLIP 2 by pointing `--train-data` and `--val-data` at the `.lance` paths and keeping `--dataset-type auto`. Keep `<output>.manifest.json` next to the dataset, the training loader reads it.

Additional arguments for [wds_to_lance.py](wds_to_lance.py):

- `--workers` - number of shards converted at the same time (default: one per available CPU, at most 32).

## How it works

The job is completed in 3 steps (performed by running `wds_to_lance.py`):

1. `conversion` - a pool of worker processes converts each tar shard into one Lance fragment: one row per sample and one column per tar member, bytes unchanged (`sci.txt` becomes column `sci`), plus `uuid` (the sample key), `source_shard` and `source_index`. Every sample must have the 11 members of the TreeOfLife-10M layout (`jpg` and ten `.txt` files), otherwise the conversion stops with an error; empty tar files are skipped. Shards written by [tol2webdataset](../tol2webdataset) have only 6 of these members and are not supported.
2. `commit` - the fragments are committed in tar order as one Lance dataset at the output path.
3. `manifest` - `<output>.manifest.json` is written next to the dataset with the row count and each shard's rows and offset, which the BioCLIP 2 training loader reads.

### File structure

- [wds_to_lance.py](wds_to_lance.py) - main file that is used to launch the whole process.
- [lance_writer.py](lance_writer.py) - script that writes one shard as one Lance fragment, commits the dataset and writes the manifest.
- [tar_reader.py](tar_reader.py) - script that reads the samples out of one tar shard.
