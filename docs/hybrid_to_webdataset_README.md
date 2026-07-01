# Create a WebDataset from the hybrid HDF5 + Parquet format

This guide covers packaging the Tree of Life dataset, stored in the hybrid format of WebP images in HDF5 (`*_images.h5`) alongside Parquet metadata (`*_metadata.parquet`), into [WebDataset](https://github.com/webdataset/webdataset) `.tar` shards for model training. It is implemented by the [`tol_hybrid_to_wds`](../src/TreeOfLife_toolbox/tol_hybrid_to_wds/README.md) tool.

## What it produces

One `shard-XXXXX.tar` per `shard_size` samples. Each sample is keyed by `uuid` and contains a JPEG plus ten text-prompt sidecars used for BioCLIP-style training:

| | |
|---|---|
| `scientific_name.txt`, `common_name.txt`, `taxonomic_name.txt` | raw label values |
| `sci.txt`, `com.txt`, `taxon.txt`, `taxonTag.txt` | `a photo of {value}.` captions |
| `sci_com.txt`, `taxon_com.txt`, `taxonTag_com.txt` | the caption with ` with common name {common}.` appended |

When no vernacular common name is available, the common name falls back to the scientific name (without author citation).

## Inputs

All three are joined by `uuid`:

1. **Hybrid dataset**: the root holding `*_images.h5` (WebP bytes under an `/images/<uuid>` group) and the paired `*_metadata.parquet`.
2. **Resolved taxonomy**: Parquet providing `scientific_name`, `common_name`, and the `kingdom` through `species` ranks per `uuid` (the text-label source).
3. **Lookup table**: Parquet/CSV with at least a `uuid` column; only listed UUIDs are converted. The `.h5` path is auto-derived as the sibling `*_images.h5` of each `*_metadata.parquet`, so no path column is needed; an optional `hdf5_path` (or `h5_file`) column overrides it.

## Prerequisites

Install the toolbox (see the top-level [installation instructions](../README.md#installation-instructions)):

```bash
pip install -e .
```

Option A (single node) needs only the Python dependencies that installs. Option B (distributed) runs on a cluster, and its Slurm scripts encode site-specific choices you must adapt:

- An **MPI** implementation with `mpi4py` installed against it. The workers run under MPI; if `mpi4py` is built against a different MPI than the one loaded at runtime, they fail to load it.
- A **Spark** runtime for the filter stage.
- The **partition** and **module** names in `scripts/tools_*.slurm`, and the virtual environment those scripts activate (by default the repo-root `.venv`, so install the toolbox there or point the scripts at your environment).

Edit `scripts/tools_*.slurm` to load your cluster's MPI and Spark modules, target your partition, and activate your environment. As committed they target one cluster (OSC Cardinal) as a worked example to replace with your own.

## Option A: single node (quick / small subsets)

Best for testing or modest UUID lists. No Spark/MPI required:

```bash
python tol_hybrid_to_wds.py \
  --input-root /path/to/hybrid_dataset \
  --lookup     /path/to/uuid_lookup.parquet \
  --taxa-glob  "/path/to/resolved_taxa/source=*/*.parquet" \
  --output-dir /path/to/wds \
  --resize 224 --shard-size 10000
```

(Generate a `uuid`-to-`h5` lookup for a UUID subset with [`generate_hdf5_lookup.py`](../generate_hdf5_lookup.py).)

## Option B: distributed (full dataset, Slurm)

The toolbox runner reads one config and submits the whole pipeline (Spark filter, scheduler, MPI workers, verifier) with the right Slurm dependencies. Start from [`config/tol_hybrid_to_wds_example.yaml`](../config/tol_hybrid_to_wds_example.yaml) and set, for your run:

- `account` and `tools_parameters` (nodes, workers, CPUs) for your allocation;
- `path_to_input` (the hybrid dataset root) and `path_to_output_folder` (the toolbox working directory);
- in the `tol_hybrid_to_wds` block: `tar_output_root`, `resize_size`, `taxa_glob` (required for the taxonomy/common-name prompts), and `lookup_table_path` (a parquet/CSV with a `uuid` column selecting which images to convert).

See the [tool README](../src/TreeOfLife_toolbox/tol_hybrid_to_wds/README.md) for the full key reference.

Then run the pipeline with a single command:

```bash
tree_of_life_toolbox config/tol_hybrid_to_wds_my_run.yaml tol_hybrid_to_wds
```

The runner reads the config, exports the environment it implies (account, node/worker counts, output folders), and submits the four chained stages. The **filter** discovers the `*_metadata.parquet` files under `path_to_input`, derives each sibling `*_images.h5`, joins the taxonomy, and restricts to the lookup UUIDs; the **scheduler** assigns shards to MPI ranks; the **workers** read the WebP bytes, resize to JPEG, write the ten prompts, and emit the tars; the **verifier** marks completion. The underlying Slurm scripts are described in [scripts_README](scripts_README.md).

## Validating output

`validate_tars.py` checks shard integrity, and `wds_taxoncom_audit.py` audits the taxonomy/common-name prompts across shards.
