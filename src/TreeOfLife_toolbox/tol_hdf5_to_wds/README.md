# tol_hdf5_to_wds

Converts the hybrid Tree of Life dataset (HDF5 images + Parquet metadata) into
[WebDataset](https://github.com/webdataset/webdataset) `.tar` shards that mirror
the taxonomy-rich text sidecars produced by the previous Parquet-based pipeline.

## Pipeline steps

1. **Filter**: Scans the converted dataset for `*_metadata.parquet` files,
   enriches each row with the matching `*_images.h5` path, and writes shard-sized
   parquet partitions under `tools/tol_hdf5_to_wds/shard_metadata`.
2. **Scheduler**: Assigns shard identifiers to MPI ranks and writes
   `schedule.csv` for the toolbox runner scripts.
3. **Runner**: For every scheduled shard, loads the shard metadata, reads
   WebP-compressed bytes from the referenced HDF5 file, converts them to JPEG
   (with an optional resize), generates ten taxonomy/common-name text prompts,
   and emits `shard-XXXXX.tar` archives via `webdataset.TarWriter`.

## Configuration highlights

Add a `tol_hdf5_to_wds` block to the standard toolbox config template:

```yaml
tol_hdf5_to_wds:
  shard_size: 10000                 # rows per shard
  shard_limit: 0                    # optional cap, 0 disables
  metadata_glob: "**/*_metadata.parquet"
  tar_output_root: "/fs/.../wds"    # defaults to tools folder if omitted
  resize_size: 224                  # square resize in pixels (0 keeps original)
  lookup_table_path: "/fs/.../subset.parquet"   # optional UUID list (CSV or Parquet)
  lookup_path_column: "path"                    # only needed for CSVs with raw paths
  taxa_glob: "/fs/.../annotations/source=*/*.parquet"  # external taxonomy parquet
  # If the lookup includes an `hdf5_path` column it will override the auto-resolved path.
```

Set `path_to_input` to the root of the converted dataset (where the
`*_metadata.parquet` and `*_images.h5` files live) and `path_to_output_folder`
to the working directory for toolbox artifacts.

If `lookup_table_path` is provided, it can be either CSV or Parquet and must
contain at least a `uuid` column. Include a column named `hdf5_path` to override
the derived file path, or a `path` column (configurable via
`lookup_path_column`) when providing CSVs that still reference the original
`data_*.parquet` names. Only UUIDs listed in the lookup are converted.

Run the tool via the standard CLI:

```bash
CONFIG_PATH=config/tol_hdf5_to_wds_example.yaml \
tree_of_life_toolbox config/tol_hdf5_to_wds_example.yaml tol_hdf5_to_wds
```

Each shard sample contains the JPEG plus ten text prompts:
`scientific_name.txt`, `taxonomic_name.txt`, `sci.txt`, `taxon.txt`,
`taxonTag.txt`, `common_name.txt`, `com.txt`, `sci_com.txt`,
`taxon_com.txt`, and `taxonTag_com.txt`. Files ending in `_com` append
the resolved common name (or `Unknown` when no vernacular exists) to
their prompt text.
