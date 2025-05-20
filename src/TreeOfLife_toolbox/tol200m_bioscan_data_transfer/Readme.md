new params to config:
- `provenance_path` - path to a procenance parquet file
- `path_to_tol_folder` - path to where you save tol data folder (a.k.a. `<output_dir>/data`)
- `bioscan_image_folder` - path to where you unziped bioscan images (up to `origianal_full` folder)

clarification on existing params in config:
- `path_to_input` - path to `tsv` file from BIOSCAN
- `urls_folder` - will contain a partitioned BIOSCAN tsv file