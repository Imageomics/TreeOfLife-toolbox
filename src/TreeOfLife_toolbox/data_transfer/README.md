Additional values in the config:

- `dst_image_folder`: The destination folder where the image (successes) files will be copied.
- `dst_error_folder`: The destination folder where the error files will be copied.

Expected input structure:

- `path_to_output_folder` + `images_folder`: contains all the images to be copied.
- the above folder is partitioned into subfolders, with the first level having "server_name=*" and the second level
  having "partition_id=*".
- Each subfolder contains:
    - `successes.parquet`: a parquet file with the downloaded images (follows `distributed-downloader` structure of
      `successes` files).
    - `errors.parquet`: a parquet file with the errors (follows `distributed-downloader` structure of `errors` files).
    - `completed`: an empty file indicating that the download is complete. (automatically created by the
      `distributed-downloader` when the download is finished for this partition).

It will create the following structure in the destination folder:

- `dst_image_folder`: the destination folder where the image (successes) files will be copied.
- `dst_error_folder`: the destination folder where the error files will be copied.
- Both will be partitioned into subfolders with only one layer `server=*`
- Each successes subfolder will contain:
    - `data_{uuid}.parquet`: a parquet file with the downloaded images (follows `distributed-downloader` structure of
      `successes` files).
- Each errors subfolder will contain:
    - `errors_{uuid}.parquet`: a parquet file with the errors (follows `distributed-downloader` structure of `errors`
      files).