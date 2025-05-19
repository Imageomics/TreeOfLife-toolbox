Step-by-step guide:

1. **Download**
    1. Set up `distributed-downloader` package and prepare all the scripts for the download as described in
       the [instruction](https://github.com/Imageomics/distributed-downloader/blob/9ef8b0d297f7a868fac31b2b9c3d5f3aa5533472/docs/scripts_README.md).
    2. Create a config file for the download,
       use [safe_download_config.yaml](../config/tree_of_life_200M/safe_download_config.yaml) as a base. Some of
       the values were already filled and should not be adjusted unless you know what you are doing.
    3. Run the download script:
       ```bash
       distributed_downloader configs/safe_download_config.yaml
       ```
    4. It might take several weeks to complete, so rerun step 3 when the number of workers is depleted or the download
       is interrupted. Completeness can be checked with `inner_checkpoint_file` file that is created in the download
       location.

2. **Postprocess**
    1. After download is completed, you will need to transfer the downloaded images into `TreeOfLife200M` dataset structure.
       To do this you can run `data_transfer` tool from this repository:
        ```bash
       tree_of_life_toolbox {config path} data_transfer
        ```
       where `{config path}` is the path to the config file for the job configuration.