Step-by-step guide:

1. **Download**
    1. Download bioscan from the official Google Disk location [here](https://drive.google.com/drive/u/1/folders/1Jc57eKkeiYrnUBc9WlIp-ZS_L1bVlT-0).
    2. Unzip the downloaded file to the location of your choice.
2. **Postprocess**
    1. After download is completed, you will need to transfer the downloaded images into `TreeOfLife200M` dataset structure.
       To do this you can run `bioscan_data_transfer` tool from this repository:
        ```bash
       tree_of_life_toolbox {config path} bioscan_data_transfer
        ```
       where `{config path}` is the path to the config file for the job configuration.