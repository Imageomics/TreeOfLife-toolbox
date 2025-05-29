# Tools for TreeOfLife dataset

This repository contains tools used in creating the [TreeOfLife-200M dataset](https://huggingface.co/datasets/imageomics/TreeOfLife-200M). They were created on the basis of [distributed-downloader](https://github.com/Imageomics/distributed-downloader), which was used for downloading all the images. Step-by-step instructions to download all of the images are provided in [docs/README](docs/README.md#treeoflife200m-dataset-download-guide).

## Installation Instructions

Currently, only the portion of this package that is required for downloading all the images in the TreeOfLife-200M dataset is installable. Our other processing code, which is _not_ required to download a copy of the dataset, is provided and described further in the [processing/](processing) directory.[^1]
[^1]: This processing code will be reworked into installable modules as appropriate over the coming months. 

### Pip installation

1. Install Python 3.10 or 3.11
2. Install MPI, any MPI should work, tested with OpenMPI and IntelMPI. Installation instructions can be found on
   official websites:
    - [OpenMPI](https://docs.open-mpi.org/en/v5.0.x/installing-open-mpi/quickstart.html)
    - [IntelMPI](https://www.intel.com/content/www/us/en/docs/mpi-library/developer-guide-linux/2021-6/installation.html)
3. Install the required package:
    - For development:

      ```commandline
      pip install -e .[dev]
      ```

### Scripts creation

After installation, you need to create scripts for the tools. Follow this instructions [here](./docs/scripts_README.md)

## Currently, the following tools are available

- [column_name_change](./src/TreeOfLife_toolbox/column_name_change) - changes the names of columns in the dataset
- [column_name_change_lila_fix](./src/TreeOfLife_toolbox/column_name_change_lila_fix) - changes the names of columns in the
  dataset (were created to fix the bug in Lila BC dataset)
- [data_merging](./src/TreeOfLife_toolbox/data_merging) - used to filter out duplicated data in freshly downloaded datasets from
  existing ones (deduplication based on `hashsum`)
- [data_transfer](./src/TreeOfLife_toolbox/data_transfer) - transfers data from one downloaded dataset to TreeOfLife dataset
- [eol_rename](./src/TreeOfLife_toolbox/eol_rename) - were used to change `source_id` from `EOL content ID` to "`EOL content ID`_
  `EOL page ID`" (the change was discarded later)
- [fathom_net_crop](./src/TreeOfLife_toolbox/fathom_net_crop) - used to crop FathomNet images to the bounding box sizes
- [fathom_net_crop_fix](./src/TreeOfLife_toolbox/fathom_net_crop_fix) - used to crop FathomNet images to the bounding box sizes
  (were created to fix the bug in FathomNet dataset)
- [filter_out_by_uuid](./src/TreeOfLife_toolbox/filter_out_by_uuid) - tool to filter out using table of Tree of life `uuid`s
- [lila_bc_filtering](./src/TreeOfLife_toolbox/lila_bc_filtering) - tool for filtering Lila BC dataset (based on some processed csv
  table)
- [lila_extra_noaa_processing](./src/TreeOfLife_toolbox/lila_extra_noaa_processing) - tool for processing Lila Extra NOAA dataset in
  TreeOfLife format
- [lila_separation_multilable_filtering](./src/TreeOfLife_toolbox/lila_separation_multilable_filtering) - tool to extract
  multilabels data from Lila BC dataset and duplicate images for each label
- [lila_separation_single_label_filtering](./src/TreeOfLife_toolbox/lila_separation_single_label_filtering) - tool to extract
  single label data from Lila BC dataset
- [mam_ansp_fix](./src/TreeOfLife_toolbox/mam_ansp_fix) - tool to fix the bug in `man ansp` server (gbif source)
- [research_filtering](./src/TreeOfLife_toolbox/research_filtering) - tool to filter out data from TreeOfLife datasets
- [transfer_and_type_change](./src/TreeOfLife_toolbox/transfer_and_type_change) - tool to transfer data (and change types) from one
  place to another on research storage (it transfers only 10Tb per day, to not overload the back-up system)
- [tol200m_bioscan_data_tranfer](./src/TreeOfLife_toolbox/tol200m_bioscan_data_tranfers) - tool to transfer data from
  Bioscan dataset to TreeOfLife dataset
- [tol200m_fathom_net_crop](./src/TreeOfLife_toolbox/tol200m_fathom_net_crop) - tool to crop FathomNet images to the bounding box sizes for the TreeOfLife dataset

## How to use the tools

To use the tools, you will need to create a `config.yaml` file, schema can be found
in [example.yaml](./config/example_config.yaml).

To run the tool, use the following command:

```bash
tree_of_life_toolbox <config_path> <tool_name> [--OPTIONS]
```

- `<config_path>` - path to the `config.yaml` file (either absolute or relative)
- `<tool_name>` - name of the tool to run
- `[--OPTIONS]` - optional arguments for the tool:
  - `--reset_filtering` - basically a full reset. It resets the first step of the tool - filtering, however, since all
      the following steps depend on the filtering step, it will reset them as well
  - `--reset_scheduling` - resets the scheduling step (useful when you want to change the number of runners/nodes per
      runner)
  - `--reset_runners` - resets the runners, meaning they will start from scratch
  - `--tool_name_override` - used to disable the tool name check

## How to create a new tool

To create a new tool, you will need to create a new folder in `src/TreeOfLife_toolbox/` and add the following files:

- `__init__.py` - empty file
- `classes.py` - file with the classes for the tool

In `classes.py` you will need to

1. Create a class for each step of the tool (filtering, scheduling, runner). Make sure that the class inherits from the
   base class for the step and that class names are unique.
2. Register the classes with their respective registry (FilterRegistery, SchedulerRegistry, RunnerRegistry) using the
   `register` decorator.
3. Add tool folder to `__init__.py` file in `src/TreeOfLife_toolbox/` folder.

The following base classes are available:

- filtering step:
  - `FilterToolBase` - bare minimum class for the filtering step
  - `SparkFilterToolBase` - base class for the filtering step using Spark, it automatically creates a Spark session
      and
      has some additional methods for working with Spark
  - `PythonFilterToolBase` - base class for the filtering step using Python, it can automatically traverse the *
      *downloaded** dataset
- scheduling step:
  - `SchedulerToolBase` - bare minimum class for the scheduling step
  - `DefaultScheduler` - base class for the scheduling step. It can perform "standard" scheduling for the runners, you
      will need to specify the `schema` for it.
- runner step:
  - `RunnerToolBase` - bare minimum class for the runner step
  - `MPIRunnerTool` - base class for the MPI based runner step, it can automatically initialize the MPI environment,
      read the schedule and call the `apply_filter` method on the separate chunks from schedule sequentially.
      You will need to implement the `apply_filter` method in your class.
  - `FilterRunnerTool` - inherits from `MPIRunnerTool` and can perform "standard" filtering based on `UUID`s. Works
      only with **downloaded** dataset schema.

## Recommended Citation

If using the [TreeOfLife-200M dataset](https://huggingface.co/datasets/imageomics/TreeOfLife-200M), please cite this repo, the dataset, and our paper.

```
@software{Kopanev_TreeOfLife-toolbox_2025,
  author = {Kopanev, Andrei and Zhang, Net and Gu, Jianyang and Stevens, Samuel and Thompson, Matthew J and Campolongo, Elizabeth G},
  license = {MIT},
  month = may,
  title = {{TreeOfLife-toolbox}},
  url = {https://github.com/Imageomics/TreeOfLife-toolbox},
  version = {0.2.0-beta},
  year = {2025}
}
```

```
@dataset{treeoflife_200m,
  title = {{T}ree{O}f{L}ife-200{M}}, 
  author = {Jianyang Gu and Samuel Stevens and Elizabeth G Campolongo and Matthew J Thompson and Net Zhang and Jiaman Wu and Andrei Kopanev and Zheda Mai and Alexander E. White and James Balhoff and Wasila M Dahdul and Daniel Rubenstein and Hilmar Lapp and Tanya Berger-Wolf and Wei-Lun Chao and Yu Su},
  year = {2025},
  url = {https://huggingface.co/datasets/imageomics/TreeOfLife-200M},
  doi = {},
  publisher = {Hugging Face}
}
```

```
@article{gu2025bioclip,
  title = {{B}io{CLIP} 2: Emergent Properties from Scaling Hierarchical Contrastive Learning}, 
  author = {Jianyang Gu and Samuel Stevens and Elizabeth G Campolongo and Matthew J Thompson and Net Zhang and Jiaman Wu and Andrei Kopanev and Zheda Mai and Alexander E. White and James Balhoff and Wasila M Dahdul and Daniel Rubenstein and Hilmar Lapp and Tanya Berger-Wolf and Wei-Lun Chao and Yu Su},
  year = {2025},
  eprint = {},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV}
}
 ```

Also consider citing [GBIF](https://gbif.org), [BIOSCAN-5M](https://github.com/bioscan-ml/BIOSCAN-5M), [EOL](https://eol.org), and [FathomNet](https://database.fathomnet.org/fathomnet/#/):

```
@misc{GBIF,
  title = {{GBIF} Occurrence Download},
  author = {GBIF.org},
  doi = {10.15468/DL.BFV433},
  url = {https://doi.org/10.15468/dl.bfv433},
  keywords = {GBIF, biodiversity, species occurrences},
  publisher = {The Global Biodiversity Information Facility},
  month = {May},
  year = {2024},
  copyright = {Creative Commons Attribution Non Commercial 4.0 International}
}

```

```
@inproceedings{gharaee2024bioscan5m,
    title={{BIOSCAN-5M}: A Multimodal Dataset for Insect Biodiversity},
    author={Zahra Gharaee and Scott C. Lowe and ZeMing Gong and Pablo Millan Arias
        and Nicholas Pellegrino and Austin T. Wang and Joakim Bruslund Haurum
        and Iuliia Zarubiieva and Lila Kari and Dirk Steinke and Graham W. Taylor
        and Paul Fieguth and Angel X. Chang
    },
    booktitle={NeurIPS},
    editor={A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
    pages={36285--36313},
    publisher={Curran Associates, Inc.},
    year={2024},
    volume={37},
}
```

```
@misc{eol,
  author = {{Encyclopedia of Life (EOL)}},
  url = {https://eol.org},
  note = {Accessed August 2024}
}
```

```
@article{katija_fathomnet_2022,
	title = {{FathomNet}: {A} global image database for enabling artificial intelligence in the ocean},
  author = {Katija, Kakani and Orenstein, Eric and Schlining, Brian and Lundsten, Lonny and Barnard, Kevin and Sainz, Giovanna and Boulais, Oceane and Cromwell, Megan and Butler, Erin and Woodward, Benjamin and Bell, Katherine L. C.},
	journal = {Scientific Reports},
	volume = {12},
	number = {1},
	pages = {15914},
	issn = {2045-2322},
	shorttitle = {{FathomNet}},
	url = {https://www.nature.com/articles/s41598-022-19939-2},
	doi = {10.1038/s41598-022-19939-2},
	month = sep,
	year = {2022},
}
