# Tools for TreeOfLife dataset

This repository contains tools for the TreeOfLife dataset. They were created on the basis of distributed downloader.

## Currently, the following tools are available:

- [column_name_change](src/DD_tools/column_name_change) - changes the names of columns in the dataset
- [column_name_change_lila_fix](src/DD_tools/column_name_change_lila_fix) - changes the names of columns in the
  dataset (were created to fix the bug in Lila BC dataset)
- [data_merging](src/DD_tools/data_merging) - used to filter out duplicated data in freshly downloaded datasets from
  existing ones (deduplication based on `hashsum`)
- [data_transfer](src/DD_tools/data_transfer) - transfers data from one downloaded dataset to TreeOfLife dataset
- [eol_rename](src/DD_tools/eol_rename) - were used to change `source_id` from `EOL content ID` to "`EOL content ID`_
  `EOL page ID`" (the change was discarded later)
- [fathom_net_crop](src/DD_tools/fathom_net_crop) - used to crop FathomNet images to the bounding box sizes
- [fathom_net_crop_fix](src/DD_tools/fathom_net_crop_fix) - used to crop FathomNet images to the bounding box sizes
  (were created to fix the bug in FathomNet dataset)
- [filter_out_by_uuid](src/DD_tools/filter_out_by_uuid) - tool to filter out using table of Tree of life `uuid`s
- [lila_bc_filtering](src/DD_tools/lila_bc_filtering) - tool for filtering Lila BC dataset (based on some processed csv
  table)
- [lila_extra_noaa_processing](src/DD_tools/lila_extra_noaa_processing) - tool for processing Lila Extra NOAA dataset in
  TreeOfLife format
- [lila_separation_multilable_filtering](src/DD_tools/lila_separation_multilable_filtering) - tool to extract
  multilabels data from Lila BC dataset and duplicate images for each label
- [lila_separation_single_label_filtering](src/DD_tools/lila_separation_single_label_filtering) - tool to extract
  single label data from Lila BC dataset
- [mam_ansp_fix](src/DD_tools/mam_ansp_fix) - tool to fix the bug in `man ansp` server (gbif source)
- [research_filtering](src/DD_tools/research_filtering) - tool to filter out data from TreeOfLife datasets
- [transfer_and_type_change](src/DD_tools/transfer_and_type_change) - tool to transfer data (and change types) from one
  place to another on research storage (it transfers only 10Tb per day, to not overload the back-up system)

## How to use the tools

To use the tools, you will need to create a `config.yaml` file, schema can be found
in [tools.yaml](src/DD_tools/main/config_templates/tools.yaml).

To run the tool, use the following command:

```bash
python src/DD_tools/main/main.py <config_path> <tool_name> [--OPTIONS]
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

To create a new tool, you will need to create a new folder in `src/DD_tools/` and add the following files:

- `__init__.py` - empty file
- `classes.py` - file with the classes for the tool

In `classes.py` you will need to

1. Create a class for each step of the tool (filtering, scheduling, runner). Make sure that the class inherits from the
   base class for the step and that class names are unique.
2. Register the classes with their respective registry (FilterRegistery, SchedulerRegistry, RunnerRegistry) using the
   `register` decorator.
3. Add tool folder to `__init__.py` file in `src/DD_tools/` folder.

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