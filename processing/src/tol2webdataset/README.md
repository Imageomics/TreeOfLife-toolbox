# Tree of Life To Webdataset Converter

This script converts `TreeOfLife` formatted dataset into `webdataset` format using `Spark` and `MPI`.

## How to run

1. Create a config file following the structure of [tol2webdataset.yaml](config_templates/tol2webdataset.yaml).
2. Run [tol2webdataset.py](tol2webdataset.py) with the path to the config file (created in step 1) as an argument.
3. Wait for all the jobs to be completed.
4. Check `t2w_checkpoint.yaml` file in `<path_to_output_folder>` folder, if "completed" is
    - `true` - the conversion is done.
    - `false` - do step 2-4 again.

Additional arguments for [tol2webdataset.py](tol2webdataset.py) (only use one of them at a time):

- `--reset_filtering` - will reset the converter and start from the begging (**all files will be overwritten**).
- `--reset_scheduling` - will reset `scheduling` and `runner` step (required if you change the number of `nodes` or `worker per node` in config _after_ the `scheduling` was already completed).
- `--reset_runners` - will reset the progress of the conversion and start the runners again without rescheduling.

## How it works

The job is completed in 4 steps (performed by running `tol2webdataset.py`):

1. `filtering` - using Spark, prepare all the metadata that will be needed for the conversion (e.g. uuids, taxa information) and separate this data into shards with `shard_size` number of elements and `shard_count_limit` total number of shards (these parameters can be configured in config file [tol2webdataset.yaml](config_templates/tol2webdataset.yaml)).
2. `scheduling` - python script that distributes work between future MPI processes (it creates a schedule for each worker; ex: worker 3 will convert shards 1,3,5, etc.). Number of workers is equal to `max_nodes_per_runner * workers_per_node` (these parameters can be configured in config file [tol2webdataset.yaml](config_templates/tol2webdataset.yaml)).
3. `runner` - using MPI for job distribution, starts configured number of workers to perform the conversion (usually includes several jobs of 2-3 hours each, so they can be started faster). Number of workers is equal to `max_nodes_per_runner * workers_per_node` (these parameters can be configured in config file [tol2webdataset.yaml](config_templates/tol2webdataset.yaml)).
4. `verification` - python script to check whether the process is completed or if additional jobs are needed.

### File structure

- [tol2webdataset.py](tol2webdataset.py) - main file that is used to launch the whole process.
- [filter.py](filter.py) - script that uses `PySpark` to prepare metadata for the conversion.
- [scheduler.py](scheduler.py) - script that distributes the work between future MPI processes.
- [runner.py](runner.py) - script that performs the conversion using `MPI`.
- [verification.py](verification.py) - script that checks whether the conversion is completed or if additional jobs are needed. It is performed in the following way. Every worker writes verification file in `verification_folder` (folder name can be changed in the config), which consists of identifiers of entries that were completed (in this case it is `shard_id`). Verifier combines all completed entries into one table and compares it with scheduled table. If any entries left unprocessed, verification reports that work is not completed yet, otherwise it changes the `completed` flag in `t2w_checkpoint.yaml`, which indicates that the job was completed, and blocks future workers.
