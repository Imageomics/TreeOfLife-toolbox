# Processing Tools

As noted in the root `README`, this repository's overall structure is still in progress, as the contents of this directory are being worked into the larger tool package.

## Data Directory

The [`data/`](data) directory contains the support set embeddings used for museum specimen image filtering (as detailed in Appendix I.2.1 of [our paper]()).

## Docs Directory

This folder contains a series of files named `requirements_<processing-step>.txt`, detailing the requisit packages for said processing step (e.g., `requirements_batch_camera_trap.txt` for camera trap image processing).

## Notebooks

The `notebooks/` directory houses the notebooks used to perform their respective processing steps (most, if not all, of which were adapted to modules run from `slurm` scripts in the `scripts/` directory).
Be sure to set the appropriate `BASE_PATH` value at the top of each notebook.

## Scripts

Be sure to set appropriate `BASE_DIR` variables in `scripts/mongo/` and `scripts/processing/` directories, the latter of which also requires replacing `YOUR_ACCOUNT` with the appropriate account code.

## Webdataset Construction

The requirements and config files for taking the TreeOfLife structured dataset and putting it in webdataset format are `requirements_tol2webdataset.txt` and `tol2webdataset_full_224.yaml`, respectively. The code for this is run through `scripts/t2w_submit.sh` using the `tol2webdataset` scritps and modules.
