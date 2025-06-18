# Processing Tools

As noted in the root `README`, this repository's overall structure is still in progress, as the contents of this directory are being worked into the larger tool package.

## Content-Based Filtering

The [`content_dedup/`](content_dedup) directory contains the content-based deduplication code that was used to generate perceptual hashes for test sets to compare to our training data that could overlap. It also contains early exploration of the method (in [`content_dedup/experiments/`](content_dedup/experiments)).

## Data Directory

The [`data/`](data) directory contains the support set embeddings used for museum specimen image filtering (as detailed in Appendix I.2.1 of [our paper](https://doi.org/10.48550/arXiv.2505.23883)).

## Docs Directory

This folder contains a series of files named `requirements_<processing-step>.txt`, detailing the requisite packages for said processing step (e.g., `requirements_batch_camera_trap.txt` for camera trap image processing).

## Notebooks

The `notebooks/` directory houses the notebooks used to perform their respective processing steps (most, if not all, of which were adapted to modules run from `slurm` scripts in the `scripts/` directory).
Be sure to set the appropriate `BASE_PATH` value at the top of each notebook.

## Scripts

Be sure to set appropriate `BASE_DIR` variables in `scripts/mongo/` and `scripts/processing/` directories, the latter of which also requires replacing `YOUR_ACCOUNT` with the appropriate account code.

[BioCLIP 2 text embeddings of TreeOfLife-200M](https://huggingface.co/datasets/imageomics/TreeOfLife-200M/blob/main/embeddings/txt_emb_species.npy) were generated with [`make_txt_embedding.py`](scripts/make_txt_embedding.py),  using the [`txt_emb_species.json`](https://huggingface.co/datasets/imageomics/TreeOfLife-200M/blob/main/embeddings/txt_emb_species.json) to provide the species names. More information about the `JSON` is provided in the [TreeOfLife-200M `embeddings/README`](https://huggingface.co/datasets/imageomics/TreeOfLife-200M/tree/main/embeddings/README.md).

## Webdataset Construction

The requirements and config files for taking the TreeOfLife structured dataset and putting it in webdataset format are `requirements_tol2webdataset.txt` and `tol2webdataset_full_224.yaml`, respectively. The code for this is run through `scripts/t2w_submit.sh` using the `tol2webdataset` scripts and modules.
