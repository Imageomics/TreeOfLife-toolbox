# `notebook/benchmark_validation`

This directory contains notebooks that document the process of validating benchmark images against the **TOL-200M dataset**. These validations ensure that images used for model evaluation are **not** included in the training data, preventing **data leakage**.

- `rare_species.ipynb` documents the process of mapping images from the [Rare Species](https://huggingface.co/datasets/imageomics/rare-species) benchmark dataset to the **TOL-200M training data**.
Be sure to set the appropriate `BASE_PATH` value at the top of the notebook.

Remaining benchmark checks use PDQ hashes and are in [src/content_dedup](../../src/content_dedup).
