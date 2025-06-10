import logging
import math
import os
import typing

import beartype
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import submitit
import tyro
from PIL import Image

from dedup import algorithms, datasets, filtering, hashing, helpers

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("exps")


@beartype.beartype
def hash_all(
    algorithm_key: typing.Annotated[
        algorithms.AlgorithmKey, tyro.conf.arg(name="algorithm")
    ],
    dataset_key: typing.Annotated[
        tyro.extras.literal_type_from_choices(datasets.list_datasets()),
        tyro.conf.arg(name="dataset"),
    ],
    split: str,
    data_root: str,
    job_size: int = 1024,
    batch_size: int = 128,
    parallelism: int = 4,
    out: str = os.path.join(".", "hashes"),
    log_to: str = os.path.join(".", "logs"),
    slurm_acct: str = "",
    hours: float = 1.0,
):
    """Hashes all images in a dataset using a perceptual hash algorithm.

    Args:
        algorithm: Which hash algorithm to use.
        dataset: Which dataset to hash.
        data_root: Root folder of dataset on disk.
        job_size: Number of images to process in each job.
        batch_size: Number of images to process in each batch.
        parallelism: Maximum number of jobs to run in parallel at any time.
        out: Where to save results.
        log_to: Where to save logs.
        slurm_acct: Slurm account. If blank, do not use slurm.
        hours: How many hours for your slurm job.
    """
    dataset = datasets.load(dataset_key, split, data_root)
    algorithm = algorithms.load(algorithm_key)

    if slurm_acct:
        executor = submitit.SlurmExecutor(folder=log_to)
        executor.update_parameters(
            time=int(60 * hours),
            partition="cpu",
            gpus_per_node=0,
            cpus_per_task=1,
            use_srun=False,
            mem_per_cpu="24gb",  # Consider making this configurable
            stderr_to_stdout=True,
            account=slurm_acct,
            array_parallelism=parallelism,
        )
    else:
        executor = submitit.DebugExecutor(folder=log_to)

    hash_size_bits = len(algorithm) * len(dataset)
    hash_size_mb = hash_size_bits / 8 / 1024 / 1024
    logger.info(
        "Total hash size: %.1f MB (%d bits x %d images)",
        hash_size_mb,
        len(algorithm),
        len(dataset),
    )
    hashes = np.zeros((len(dataset), len(algorithm) // 8), dtype=np.uint8)

    # Calculate job indices using BatchedIdx
    job_indices = list(helpers.batched_idx(len(dataset), job_size))
    starts, ends = zip(*job_indices) if job_indices else ([], [])
    logger.info("Submitting %d jobs.", len(starts))

    worker = hashing.Worker(algorithm_key, dataset_key, split, data_root, batch_size)

    arr_size = int(helpers.get_slurm_max_array_size() * 0.8)
    # Process jobs in batches to respect Slurm's maximum array size
    for i, (arr_start, arr_end) in enumerate(
        helpers.batched_idx(len(starts), arr_size)
    ):
        logger.info(
            "Submitting array of %d jobs (%d to %d)",
            arr_end - arr_start,
            arr_start,
            arr_end - 1,
        )

        arr_starts = starts[arr_start:arr_end]
        arr_ends = ends[arr_start:arr_end]

        jobs = executor.map_array(worker, arr_starts, arr_ends)

        failed = {}
        # Keep f-string for progress desc
        for start, end, job in zip(
            arr_starts,
            arr_ends,
            helpers.progress(jobs, desc=f"Batch {i + 1}", every=len(jobs) // 100 + 1),
        ):
            # Try/except, then resubmit with job.submission().function and job.submission.args.
            try:
                result = job.result()
                hashes[start:end] = result
            except submitit.core.utils.UncompletedJobError as err:
                logger.warning("Error for %d:%d: %s", start, end, err)
                failed[(start, end)] = job.submission()

        # Resubmit failed jobs until all succeed
        while failed:
            logger.info("Resubmitting %d failed jobs", len(failed))
            arr_starts, arr_ends = zip(*failed.keys())

            # Resubmit the jobs
            resubmit_jobs = executor.map_array(worker, arr_starts, arr_ends)

            # Clear the failed dict and repopulate with any new failures
            failed = {}
            for start, end, job in zip(
                arr_starts,
                arr_ends,
                helpers.progress(resubmit_jobs, desc="resubmitted"),
            ):
                try:
                    result = job.result()
                    hashes[start:end] = result
                except submitit.core.utils.UncompletedJobError as err:
                    logger.warning("Still failed for %d:%d: %s", start, end, err)
                    failed[(start, end)] = job.submission()

    # Save hashes.
    os.makedirs(out, exist_ok=True)
    # Keep f-string for filename generation
    np.save(
        os.path.join(out, f"{dataset}_{split}-{algorithm}.npy".replace("/", "_")),
        hashes,
    )


@beartype.beartype
def make_hist(
    algorithm: algorithms.AlgorithmKey,
    hashes_path: str,  # Renamed from hashes for clarity
    n: int = 100_000,
    seed: int = 17,
    bsz: int = 1_024,
    out: str = os.path.join(".", "plots"),
):
    """Randomly samples pairs of hashes and calculates a distribution (histogram) of distances."""
    plt_fname, _ = os.path.splitext(os.path.basename(hashes_path))
    # Keep f-string for filename generation
    plt_fpath = os.path.join(out, f"{plt_fname}.png")
    os.makedirs(out, exist_ok=True)

    algorithm_obj = algorithms.load(algorithm)  # Renamed from algorithm

    hashes_data = np.load(hashes_path)  # Renamed from hashes
    n_hashed, n_bytes = hashes_data.shape

    assert len(algorithm_obj) == n_bytes * 8

    if n_hashed < n:
        logger.warning(
            "Asked for %d samples but only have %d hashes. Using all hashes.",
            n,
            n_hashed,
        )
        n = n_hashed
        sampled_hashes = hashes_data  # Use all hashes
    else:
        rng = np.random.default_rng(seed=seed)
        # Sample indices first, then select hashes
        sampled_indices = rng.choice(n_hashed, size=n, replace=False)
        sampled_hashes = hashes_data[sampled_indices]

    # Use the N x N distance calculation function from hashing module
    dists = hashing.calc_dists(sampled_hashes, bsz)  # Returns uint16

    # Extract upper triangle (excluding diagonal) for histogram
    rows, cols = np.triu_indices(n, k=1)  # k=1 excludes the diagonal
    if rows.size > 0:  # Check if there are any pairs
        pairwise_dists = dists[rows, cols]
    else:
        pairwise_dists = np.array([], dtype=dists.dtype)  # Empty array if n < 2

    fig, ax = plt.subplots()
    if pairwise_dists.size > 0:
        # Ensure bins cover the full range potentially up to 256
        max_possible_dist = n_bytes * 8
        ax.hist(pairwise_dists, bins=np.arange(0, max_possible_dist + 2, 2))
    else:
        logger.warning("No pairs to plot histogram for.")
    ax.set_xlabel("Hamming Distance")
    ax.set_ylabel("Frequency")
    ax.set_yscale("log")
    ax.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(plt_fpath)
    logger.info("Saved histogram to %s", plt_fpath)


@beartype.beartype
def make_clusters(
    algorithm: algorithms.AlgorithmKey,
    dataset: tyro.extras.literal_type_from_choices(datasets.list_datasets()),
    split: str,
    data_root: str,
    hashes_path: str,  # Renamed from hashes
    n: int = 30_000,
    seed: int = 17,
    bsz: int = 1_024,
    # n_dists: int = 10, # This seemed unused, using fixed thresholds instead
    n_clusters_per_dist: int = 10,  # Renamed from n_clusters
    max_imgs_per_cluster: int = 20,  # Renamed from max_imgs
    out: str = os.path.join(".", "clusters"),
):
    """
    Creates clusters of images based on Hamming distance between their hashes.

    Steps:
    1. Load hashes and sample a subset (`n`).
    2. Calculate pairwise distances for the sampled subset.
    3. Define distance thresholds (e.g., [0, 2, 5, ...]).
    4. For each distance range [lower, upper]:
       a. Find query images that have *at least one* neighbor in that distance range.
       b. Randomly select `n_clusters_per_dist` query images from those found.
       c. For each selected query image:
          i. Find all its neighbors within the distance range [lower, upper).
          ii. Limit neighbors to `max_imgs_per_cluster`.
          iii. Save the query image and its neighbor images to a structured directory.

    Args:
        algorithm: Which hash algorithm was used.
        dataset: The dataset key.
        split: The dataset split.
        data_root: Root directory of the dataset.
        hashes_path: Path to the stored image hashes (.npy file).
        n: Number of hashes to sample for distance calculation and cluster seeding.
        seed: Random seed.
        bsz: Batch size for calculating distance matrix.
        n_clusters_per_dist: Number of example clusters to generate for each distance range.
        max_imgs_per_cluster: Maximum number of similar images to save per cluster (excluding the query).
        out: Root directory to save the output image clusters.
    """
    algorithm_obj = algorithms.load(algorithm)  # Renamed from algorithm

    hashes_data = np.load(hashes_path)  # Renamed from hashes
    n_hashed, n_bytes = hashes_data.shape

    assert len(algorithm_obj) == n_bytes * 8

    rng = np.random.default_rng(seed=seed)

    if n_hashed < n:
        logger.warning(
            "Asked for %d samples but only have %d hashes. Using all %d hashes.",
            n,
            n_hashed,
            n_hashed,
        )
        n = n_hashed
        sampled_indices = np.arange(n)  # Indices are 0 to n-1
        sampled_hashes = hashes_data  # Hashes are the full dataset
    else:
        # Sample indices first
        sampled_indices = rng.choice(n_hashed, size=n, replace=False)
        # Then select the corresponding hashes
        sampled_hashes = hashes_data[sampled_indices]

    logger.info("Calculating pairwise distances for %d sampled hashes...", n)
    # Use the N x N distance calculation function
    paired_dists = hashing.calc_dists(sampled_hashes, bsz)  # Returns uint16
    logger.info("Distance calculation complete.")

    # Define distance thresholds for clustering
    # Using np.max().item() ensures we get a Python scalar
    max_dist = np.max(paired_dists).item() if n > 0 else 0
    # Ensure thresholds are within the valid range of the distance dtype (uint16)
    cluster_thresholds = sorted(
        list(
            set([0, 2, 5, 10, 20, 50, 100, 200, int(max_dist)])
        )  # Cast max_dist to int
    )

    dataset_obj = datasets.load(dataset, split, data_root)  # Renamed from dataset

    # Iterate through distance ranges [lower, upper)
    for lower, upper in zip(
        helpers.progress(
            cluster_thresholds[:-1], desc="Generating clusters", every=1
        ),  # Exclude last element for lower bound
        cluster_thresholds[1:],  # Exclude first element for upper bound
    ):
        if (
            lower >= upper
        ):  # Skip if thresholds are identical (e.g., max_dist was already in the list)
            continue

        # Find indices (within the sampled set) of images that have *at least one* neighbor
        # with distance D such that lower <= D < upper.
        # np.any checks if *any* distance in a row falls within the range.
        # np.flatnonzero gives the indices where the condition is true.
        eligible_query_indices = np.flatnonzero(
            np.any((lower <= paired_dists) & (paired_dists < upper), axis=1)
        )

        if not eligible_query_indices.size:
            logger.warning(
                "No image pairs found with distance in range [%d, %d). Skipping.",
                lower,
                upper,
            )
            continue

        # Randomly select 'n_clusters_per_dist' query images from the eligible ones
        num_to_select = min(eligible_query_indices.size, n_clusters_per_dist)
        selected_query_indices = rng.choice(
            eligible_query_indices, size=num_to_select, replace=False
        )

        logger.info(
            "Generating %d clusters for distance range [%d, %d).",
            num_to_select,
            lower,
            upper,
        )

        # Process each selected query image
        for query_idx_in_sample in selected_query_indices:
            # Get the original index in the full dataset
            original_query_idx = sampled_indices[query_idx_in_sample]

            # Define output directory for this cluster
            # Keep f-strings for path generation as it's clearer
            out_cluster_dir = os.path.join(
                out,
                f"{str(dataset_obj).replace('/', '_')}",
                f"dist_{lower}-{upper}",
                f"query_{original_query_idx}",
            )
            os.makedirs(out_cluster_dir, exist_ok=True)

            # --- Save the query image ---
            try:
                query_img_array, _ = dataset_obj[original_query_idx.item()]
                query_img = Image.fromarray(query_img_array)
                # Keep f-string for filename
                query_img_filename = f"query_{original_query_idx}.jpg"
                query_img.save(os.path.join(out_cluster_dir, query_img_filename))
            except (
                Exception
            ) as e:  # Catch potential errors during dataset access or saving
                logger.warning(
                    "Could not load/save query image (original index %d): %s. Skipping query.",
                    original_query_idx,
                    e,
                )
                continue  # Skip to the next query image

            # --- Find and save neighbor images ---
            # Find indices (within the sampled set) of neighbors whose distance to the query image
            # is within the range [lower, upper).
            neighbor_indices_in_sample = np.flatnonzero(
                (lower <= paired_dists[query_idx_in_sample])
                & (paired_dists[query_idx_in_sample] < upper)
            )

            # Exclude the query image itself if it appears in the neighbors
            neighbor_indices_in_sample = neighbor_indices_in_sample[
                neighbor_indices_in_sample != query_idx_in_sample
            ]

            # Limit the number of neighbors to save
            if len(neighbor_indices_in_sample) > max_imgs_per_cluster:
                neighbor_indices_in_sample = rng.choice(
                    neighbor_indices_in_sample, size=max_imgs_per_cluster, replace=False
                )

            # Save the neighbor images
            saved_neighbors = 0
            for neighbor_idx_in_sample in neighbor_indices_in_sample:
                original_neighbor_idx = sampled_indices[neighbor_idx_in_sample]
                # Get the distance (uint16)
                distance = paired_dists[query_idx_in_sample, neighbor_idx_in_sample]

                try:
                    neighbor_img_array, _ = dataset_obj[original_neighbor_idx.item()]
                    neighbor_img = Image.fromarray(neighbor_img_array)
                    # Keep f-string for filename generation
                    neighbor_filename = (
                        f"match_{original_neighbor_idx}_dist-{distance}.jpg"
                    )
                    neighbor_img.save(os.path.join(out_cluster_dir, neighbor_filename))
                    saved_neighbors += 1
                except Exception as e:
                    logger.warning(
                        "Could not load/save neighbor image (original index %d) for query %d: %s",
                        original_neighbor_idx,
                        original_query_idx,
                        e,
                    )
                    continue  # Skip to the next neighbor

            logger.debug(
                "Saved %d neighbors for query %d in range [%d, %d).",
                saved_neighbors,
                original_query_idx,
                lower,
                upper,
            )


@beartype.beartype
def filter_tol(
    tol_hash_root: str,
    tol_splits: list[str],
    test_hashes: list[str],
    output_pf: str,
    threshold: int = 10,
    batch_size: int = 1024,
    job_size: int = 1024 * 256,
    debug: bool = False,
    # Slurm arguments
    slurm_acct: str = "",
    hours: float = 0.5,
    mem_per_cpu: str = "10gb",
    parallelism: int = 256,
    log_to: str = os.path.join(".", "logs"),
):
    """
    Filters Tree of Life (ToL) hashes based on proximity to test set hashes using Slurm.

    Identifies ToL UUIDs whose PDQ hash has a Hamming distance <= threshold to *any* hash in the provided test sets. Each unique combination of (ToL row group batch chunk, test hash file)is processed by a separate Slurm job. Optionally filters an input CSV based on these UUIDs.

    Args:
        tol_hash_root: Root directory containing ToL PDQ hashes in parquet format.
        tol_splits: List of specific source splits to process within tol_hash_root.
        test_hashes: List of paths to test set PDQ hashes (.npy format).
        output_pf: Where to save the excluded UUIDs.
        threshold: Maximum Hamming distance to consider a match.
        batch_size: Number of ToL hashes to load and compare in each iteration within a job.
        job_size: Number of ToL hashes to process per job.
        debug: If True, only process the first chunk of batches for each row group.
        slurm_acct: Slurm account. If blank, run locally (for debugging).
        hours: Time limit for each Slurm job.
        mem_per_cpu: Memory per CPU for each Slurm job (e.g., "8gb").
        parallelism: Maximum number of Slurm jobs to run concurrently.
        log_to: Directory to save Slurm logs.
    """

    # --- 1. Find ToL Parquet Files and create job inputs ---
    # Find ToL parquet files.
    tol_pq_files = filtering.get_tol_pq_files(tol_hash_root, tol_splits)
    if not tol_pq_files:
        logger.error(
            "No Parquet files found for any specified splits in %s", tol_hash_root
        )
        return
    logger.info("Found %d ToL parquet files.", len(tol_pq_files))

    missing = False
    for test_hash in test_hashes:
        try:
            np.load(test_hash)
        except Exception:
            logger.error("Missing hash '%s'", test_hash)
            missing = True

    if missing:
        logger.error("Not submitting jobs due to missing hashes.")
        return

    # Create job inputs using the refactored function
    job_inputs = filtering.make_job_args(
        tol_pq_files, test_hashes, job_size, debug=debug
    )

    if not job_inputs:
        logger.error("No jobs created. Check input files and parameters.")
        return

    # --- 2. Setup Slurm Executor ---
    os.makedirs(log_to, exist_ok=True)
    if slurm_acct:
        executor = submitit.SlurmExecutor(folder=log_to)
        executor.update_parameters(
            time=int(60 * hours),
            partition="cpu",
            gpus_per_node=0,
            cpus_per_task=1,
            mem_per_cpu=mem_per_cpu,
            stderr_to_stdout=True,
            account=slurm_acct,
            array_parallelism=parallelism,
        )
        logger.info("Configured Slurm executor with parallelism %d", parallelism)
    else:
        logger.info("No Slurm account provided, running locally for debugging.")
        executor = submitit.DebugExecutor(folder=log_to)

    # --- 3. Prepare and Submit Jobs ---
    worker = filtering.FilterTolWorker(threshold=threshold, batch_size=batch_size)

    # --- 4. Collect Results ---
    all_uuids_to_remove: set[str] = set()
    failed_jobs = 0
    arr_size = int(helpers.get_slurm_max_array_size() * 0.8)
    # Process jobs in batches to respect Slurm's maximum array size
    n_arrs = math.ceil(len(job_inputs) / arr_size)
    for i, (arr_start, arr_end) in enumerate(
        helpers.batched_idx(len(job_inputs), arr_size)
    ):
        logger.info(
            "Submitting array of %d jobs (%d to %d)",
            arr_end - arr_start,
            arr_start,
            arr_end - 1,
        )

        arr_inputs = job_inputs[arr_start:arr_end]

        jobs = executor.map_array(worker, arr_inputs)

        for job in helpers.progress(
            jobs, desc=f"array {i + 1}/{n_arrs}", every=len(jobs) // 20 + 1
        ):
            all_uuids_to_remove.update(job.result())

    # --- 5. Log Summary ---
    logger.info("Finished collecting results.")
    if failed_jobs > 0:
        logger.warning("%d jobs failed.", failed_jobs)
    else:
        logger.info("All jobs completed successfully.")

    logger.info(
        "Found %d unique ToL UUIDs with Hamming distance <= %d to any test hash.",
        len(all_uuids_to_remove),
        threshold,
    )

    # --- 6. Create 'exclude' Parquet File ---
    output_dir = os.path.dirname(output_pf)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df = pl.DataFrame({"uuid": list(sorted(all_uuids_to_remove))})
    df.write_parquet(output_pf)
    # Also write a CSV version
    csv_path = output_pf.replace(".parquet", ".csv")
    df.write_csv(csv_path)
    logger.info(
        "Wrote %d UUIDs to %s and %s", len(all_uuids_to_remove), output_pf, csv_path
    )


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({
        "hash-all": hash_all,
        "make-hist": make_hist,
        "make-clusters": make_clusters,
        "filter-tol": filter_tol,
    })
