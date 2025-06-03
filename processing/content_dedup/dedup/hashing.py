import concurrent.futures
import logging
import multiprocessing
import multiprocessing.shared_memory
import os

import beartype
import numpy as np
from jaxtyping import UInt8, UInt16, jaxtyped

from . import algorithms, datasets, helpers

logger = logging.getLogger(__name__)

# Precompute a lookup table for population count (Hamming weight) of a byte.
# This is much faster than calculating it repeatedly.
POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


@beartype.beartype
class Worker:
    def __init__(
        self,
        algorithm_key: algorithms.AlgorithmKey,
        dataset_key: datasets.DatasetKey,
        split: str,
        root: str,
        batch_size: int,
    ):
        self.algorithm_key = algorithm_key
        self.dataset_key = dataset_key
        self.split = split
        self.root = root
        self.bsz = batch_size

    def __call__(
        self,
        start: int,
        end: int,
    ) -> UInt8[np.ndarray, "n_imgs n_bytes"]:
        # We have to reset the logging format/level because submitit doesn't run the global statements.
        log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
        logging.basicConfig(level=logging.INFO, format=log_format)
        helpers.log_mem("called worker")

        algorithm = algorithms.load(self.algorithm_key)
        helpers.log_mem("loaded algorithm")
        dataset = datasets.load(
            self.dataset_key, self.split, self.root, batch_size=self.bsz
        )
        helpers.log_mem("loaded dataset")

        # Create array to store hashes
        n_imgs = end - start
        hash_bytes = len(algorithm) // 8
        hashes = np.zeros((n_imgs, hash_bytes), dtype=np.uint8)
        helpers.log_mem("empty hashes.")

        # Process images in batches
        for b, (b_start, b_end) in enumerate(
            helpers.progress(helpers.batched_idx(n_imgs, self.bsz))
        ):
            # Map batch indices to dataset indices
            ds_start = start + b_start
            ds_end = start + b_end

            # Load batch of imgs
            batch_imgs = []
            for i in range(ds_start, ds_end):
                img, _ = dataset[i]
                batch_imgs.append(img)
                # Removed excessive logging: helpers.log_mem(f"{b} {i}")

            logger.debug("Got batch %d.", b)
            helpers.log_mem("after imgs")

            # Hash the batch and store results
            hashes[b_start:b_end] = algorithm.batch(batch_imgs)
            helpers.log_mem("after alg.batch")

        return hashes


@jaxtyped(typechecker=beartype.beartype)
def hash_all_slurm_fn(
    algorithm_key: algorithms.AlgorithmKey,
    dataset_key: datasets.DatasetKey,
    split: str,
    root: str,
    n_workers: int,
    batch_size: int,
    out: str,
) -> UInt8[np.ndarray, "n_imgs 32"]:
    # We have to reset the logging format/level because submitit doesn't run the global statements.
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger("hashing")

    multiprocessing.set_start_method("spawn")

    # 1. Load dataset
    dataset = datasets.load(dataset_key, split, root)
    algorithm = algorithms.load(algorithm_key)

    # 2. Calculate hash size.
    hash_size_bits = len(algorithm) * len(dataset)
    hash_size_mb = hash_size_bits / 8 / 1024 / 1024
    logger.info(
        "Total hash size: %.1f MB (%d bits x %d images)", # Use %d for len(dataset)
        hash_size_mb,
        len(algorithm),
        len(dataset)
    )

    hashes = np.zeros((len(dataset), len(algorithm) // 8), dtype=np.uint8)

    try:
        shm = multiprocessing.shared_memory.SharedMemory(
            create=True, size=hashes.nbytes
        )
        shared_hashes = np.ndarray(hashes.shape, dtype=hashes.dtype, buffer=shm.buf)
        shared_hashes[:] = hashes[:]  # Copy initial data

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            # initializer=init_worker, # Removed initializer as init_worker is not defined here
            # initargs=(dataset_key, split, root),
        ) as pool:
            jobs = []
            for start, end in helpers.batched_idx(len(dataset), batch_size):
                jobs.append(
                    pool.submit(
                        # hash_batch_worker_fn, # Removed as hash_batch_worker_fn is not defined here
                        # Placeholder: Need the actual worker function logic here
                        # For now, just submitting None to avoid error, replace with actual task
                        lambda *args: None,
                        algorithm,
                        shm.name,
                        hashes.shape,
                        hashes.dtype,
                        start,
                        end,
                    )
                )

            for job in helpers.progress(
                concurrent.futures.as_completed(jobs), total=len(jobs)
            ):
                job.result()

        hashes[:] = shared_hashes[:]

        # Free and remove the shared memory
        shm.close()
        shm.unlink()
    except Exception as err:
        logger.error("Error in hash_all_slurm_fn(): %s", err)
        # Ensure shared memory is cleaned up
        if "shm" in locals():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
        raise
    finally:
        # Ensure worker processes are terminated properly
        if "pool" in locals():
            pool.shutdown(wait=True)

    # Save hashes.
    os.makedirs(out, exist_ok=True)
    # Keep f-string for filename generation
    np.save(
        os.path.join(out, f"{dataset}_{split}-{algorithm}.npy".replace("/", "_")),
        hashes,
    )


@jaxtyped(typechecker=beartype.beartype)
def calculate_hamming_distances(
    hashes1: UInt8[np.ndarray, "N n_bytes"],
    hashes2: UInt8[np.ndarray, "M n_bytes"],
) -> UInt16[np.ndarray, "N M"]:
    """
    Calculates the Hamming distance between two arrays of hashes (uint8).

    Uses a precomputed popcount table and broadcasting for efficiency.

    Args:
        hashes1: NumPy array of shape (N, n_bytes), dtype=uint8.
        hashes2: NumPy array of shape (M, n_bytes), dtype=uint8.

    Returns:
        NumPy array of shape (N, M) containing Hamming distances, dtype=uint16.
    """
    assert hashes1.ndim == 2 and hashes1.dtype == np.uint8
    assert hashes2.ndim == 2 and hashes2.dtype == np.uint8
    assert hashes1.shape[1] == hashes2.shape[1], "Number of bytes must match"

    # Use broadcasting for XOR: (N, 1, n_bytes) ^ (1, M, n_bytes) -> (N, M, n_bytes)
    xor_result = np.bitwise_xor(hashes1[:, np.newaxis, :], hashes2[np.newaxis, :, :])

    # Apply popcount lookup table: (N, M, n_bytes) -> (N, M, n_bytes)
    popcounts = POPCOUNT_TABLE[xor_result]

    # Sum across the byte dimension: (N, M, n_bytes) -> (N, M)
    # Use uint16 for the sum as max distance (e.g., 256) can exceed uint8 max (255)
    distances = np.sum(popcounts, axis=2, dtype=np.uint16)

    return distances


@jaxtyped(typechecker=beartype.beartype)
def calc_dists(
    hashes: UInt8[np.ndarray, "n n_bytes"], bsz: int
) -> UInt16[np.ndarray, "n n"]:
    """
    Calculates the pairwise Hamming distances within a single set of hashes.

    This version computes the full N x N distance matrix, potentially using
    batching (`bsz`) to manage memory for the intermediate XOR results if N is large.
    It leverages the precomputed POPCOUNT_TABLE.

    Args:
        hashes: A NumPy array of shape (n, n_bytes) containing the hashes.
        bsz: The batch size to use for calculating distances block by block.

    Returns:
        A NumPy array of shape (n, n) containing the pairwise Hamming distances, dtype=uint16.
    """
    n, n_bytes = hashes.shape

    # Output distance matrix. Use uint16 as max distance can exceed 255.
    dists = np.zeros((n, n), dtype=np.uint16)
    logger.info("Initialized distance matrix with %d elements.", dists.size)

    for i in helpers.progress(range(0, n, bsz), desc="Calculating distances"):
        i_end = min(i + bsz, n)
        # Calculate block for hashes[i:i_end] vs hashes[i:n]
        # This avoids redundant calculations by only computing the upper triangle + diagonal
        for j in range(i, n, bsz):
            j_end = min(j + bsz, n)

            # Extract blocks
            block_i = hashes[i:i_end]  # Shape: (current_bsz_i, n_bytes)
            block_j = hashes[j:j_end]  # Shape: (current_bsz_j, n_bytes)

            # Calculate distances between block_i and block_j using the general function
            # Result shape: (current_bsz_i, current_bsz_j), dtype=uint16
            block_dists = calculate_hamming_distances(block_i, block_j)

            # Assign to the correct block in the output matrix
            dists[i:i_end, j:j_end] = block_dists

            # If it's not a diagonal block, assign the transpose to the lower triangle
            if i != j:
                dists[j:j_end, i:i_end] = block_dists.T

    return dists
