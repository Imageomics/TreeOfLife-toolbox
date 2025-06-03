import glob
import logging
import math
import os

import beartype
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from jaxtyping import UInt8, jaxtyped

from . import hashing, helpers

logger = logging.getLogger(__name__)


@jaxtyped(typechecker=beartype.beartype)
def convert_arrow_hashes_to_numpy(
    hashes_col: pa.Array, batch_len: int, file_basename: str, batch_num: int
) -> UInt8[np.ndarray, "batch_len 32"] | None:
    """Converts an Arrow array of hashes to a NumPy uint8 array."""
    try:
        if hashes_col.null_count > 0:
            logger.warning(
                "Batch %d in %s contains null hashes. Falling back to slower conversion.",
                batch_num + 1,
                file_basename,
            )
            tol_batch_hashes_list = [b.as_py() if b else None for b in hashes_col]
            valid_hashes_list = [
                h for h in tol_batch_hashes_list if h is not None and len(h) == 32
            ]
            if len(valid_hashes_list) != batch_len:
                logger.error(
                    "  Skipping batch %d due to %d null/invalid length hashes (complex index alignment).",
                    batch_num + 1,
                    batch_len - len(valid_hashes_list),
                )
                return None
            tol_batch_hashes = np.array(
                [np.frombuffer(b, dtype=np.uint8) for b in valid_hashes_list],
                dtype=np.uint8,
            )

        elif isinstance(hashes_col, (pa.BinaryArray, pa.LargeBinaryArray)):
            # Check if offsets buffer exists and is valid before accessing data buffer
            offsets = hashes_col.buffers()[1]
            if offsets is None:
                raise ValueError(
                    "Offsets buffer is missing for Binary/LargeBinaryArray."
                )

            data_buffer = hashes_col.buffers()[2]
            if data_buffer is None:
                raise ValueError("Data buffer is missing for Binary/LargeBinaryArray.")

            expected_buffer_size = batch_len * 32
            # Determine offset dtype based on array type
            offset_dtype = (
                np.int64 if isinstance(hashes_col, pa.LargeBinaryArray) else np.int32
            )
            offset_values = np.frombuffer(offsets, dtype=offset_dtype)
            # Check if offsets indicate fixed size (more reliable check)
            actual_lengths = np.diff(offset_values)
            if not np.all(actual_lengths == 32):
                logger.warning(
                    "  Variable length hashes detected in Binary/LargeBinaryArray based on offsets. Falling back."
                )
                # Fallback logic
                tol_batch_hashes_list = [b.as_py() for b in hashes_col]
                valid_hashes_list = [h for h in tol_batch_hashes_list if len(h) == 32]
                if len(valid_hashes_list) != batch_len:
                    logger.error(
                        "  Skipping batch %d due to hash length mismatch during fallback.",
                        batch_num + 1,
                    )
                    return None
                tol_batch_hashes = np.array(
                    [np.frombuffer(b, dtype=np.uint8) for b in valid_hashes_list],
                    dtype=np.uint8,
                )
            elif len(data_buffer) == expected_buffer_size:
                # Only use direct buffer access if sizes match exactly and offsets confirm fixed size
                tol_batch_hashes = np.frombuffer(data_buffer, dtype=np.uint8).reshape(
                    batch_len, 32
                )
            else:
                # Fallback if buffer size doesn't match expectation
                logger.warning(
                    "  Data buffer size %d != expected %d. Falling back.",
                    len(data_buffer),
                    expected_buffer_size,
                )
                # Fallback logic
                tol_batch_hashes_list = [b.as_py() for b in hashes_col]
                valid_hashes_list = [h for h in tol_batch_hashes_list if len(h) == 32]
                if len(valid_hashes_list) != batch_len:
                    logger.error(
                        "  Skipping batch %d due to hash length mismatch during fallback.",
                        batch_num + 1,
                    )
                    return None
                tol_batch_hashes = np.array(
                    [np.frombuffer(b, dtype=np.uint8) for b in valid_hashes_list],
                    dtype=np.uint8,
                )

        elif isinstance(hashes_col, pa.FixedSizeBinaryArray):
            if hashes_col.type.byte_width != 32:
                raise TypeError(
                    "Expected FixedSizeBinaryArray byte_width 32, got %d"
                    % hashes_col.type.byte_width
                )
            data_buffer = hashes_col.buffers()[1]
            if data_buffer is None:
                raise ValueError("Data buffer is missing for FixedSizeBinaryArray.")
            tol_batch_hashes = np.frombuffer(data_buffer, dtype=np.uint8).reshape(
                batch_len, 32
            )

        else:
            logger.warning(
                "  Unexpected Arrow array type for hashes: %s. Falling back to slower conversion.",
                type(hashes_col),
            )
            tol_batch_hashes_list = [b.as_py() for b in hashes_col]
            tol_batch_hashes = np.array(
                [
                    np.frombuffer(b, dtype=np.uint8)
                    for b in tol_batch_hashes_list
                    if len(b) == 32
                ],
                dtype=np.uint8,
            )
            if tol_batch_hashes.shape[0] != batch_len:
                logger.error(
                    "  Skipping batch %d due to hash length mismatch during fallback conversion.",
                    batch_num + 1,
                )
                return None

        # Final shape validation
        if tol_batch_hashes.ndim != 2 or tol_batch_hashes.shape != (batch_len, 32):
            logger.error(
                "  Skipping batch %d: Incorrect final hash shape (%s). Expected (%d, 32).",
                batch_num + 1,
                tol_batch_hashes.shape,
                batch_len,
            )
            return None

        return tol_batch_hashes

    except Exception as e:
        logger.error(
            "  Skipping batch %d in %s due to hash conversion error: %s",
            batch_num + 1,
            file_basename,
            e,
            exc_info=True,
        )
        return None


@jaxtyped(typechecker=beartype.beartype)
def process_tol_row_group(
    pq_path: str,
    row_group: int,
    test_hashes: UInt8[np.ndarray, "M 32"],
    threshold: int,
    batch_size: int,
    start: int,
    end: int,
) -> set[str]:
    """Processes a chunk of batches within a ToL Parquet row group against multiple test hashes."""
    chunk_uuids_to_remove = set()

    filename = os.path.basename(pq_path)
    chunk_desc = f"{filename} RG {row_group} Rows {start}-{end} "

    pf = pq.ParquetFile(pq_path)

    # Basic validation
    if row_group >= pf.num_row_groups:
        logger.error(
            "Invalid row_group_index %d for file %s with %d row groups.",
            row_group,
            filename,
            pf.num_row_groups,
        )
        return set()

    # Create batch iterator for the specified row group
    it = pf.iter_batches(
        batch_size=batch_size,
        columns=["uuid", "hash_pdq"],
        row_groups=[row_group],
    )

    # Skip first set of rows.
    n_seen = 0
    while n_seen + batch_size < start:
        n_seen += len(next(it))
    logger.info("Skipped %d examples in order to start at %d.", n_seen, start)

    # Calculate number of batches this job is responsible for
    n_batches = math.ceil((end - start) / batch_size)
    if n_batches <= 0:
        logger.warning("%s has zero batches to process. Skipping.", chunk_desc)
        return set()
    it = helpers.progress(it, desc=chunk_desc, total=n_batches)

    for b, batch in enumerate(it):
        uuids_col = batch["uuid"].to_pylist()
        hashes_col = batch["hash_pdq"]

        # Convert hashes
        tol_hashes = convert_arrow_hashes_to_numpy(hashes_col, len(batch), filename, b)
        if tol_hashes is None:
            continue

        # Calculate distances against all test hashes in the provided array
        # Result shape: (batch_len, M)
        distances = hashing.calculate_hamming_distances(tol_hashes, test_hashes)

        # Find the minimum distance for each ToL hash across all test hashes
        # Result shape: (batch_len,)
        min_distances = np.min(distances, axis=1)

        # Filter based on the minimum distance
        match_indices_in_batch = np.where(min_distances <= threshold)[0]

        # Collect matching UUIDs
        if len(match_indices_in_batch) > 0:
            matched_uuids = {uuids_col[idx] for idx in match_indices_in_batch}
            chunk_uuids_to_remove.update(matched_uuids)

        n_seen += len(batch)
        if n_seen >= end:
            logger.info("Breaking after %d batches, %d seen.", b, n_seen)
            break

    return chunk_uuids_to_remove


@beartype.beartype
class FilterTolWorker:
    def __init__(self, threshold: int, batch_size: int):
        self.threshold = threshold
        self.batch_size = batch_size

    def __call__(self, job_input: tuple[str, int, str, int, int]) -> set[str]:
        """Processes a chunk of batches within a row group against test hashes from a given file."""
        pq_path, row_group, test_hash_path, start, end = job_input

        # Reset logging for the Slurm job
        log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
        logging.basicConfig(level=logging.INFO, format=log_format)

        try:
            # Use np.load directly as load_test_hash was modified to load only one
            test_hashes = np.load(test_hash_path)
            # Validate the loaded hashes
            if not (
                test_hashes.ndim == 2
                and test_hashes.shape[1] == 32
                and test_hashes.dtype == np.uint8
            ):
                logger.error(
                    "Invalid test hashes loaded from %s: Shape %s, Dtype %s. Expected (M, 32) uint8. Skipping job.",
                    test_hash_path,
                    test_hashes.shape,
                    test_hashes.dtype,
                )
                return set()
            if test_hashes.shape[0] == 0:
                logger.warning(
                    "Test hash file %s is empty. Skipping job.", test_hash_path
                )
                return set()

        except FileNotFoundError:
            logger.error("Test hash file not found: %s. Skipping job.", test_hash_path)
            return set()
        except Exception as e:
            logger.error(
                "Failed to load test hashes from %s: %s. Skipping job.",
                test_hash_path,
                e,
                exc_info=True,
            )
            return set()

        # Call the function that processes the specific chunk of batches in the row group
        return process_tol_row_group(
            pq_path, row_group, test_hashes, self.threshold, self.batch_size, start, end
        )


@beartype.beartype
def get_tol_pq_files(root: str, splits: list[str]) -> list[str]:
    """Finds all Parquet files for the given splits in the root directory."""
    all_files = []
    for split in splits:
        glob_path = os.path.join(root, f"source={split}", "*.parquet")
        files = sorted(glob.glob(glob_path))
        if not files:
            logger.warning(
                "No Parquet files for split '%s' matching '%s'", split, glob_path
            )
            continue
        logger.info("Found %d Parquet files for split '%s'", len(files), split)
        all_files.extend(files)

    return all_files


@beartype.beartype
def make_job_args(
    pq_files: list[str], test_hashes: list[str], job_size: int, *, debug: bool
) -> list[tuple[str, int, str, int, int]]:
    """
    Creates the list of job arguments for filter_tol.

    Args:
        pq_files: List of paths to ToL Parquet files.
        test_hashes: List of paths to test set hash files (.npy).
        job_size: The number of hashes to assign to each job.
        debug: If True, only create jobs for the first batch chunk of each row group.

    Returns:
        The list of job arguments, where each element is
            (path, rg_idx, test_hash_path, batch_start_in_rg, batch_end_in_rg).
    """
    job_inputs = []
    logger.info("Scanning files for row groups...")
    for path in pq_files:
        pf = pq.ParquetFile(path)
        for rg_i in range(pf.num_row_groups):
            rg_meta = pf.metadata.row_group(rg_i)
            if rg_meta.num_rows == 0:
                continue  # Skip empty row groups

            for test_hash in test_hashes:
                for start, end in helpers.batched_idx(rg_meta.num_rows, job_size):
                    job_inputs.append((path, rg_i, test_hash, start, end))
                    if debug:
                        break

    return job_inputs
