import collections.abc
import itertools
import logging
import os
import re
import subprocess
import time

import beartype
import psutil

logger = logging.getLogger(__name__)


@beartype.beartype
class progress:
    def __init__(self, it, *, every: int = 10, desc: str = "progress", total: int = 0):
        """
        Wraps an iterable with a logger like tqdm but doesn't use any control codes to manipulate a progress bar, which doesn't work well when your output is redirected to a file. Instead, simple logging statements are used, but it includes quality-of-life features like iteration speed and predicted time to finish.

        Args:
            it: Iterable to wrap.
            every: How many iterations between logging progress.
            desc: What to name the logger.
            total: If non-zero, how long the iterable is.
        """
        self.it = it
        self.every = every
        self.logger = logging.getLogger(desc)
        self.total = total

    def __iter__(self):
        start = time.time()

        try:
            total = len(self)
        except TypeError:
            total = None

        for i, obj in enumerate(self.it):
            yield obj

            if (i + 1) % self.every == 0:
                now = time.time()
                duration_s = now - start
                # Avoid division by zero if duration is extremely small
                per_min = (i + 1) / (duration_s / 60) if duration_s > 0 else 0

                if total is not None and total > 0 and per_min > 0:
                    pred_min = (total - (i + 1)) / per_min
                    if pred_min > 120:
                        pred_hr = pred_min / 60
                        self.logger.info(
                            f"{i + 1:,}/{total:,} (%.1f%%) | {per_min:,.1f} it/m (expected finish in %.1fh)",
                            (i + 1) / total * 100,
                            pred_hr,
                        )
                    else:
                        self.logger.info(
                            f"{i + 1:,}/{total:,} (%.1f%%) | {per_min:,.1f} it/m (expected finish in %.1fh)",
                            (i + 1) / total * 100,
                            pred_min,
                        )
                elif total is not None and total > 0:
                    # Case where per_min is 0 (e.g., first few iterations very fast)
                    self.logger.info(
                        "%d/%d (%.1f%%) | --- it/m (calculating...)",
                        i + 1,
                        total,
                        (i + 1) / total * 100,
                    )
                else:
                    # Case where total is unknown
                    self.logger.info(f"{i + 1:,}/? | {per_min:,.1f} it/m")

    def __len__(self) -> int:
        if self.total > 0:
            return self.total

        # Will throw exception if underlying iterator doesn't support len()
        return len(self.it)


@beartype.beartype
def batched(iterable, n: int, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


class MyIterable:
    def __init__(self, n):
        self.n = n

    def __iter__(self):
        for i in range(self.n):
            yield i**2  # or whatever logic

    def __len__(self):
        return self.n


@beartype.beartype
class batched_idx:
    """
    A class that iterates over (start, end) indices for a dataset of total_size, where each batch has at most batch_size elements.

    This class implements __len__ to be compatible with tqdm progress bars.
    """

    def __init__(self, total_size: int, batch_size: int):
        """
        Initialize a BatchedIdx iterator.

        Args:
            total_size: Total number of examples in the dataset
            batch_size: Maximum number of examples per batch
        """
        self.total_size = total_size
        self.batch_size = batch_size

    def __iter__(self) -> collections.abc.Iterator[tuple[int, int]]:
        """
        Iterate over (start, end) index pairs.

        Returns:
            Iterator of (start, end) tuples that can be used for slicing
        """
        for start in range(0, self.total_size, self.batch_size):
            stop = min(start + self.batch_size, self.total_size)
            yield start, stop

    def __len__(self) -> int:
        """
        Get the number of batches.

        Returns:
            Number of batches that will be generated
        """
        # Avoid division by zero if batch_size is invalid
        if self.batch_size <= 0:
            return 0
        return (self.total_size + self.batch_size - 1) // self.batch_size


def log_mem(tag=""):
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    all_mem = total_mem()
    logger.debug("[MEM %d]: %.0f MB, %.0f MB (%s)", os.getpid(), mem, all_mem, tag)


def total_mem():
    parent = psutil.Process()
    children = parent.children(recursive=True)
    rss = parent.memory_info().rss
    for child in children:
        try:
            rss += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return rss / (1024**2)  # Convert to MB


@beartype.beartype
def get_slurm_max_array_size() -> int:
    """
    Get the MaxArraySize configuration from the current Slurm cluster.

    Returns:
        int: The maximum array size allowed on the cluster. Returns 1000 as fallback if unable to determine.
    """
    try:
        # Run scontrol command to get config information
        result = subprocess.run(
            ["scontrol", "show", "config"], capture_output=True, text=True, check=True
        )

        # Search for MaxArraySize in the output
        match = re.search(r"MaxArraySize\s*=\s*(\d+)", result.stdout)
        if match:
            max_array_size = int(match.group(1))
            logger.info("Detected MaxArraySize = %d", max_array_size)
            return max_array_size
        else:
            logger.warning(
                "Could not find MaxArraySize in scontrol output, using default of 1000"
            )
            return 1000

    except subprocess.SubprocessError as e:
        logger.error("Error running scontrol: %s", e)
        return 1000  # Safe default
    except ValueError as e:
        logger.error("Error parsing MaxArraySize: %s", e)
        return 1000  # Safe default
    except FileNotFoundError:
        logger.warning(
            "scontrol command not found. Assuming not in Slurm environment. Returning default MaxArraySize=1000."
        )
        return 1000


@beartype.beartype
def get_partition_info() -> dict[str, dict]:
    """
    Get information about available Slurm partitions.

    Returns:
        dict: Dictionary mapping partition names to their properties.
    """
    partitions = {}
    try:
        # Run scontrol to get partition information
        result = subprocess.run(
            ["scontrol", "show", "partition"],
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse the output to extract partition information
        current_partition = None
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("PartitionName="):
                # Extract partition name
                name_match = re.search(r"PartitionName=(\S+)", line)
                if name_match:
                    current_partition = name_match.group(1)
                    partitions[current_partition] = {}

                    # Extract other properties
                    for prop in ["MaxNodes", "MaxTime", "MaxCPUsPerNode", "DefaultMem"]:
                        prop_match = re.search(rf"{prop}=(\S+)", line)
                        if prop_match:
                            partitions[current_partition][prop] = prop_match.group(1)

        logger.info("Found %d Slurm partitions", len(partitions))
        return partitions

    except subprocess.SubprocessError as e:
        logger.error("Error getting partition info: %s", e)
        return {}
    except FileNotFoundError:
        logger.warning("scontrol command not found. Assuming not in Slurm environment.")
        return {}


@beartype.beartype
def get_optimal_job_chunks(total_jobs: int) -> tuple[int, int]:
    """
    Calculate optimal job array size and concurrent jobs based on cluster limits.

    Args:
        total_jobs: Total number of jobs to process

    Returns:
        tuple: (array_size, concurrent_jobs) - How to structure the job array
    """
    max_array_size = get_slurm_max_array_size()

    # Determine array size (respecting MaxArraySize limit)
    array_size = min(total_jobs, max_array_size)

    # Determine concurrent jobs (rule of thumb: about 25% of array size, but no more than 256)
    concurrent_jobs = min(256, max(1, array_size // 4))

    logger.info(
        "Optimal job configuration: array_size=%d, concurrent_jobs=%d for %d total jobs",
        array_size,
        concurrent_jobs,
        total_jobs,
    )

    return array_size, concurrent_jobs
