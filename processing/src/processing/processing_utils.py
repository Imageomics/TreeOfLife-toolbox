import os
from collections import defaultdict
import logging
import torch
import torch.distributed as dist

import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms

import pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def format_time(seconds):
    """Format seconds into a human-readable string with appropriate units."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds %= 60
        return f"{int(minutes)}m {seconds:.2f}s"
    else:
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"

def setup_ddp(global_rank, world_size, local_rank, inference=True):

    logging.info(f"Available CPUs:{os.cpu_count()}")
    logging.info(f"World size: {world_size}")
    logging.info(f"[Rank {global_rank}] Local rank: {local_rank}")
    logging.info(f"[Rank {global_rank}] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    logging.info(f"[Rank {global_rank}] torch.cuda.device_count(): {torch.cuda.device_count()}")
    logging.info(f"[Rank {global_rank}] torch.cuda.current_device(): {torch.cuda.current_device()}")

    if not inference:
        dist.init_process_group("nccl", rank=global_rank, world_size=world_size)
    torch.cuda.set_device(local_rank) # Local
    

def cleanup_ddp(inference=True):
    if not inference:
        dist.destroy_process_group()
    logging.info("Distributed process group destroyed.")
    

class ParquetIterableDataset(IterableDataset):
    """
    An IterableDataset that reads images from Parquet files in a distributed manner.
    Each worker reads a subset of the files based on its rank and world size.
    The dataset yields tuples of (uuid, image) where uuid is the unique identifier
    for the image and image is a transformed tensor.
    The dataset can be used with PyTorch's DataLoader for distributed inference & training.
    
    Args:
        parquet_files (list): List of paths to Parquet files.
        rank (int): Rank of the current process.
        world_size (int): Total number of processes.
        decode_fn (callable): Function to decode images from the Parquet file.
        preprocess (callable, optional): Preprocessing function to apply to the images.
            Defaults to a series of transformations including resizing, cropping, and normalization.
        read_batch_size (int, optional): Number of rows to read from each Parquet file at once. Defaults to 100.
        processed_files_log (str, optional): Path to the log file for tracking processed files. Defaults to None.
        evenly_distribute (bool, optional): Whether to distribute files evenly based on size. Defaults to True.
        If False, files are distributed in a round-robin manner.
    """
    def __init__(self, parquet_files, rank, world_size, decode_fn, preprocess=None, read_batch_size=100, processed_files_log=None, evenly_distribute=True):
        
        self.decode_fn = decode_fn  # decode_image_to_pil function
        self.preprocess = preprocess or transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        self.read_batch_size = read_batch_size
        self.rank = rank # Store rank for logging
        self.world_size = world_size

        self.files = self.assign_files_to_rank(parquet_files, evenly_distribute=evenly_distribute)

        self.processed_files_log = processed_files_log or f"processed_files_rank{rank}.log"
        self.processed_files = self.load_processed_files()

        logging.info(f"[Rank {self.rank}] Assigned {len(self.files)} parquet files")

    def load_processed_files(self):
        """Load processed files from the log."""
        if os.path.exists(self.processed_files_log):
            with open(self.processed_files_log, "r") as f:
                return set(f.read().splitlines())
        return set()

    def save_processed_file(self, file_path):
        """Save a processed file to the log file, creating the directory if needed."""
        log_dir = os.path.dirname(self.processed_files_log)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        with open(self.processed_files_log, "a") as f:
            f.write(f"{file_path}\n")

    def assign_files_to_rank(self, parquet_files, evenly_distribute=True):
        """
        Assign files to the current rank based on the world size.
        This method ensures that each rank gets a unique set of files to process.
        The files can be distributed evenly based on their size (LPT algorithm) or simply by their order.
        This is useful for large datasets where some files may be significantly larger than others.

        Args:
            self (ParquetIterableDataset): The current instance.
            parquet_files (List[str]): List of file paths.
            rank (int): Current process rank.
            world_size (int): Total number of processes.
            evenly_distribute (bool): Whether to distribute files evenly based on size.

        Returns:
            List[str]: File paths assigned to the given rank.
        """

        if not evenly_distribute:
            return parquet_files[self.rank::self.world_size]

        # Get file sizes
        file_sizes = [(f, os.path.getsize(f)) for f in parquet_files]
        # Sort files by size
        file_sizes.sort(key=lambda x: x[1], reverse=True)

        assignments = defaultdict(list)
        load_per_rank = [0] * self.world_size

        for fpath, size in file_sizes:
            min_rank = load_per_rank.index(min(load_per_rank))
            assignments[min_rank].append(fpath)
            load_per_rank[min_rank] += size
        
        return assignments[self.rank]




    def parse_row(self, row):
        # Assuming decode_fn and preprocess might raise exceptions
        image = self.decode_fn(row)   # decode image to PIL
        if image is None: # Handle cases where decode_fn returns None
             raise ValueError("Decoding failed, returned None")
        image = self.preprocess(image)
        uuid = row.get('uuid', 'UUID_MISSING') # Use .get for safety
        return uuid, image

    def __iter__(self):
        """
        Iterate over the dataset, yielding (uuid, image) tuples.
        Each worker processes its assigned files and yields the results.
        The dataset is designed to be used with PyTorch's DataLoader for distributed processing.
        The function handles exceptions at various levels to ensure robust processing.
        It skips already processed files and logs errors for individual rows and batches.
        This allows for efficient and fault-tolerant processing of large datasets.
        
        Yields:
            tuple: A tuple containing the UUID and the preprocessed image tensor.
        """
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        
        # Assign files to workers
        worker_files = self.files[worker_id::num_workers]

        for path in worker_files:
            if path in self.processed_files:
                logging.info(f"[Rank {self.rank}/Worker {worker_id}] Skipping already processed file: {path}")
                continue

            logging.info(f"[Rank {self.rank}/Worker {worker_id}] Processing file: {path}")
            try:
                pf = pq.ParquetFile(path)
                for batch_idx, batch in enumerate(pf.iter_batches(batch_size=self.read_batch_size)):
                    try:
                        df = batch.to_pandas()
                        for _, row in df.iterrows():
                            try:
                                yield self.parse_row(row)
                            except Exception as e:
                                uuid = row.get('uuid', 'UUID_UNKNOWN')
                                logging.error(f"[Rank {self.rank}/Worker {worker_id}] Error parsing row UUID={uuid} in {path}: {e}", exc_info=True)
                                continue
                    except Exception as e:
                        logging.error(f"[Rank {self.rank}/Worker {worker_id}] Error in batch {batch_idx} in file {path}: {e}", exc_info=True)
                        continue
                self.save_processed_file(path)  # Mark file as processed
            except Exception as e:
                logging.error(f"[Rank {self.rank}/Worker {worker_id}] Failed to open file {path}: {e}", exc_info=True)
                continue
