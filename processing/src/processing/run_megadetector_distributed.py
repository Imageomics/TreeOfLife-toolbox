from processing_utils import ParquetIterableDataset, setup_ddp, cleanup_ddp

import os
from pathlib import Path
import argparse

import torch
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife import utils as pw_utils
from PytorchWildlife.data import transforms as pw_trans

from decode_images import decode_image_to_pil

import torch.nn.functional as F
from torch.utils.data import DataLoader


def main(target_dir, output_dir):
    
    global_rank = int(os.environ["SLURM_PROCID"])
    local_rank = 0 # int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])

    setup_ddp(global_rank, world_size, local_rank, inference=True)

    try:
        model = pw_detection.MegaDetectorV6_Distributed(
            device=f"cuda:{local_rank}",
            pretrained=True,
            version="MDV6-yolov10-e"
        )

        parquet_files = [str(p) for p in Path(target_dir).rglob('*.parquet')]

        logs_dir = os.path.join(output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        dataset = ParquetIterableDataset(
            parquet_files=parquet_files,
            rank=global_rank,
            world_size=world_size,
            decode_fn=decode_image_to_pil,
            preprocess=pw_trans.MegaDetector_v5_Transform(target_size=640, stride=32),
            read_batch_size=128,
            processed_files_log=os.path.join(logs_dir, f"processed_files_rank{global_rank}.log"),
            evenly_distribute=True
        )

        loader = DataLoader(
            dataset,
            batch_size=128,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True
        )

        results = model.batch_image_detection(
            loader=loader,
            batch_size=128,
            global_rank=global_rank,
            local_rank=local_rank,
            output_dir=output_dir,
            det_conf_thres=0.2,
            checkpoint_frequency=1000
        )

    except Exception as e:
        print(f"[Rank {global_rank}] Caught exception: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print(f"[Rank {global_rank}] Calling cleanup_ddp()")
        cleanup_ddp(inference=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MegaDetector on parquet files in a directory")
    parser.add_argument("--target_dir", type=str, help="Directory containing parquet files")
    parser.add_argument("--output_dir", type=str, help="Directory to save output files")
    args = parser.parse_args()

    target_dir = args.target_dir
    output_dir = args.output_dir

    main(target_dir, output_dir)