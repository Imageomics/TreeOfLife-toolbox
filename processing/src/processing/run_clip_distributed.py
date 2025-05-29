"""
Distributed CLIP Inference for Image Classification

This script performs distributed inference using the CLIP model to classify images
stored in parquet files. 

Features:
- Supports multi-GPU and multi-node distributed processing
- Processes binary image data from parquet files
- Uses CLIP embeddings and pre-computed class embeddings for classification
- Measures standardized distances between image embeddings and class embeddings
- Checkpoints results periodically to handle large datasets safely

Usage:
    Run with torchrun or similar distributed launcher:
    srun python run_clip_distributed.py 
        --target_dir={PARQUET_DIR} 
        --output_dir={OUTPUT_DIR} 
        --class_embeddings_dict_path={EMBEDDINGS_PATH}
    `scripts/run_clip_distributed.slurm` is provided for jobs execution

Performance Tuning:
- Batch size and worker count can be adjusted based on available resources
- For low GPU utilization, increase `batch_size`
- For slow data loading, increase `num_workers`

Requirements:
- PyTorch with CUDA support
- CLIP
- Parquet reading capabilities
see `docs/create_clip_venv.md` for virtual environment setup
"""
from processing_utils import ParquetIterableDataset, setup_ddp, cleanup_ddp, format_time

import os
from pathlib import Path
import time
import argparse

import torch
import clip
import pandas as pd

from decode_images import decode_image_to_pil

import torch.nn.functional as F
from torch.utils.data import DataLoader

@torch.no_grad()
def run_CLIP_classification(model, loader, class_embeddings_dict, global_rank, local_rank, output_dir, checkpoint_frequency=100):
    """
    Args:
        model: CLIP model (unwrapped from DDP).
        loader: DataLoader yielding (uuids, images) batches.
        class_embeddings_dict: Dict[str -> Tensor]
        global_rank: Global rank of the process.
        local_rank: Local rank of the process.
        output_dir: Directory to save predictions.
        checkpoint_frequency: Save intermediate results every N batches
    """
    # Evaluation mode
    # Ensures deterministic, stable outputs during evaluation/inference
    model.eval()
    results = []

    # Create checkpoint directory
    # Track batches and processed items
    checkpoint_dir = os.path.join(output_dir, f"checkpoints_rank{global_rank}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    batch_counter = 0
    processed_count = 0
    start_time = time.time()
    

    # Prepare label list and class embedding matrix
    class_labels = list(class_embeddings_dict.keys())  # e.g., ['file', 'specimen', 'plants']
    # [num_classes, embedding_dim]
    class_embeddings = F.normalize(
        torch.stack(list(class_embeddings_dict.values())).squeeze(1),
        dim=-1
    ).to(local_rank)  # Local 

    for uuids, images in loader:
        batch_counter += 1
        processed_count += len(images)

        # Move images from CPU to GPU{rank}
        images = images.cuda(local_rank, non_blocking=True) # Local

        # Get normalized image embeddings
        image_embeddings = F.normalize(
            model.encode_image(images),
            dim=-1
        )  # [batch_size, embedding_dim]

        del images

        # Compute cosine similarity: [batch_size, num_classes]
        image_embeddings = image_embeddings.to(torch.float32)
        class_embeddings = class_embeddings.to(torch.float32)
        similarity = image_embeddings @ class_embeddings.T

        del image_embeddings

        # Top-1 prediction for each image
        pred_scores, pred_indices = similarity.max(dim=1)

        for uuid, pred_idx, score in zip(uuids, pred_indices.cpu(), pred_scores.cpu()):
            results.append({
                "uuid": uuid,
                "pred_label": class_labels[pred_idx],
                "pred_score": float(score)
            })
        
        del similarity, pred_scores, pred_indices
        torch.cuda.empty_cache()  # Free up GPU memory

        # Save checkpoint periodically
        if batch_counter % checkpoint_frequency == 0:
            elapsed = time.time() - start_time
            elapsed_str = format_time(elapsed)
            print(f"[Rank {global_rank}] Processed {processed_count} images in {elapsed_str}")
            
            # Save intermediate results
            checkpoint_path = os.path.join(
                checkpoint_dir, 
                f"checkpoint_{batch_counter:06d}.parquet"
            )
            
            df = pd.DataFrame(results)
            df.to_parquet(checkpoint_path, index=False)
            print(f"[Rank {global_rank}] Saved checkpoint to {checkpoint_path}")


    # Save results to disk
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results)
    out_path = os.path.join(output_dir, f"predictions_rank{global_rank}.parquet")
    df.to_parquet(out_path, index=False)
    print(f"[rank {global_rank}] Saved predictions to {out_path}")



def main(target_dir, output_dir, class_embeddings_dict_path):

    # Get rank, local rank, world size
    # torchrun DDP
    # global_rank = int(os.environ["RANK"])
    # local_rank = int(os.environ["LOCAL_RANK"])
    # world_size = int(os.environ["WORLD_SIZE"])
    
    # srun 
    global_rank = int(os.environ["SLURM_PROCID"])
    local_rank = 0 # int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])

    setup_ddp(global_rank, world_size, local_rank, inference=True)

    # Initialize CLIP model and preprocess function
    model, preprocess = clip.load("ViT-L/14@336px", device=f"cuda:{local_rank}") # Local
    # model = torch.nn.parallel.DistributedDataParallel(
    #     model.to(f"cuda:{local_rank}"),
    #     device_ids=[local_rank]
    # )  


    # Get all nested parquet files in the target directory    
    parquet_files = [str(p) for p in Path(target_dir).rglob('*.parquet')]
    
    # Initialize dataset and loader
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    dataset = ParquetIterableDataset(
        parquet_files=parquet_files,
        rank=global_rank,                       # Use global rank for data distribution
        world_size=world_size,
        decode_fn=decode_image_to_pil,
        preprocess=preprocess,
        read_batch_size=128,
        processed_files_log=os.path.join(logs_dir, f"processed_files_rank{global_rank}.log"),
        evenly_distribute=True
    )

    # Low GPU usage -> Increase batch_size
    # High GPU usage, slow load -> Increase num_workers
    batch_size = 256         # TODO: Consider scaling with GPU memory as well 
    num_workers = 16          # min(8, os.cpu_count() // world_size)  # Scale with CPU cores
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True
    )

    # Classification
    class_embeddings_dict = torch.load(class_embeddings_dict_path)

    run_CLIP_classification(
        model,                
        loader, class_embeddings_dict,
        global_rank, local_rank,
        output_dir,
        checkpoint_frequency = 1000  # Save Intermediate results every 1000 batches
    )

    cleanup_ddp(inference=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", type=str, required=True, help="Directory containing parquet files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save predictions.")
    parser.add_argument("--class_embeddings_dict_path", type=str, required=True, help="Path to class embeddings dict.")
    args = parser.parse_args()

    main(args.target_dir, args.output_dir, args.class_embeddings_dict_path)