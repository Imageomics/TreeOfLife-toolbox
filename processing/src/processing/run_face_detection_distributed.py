"""
Distributed Face Detection

This script performs distributed face detection with the MTCNN model for images stored in parquet files. 

Features:
- Supports multi-GPU and multi-node distributed processing
- Processes binary image data from parquet files
- Uses MTCNN to detect faces in the images
- Checkpoints results periodically to handle large datasets safely

Usage:
    Run with torchrun or similar distributed launcher:
    srun python run_face_detection_distributed.py
        --target_dir={PARQUET_DIR}
        --output_dir={OUTPUT_DIR}
    `scripts/run_face_detection_distributed.slurm` is provided for jobs execution.

Performance Tuning:
- Batch size and worker count can be adjusted based on available resources
- For low GPU utilization, increase `batch_size`
- For slow data loading, increase `num_workers`

Requirements:
- PyTorch with CUDA support
- Parquet reading capabilities
- `face_pytorch` Python package

References:
@article{zhang2016joint,
  title={Joint face detection and alignment using multitask cascaded convolutional networks},
  author={Zhang, Kaipeng and Zhang, Zhanpeng and Li, Zhifeng and Qiao, Yu},
  journal={IEEE signal processing letters},
  volume={23},
  number={10},
  pages={1499--1503},
  year={2016},
  publisher={IEEE}
}
"""

import os
from pathlib import Path
import time
import argparse
import numpy as np
import pandas as pd
from facenet_pytorch import MTCNN
from facenet_pytorch.models.utils.detect_face import imresample, generateBoundingBox, batched_nms, rerec, pad, fixed_batch_process, bbreg, batched_nms_numpy

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import DataLoader

from decode_images import decode_image_to_pil
from processing_utils import ParquetIterableDataset, format_time, setup_ddp, cleanup_ddp


def detect_face(imgs, minsize, pnet, rnet, onet, threshold, factor):
    h, w = imgs.shape[2:4]
    m = 12.0 / minsize
    minl = min(h, w)
    minl = minl * m

    # Create scale pyramid
    scale_i = m
    scales = []
    while minl >= 12:
        scales.append(scale_i)
        scale_i = scale_i * factor
        minl = minl * factor

    # First stage
    boxes = []
    image_inds = []
    scale_picks = []
    offset = 0

    for scale in scales:
        im_data = imresample(imgs, (int(h * scale + 1), int(w * scale + 1)))
        im_data = (im_data - 127.5) * 0.0078125
        reg, probs = pnet(im_data)
    
        boxes_scale, image_inds_scale = generateBoundingBox(reg, probs[:, 1], scale, threshold[0])
        boxes.append(boxes_scale)
        image_inds.append(image_inds_scale)

        pick = batched_nms(boxes_scale[:, :4], boxes_scale[:, 4], image_inds_scale, 0.5)
        scale_picks.append(pick + offset)
        offset += boxes_scale.shape[0]

    boxes = torch.cat(boxes, dim=0)
    image_inds = torch.cat(image_inds)
    scale_picks = torch.cat(scale_picks, dim=0)
    boxes, image_inds = boxes[scale_picks], image_inds[scale_picks]

    # NMS within each image
    pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
    boxes, image_inds = boxes[pick], image_inds[pick]

    regw = boxes[:, 2] - boxes[:, 0]
    regh = boxes[:, 3] - boxes[:, 1]
    qq1 = boxes[:, 0] + boxes[:, 5] * regw
    qq2 = boxes[:, 1] + boxes[:, 6] * regh
    qq3 = boxes[:, 2] + boxes[:, 7] * regw
    qq4 = boxes[:, 3] + boxes[:, 8] * regh
    boxes = torch.stack([qq1, qq2, qq3, qq4, boxes[:, 4]]).permute(1, 0)
    boxes = rerec(boxes)
    y, ey, x, ex = pad(boxes, w, h)
    
    # Second stage
    if len(boxes) > 0:
        im_data = []
        for k in range(len(y)):
            if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                img_k = imgs[image_inds[k], :, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
                im_data.append(imresample(img_k, (24, 24)))
        im_data = torch.cat(im_data, dim=0)
        im_data = (im_data - 127.5) * 0.0078125

        # This is equivalent to out = rnet(im_data) to avoid GPU out of memory.
        out = fixed_batch_process(im_data, rnet)

        out0 = out[0].permute(1, 0)
        out1 = out[1].permute(1, 0)
        score = out1[1, :]
        ipass = score > threshold[1]
        boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
        image_inds = image_inds[ipass]
        mv = out0[:, ipass].permute(1, 0)

        # NMS within each image
        pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
        boxes, image_inds, mv = boxes[pick], image_inds[pick], mv[pick]
        boxes = bbreg(boxes, mv)
        boxes = rerec(boxes)

    # Third stage
    points = torch.zeros(0, 5, 2).to(imgs.device)
    if len(boxes) > 0:
        y, ey, x, ex = pad(boxes, w, h)
        im_data = []
        for k in range(len(y)):
            if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                img_k = imgs[image_inds[k], :, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
                im_data.append(imresample(img_k, (48, 48)))
        im_data = torch.cat(im_data, dim=0)
        im_data = (im_data - 127.5) * 0.0078125
        
        # This is equivalent to out = onet(im_data) to avoid GPU out of memory.
        out = fixed_batch_process(im_data, onet)

        out0 = out[0].permute(1, 0)
        out1 = out[1].permute(1, 0)
        out2 = out[2].permute(1, 0)
        score = out2[1, :]
        points = out1
        ipass = score > threshold[2]
        points = points[:, ipass]
        boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
        image_inds = image_inds[ipass]
        mv = out0[:, ipass].permute(1, 0)

        w_i = boxes[:, 2] - boxes[:, 0] + 1
        h_i = boxes[:, 3] - boxes[:, 1] + 1
        points_x = w_i.repeat(5, 1) * points[:5, :] + boxes[:, 0].repeat(5, 1) - 1
        points_y = h_i.repeat(5, 1) * points[5:10, :] + boxes[:, 1].repeat(5, 1) - 1
        points = torch.stack((points_x, points_y)).permute(2, 1, 0)
        boxes = bbreg(boxes, mv)

        # NMS within each image using "Min" strategy
        pick = batched_nms_numpy(boxes[:, :4], boxes[:, 4], image_inds, 0.7, 'Min')
        boxes, image_inds, points = boxes[pick], image_inds[pick], points[pick]

    boxes = boxes.cpu().numpy()
    points = points.cpu().numpy()

    image_inds = image_inds.cpu()

    batch_boxes = []
    batch_points = []
    for b_i in range(imgs.shape[0]):
        b_i_inds = np.where(image_inds == b_i)
        batch_boxes.append(boxes[b_i_inds].copy())
        batch_points.append(points[b_i_inds].copy())

    batch_boxes, batch_points = np.array(batch_boxes, dtype=object), np.array(batch_points, dtype=object)

    scores = np.zeros((imgs.shape[0]))
    for sample_index, boxes in enumerate(batch_boxes):
        if len(boxes) > 0:
            score = boxes[:, 4].mean()
            if score > 0.985:
                scores[sample_index] = score
    return scores

@torch.no_grad()
def run_face_detection(model, loader, global_rank, local_rank, output_dir, checkpoint_frequency = 100):

    model.eval()
    results = []

    # Create checkpoint directory
    checkpoint_dir = os.path.join(output_dir, f"checkpoints_rank{global_rank}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    batch_counter = 0
    processed_count = 0
    start_time = time.time()

    for uuids, images in loader:

        batch_counter += 1
        processed_count += len(images)

        # Move images from CPU to GPU{rank}
        images = images.cuda(local_rank, non_blocking=True) # Local
        images = images * 255

        pred_scores = detect_face(
            images, 
            minsize=model.min_face_size,
            pnet=model.pnet,
            rnet=model.rnet, 
            onet=model.onet,
            threshold=model.thresholds,
            factor=model.factor
        )
        
        for uuid, score in zip(uuids, pred_scores):
            results.append({
                "uuid": uuid,
                "pred_label": float(score) > 0.,
                "pred_score": float(score)
            })

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


def main(target_dir: str, output_dir: str, file_paths_parquet:str = None):

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

    # Create a face detection pipeline using MTCNN:
    model = MTCNN(
        image_size=160, margin=0, thresholds=[0.9, 0.95, 0.95],
        device=f"cuda:{local_rank}"
    )

    preprocess = T.Compose([
        T.Resize((720, 720)),
        T.ToTensor()
    ])

    if file_paths_parquet is not None:
        # Use the provided parquet file paths
        print(f"Reading file paths from {file_paths_parquet}")
        df_file_paths = pd.read_parquet(file_paths_parquet)
        parquet_files = df_file_paths['path'].tolist()
    else:
        parquet_files = [str(p) for p in Path(target_dir).rglob('*.parquet')]
    
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    dataset = ParquetIterableDataset(
        parquet_files=parquet_files,
        rank=global_rank,
        world_size=world_size,                      # Set to 1 to read all files
        decode_fn=decode_image_to_pil,
        preprocess=preprocess,
        read_batch_size=128,
        processed_files_log=os.path.join(logs_dir, f"processed_files_rank{global_rank}.log"),
        evenly_distribute=True
    )

    # Low GPU usage -> Increase batch_size
    # High GPU usage, slow load -> Increase num_workers
    
    loader = DataLoader(
        dataset,
        batch_size=128,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True
    )

    run_face_detection(
        model=model,
        loader=loader,
        global_rank=global_rank,
        local_rank=local_rank,
        output_dir=output_dir,
        checkpoint_frequency=1000
    )

    cleanup_ddp(inference=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Face Detection on parquet files in a directory")
    parser.add_argument("--target_dir", type=str, required=True, help="Directory containing parquet files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output")
    parser.add_argument("--file_paths_parquet", type=str, required=False, help="Path to parquet file containing data paths in `path` column")

    args = parser.parse_args()

    main(args.target_dir, args.output_dir, args.file_paths_parquet)
