"""
Makes the entire set of BioCLIP 2 text emebeddings for all possible names in the tree of life. 
Designed for the txt_emb_species.json file from TreeOfLife-200M.
"""
import argparse
import json
import os
import logging

import numpy as np
import torch
import torch.nn.functional as F

from open_clip import create_model, get_tokenizer
from tqdm import tqdm

from templates import openai_imagenet_template

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger()

model_str = "hf-hub:imageomics/bioclip-2"
tokenizer_str = "ViT-L-14"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def write_txt_features(all_names):
    if os.path.isfile(args.out_path):
        all_features = np.load(args.out_path)
    else:
        all_features = np.zeros((768, len(all_names)), dtype=np.float32)

    batch_size = args.batch_size // len(openai_imagenet_template)
    num_batches = int(len(all_names) / batch_size)
    for batch_idx in tqdm(range(num_batches), desc="Extracting text features"):
        start = batch_idx * batch_size
        end = start + batch_size
        if all_features[:, start:end].any():
            logger.info(
                "Skipping batch %d (%d to %d) because it already exists in the output file.",
                batch_idx, start, end
            )
            continue

        names = all_names[start:end]
        names = [' '.join(name[0]) + ' ' + name[1] for name in names]

        txts = [
            template(name) for name in names for template in openai_imagenet_template
        ]
        txts = tokenizer(txts).to(device)
        txt_features = model.encode_text(txts)
        txt_features = torch.reshape(
            txt_features, (len(names), len(openai_imagenet_template), 768)
        )
        txt_features = F.normalize(txt_features, dim=2).mean(dim=1)
        txt_features /= txt_features.norm(dim=1, keepdim=True)
        all_features[:, start:end] = txt_features.T.cpu().numpy()

        if batch_idx % 100 == 0:
            np.save(args.out_path, all_features)

    np.save(args.out_path, all_features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--names-path", help="Path to the taxonomic names file (e.g., txt_emb_species.json).", required=True)
    parser.add_argument("--out-path", help="Path to the output file.", required=True)
    parser.add_argument("--batch-size", help="Batch size.", default=2**14, type=int)
    args = parser.parse_args()

    model = create_model(model_str, output_dict=True, require_pretrained=True)
    model = model.to(device)
    logger.info("Created model.")
    model = torch.compile(model)
    logger.info("Compiled model.")

    with open(args.names_path) as fd:
        names = json.load(fd)

    tokenizer = get_tokenizer(tokenizer_str)
    write_txt_features(names)
