"""
Makes the entire set of text embeddings for all possible taxonomic names in the tree of life.
Designed for the txt_emb_species.json file from TreeOfLife-200M.

Generalized for any open_clip-compatible model accessible via Hugging Face Hub. Use
--preset for the common BioCLIP variants, or pass --model / --tokenizer / --embed-dim
to point at any other model (e.g. BioCAP, future BioCLIP releases).

Usage:
    python make_txt_embedding.py \\
        --names-path  NAMES.json \\
        --out-path    OUT.npy \\
        (--preset PRESET | --model MODEL [--tokenizer TOKENIZER] --embed-dim N) \\
        [--batch-size N]

Examples:
    # BioCLIP 2 (ViT-L-14, 768-dim) via preset
    python make_txt_embedding.py \\
        --names-path  txt_emb_bioclip-2.json \\
        --out-path    txt_emb_bioclip-2.npy \\
        --preset      bioclip-2 \\
        --batch-size  16384

    # BioCLIP 2.5 Huge (ViT-H-14, 1024-dim) via preset
    python make_txt_embedding.py \\
        --names-path  txt_emb_bioclip-2.5-vith14.json \\
        --out-path    txt_emb_bioclip-2.5-vith14.npy \\
        --preset      bioclip-2.5-vith14 \\
        --batch-size  16384

    # Arbitrary model via explicit args (e.g. BioCAP or a future release)
    python make_txt_embedding.py \\
        --names-path  txt_emb_species.json \\
        --out-path    txt_emb_custom.npy \\
        --model       hf-hub:imageomics/<model-id> \\
        --tokenizer   hf-hub:imageomics/<model-id> \\
        --embed-dim   1024 \\
        --batch-size  8192
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

# Known model presets: (model_str, tokenizer_str, embed_dim).
# --preset is a shorthand; passing --model / --tokenizer / --embed-dim overrides.
PRESETS = {
    "bioclip-2": {
        "model":     "hf-hub:imageomics/bioclip-2",
        "tokenizer": "ViT-L-14",
        "embed_dim": 768,
    },
    "bioclip-2.5-vith14": {
        "model":     "hf-hub:imageomics/bioclip-2.5-vith14",
        "tokenizer": "hf-hub:imageomics/bioclip-2.5-vith14",
        "embed_dim": 1024,
    },
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def write_txt_features(all_names, embed_dim):
    if os.path.isfile(args.out_path):
        all_features = np.load(args.out_path)
        if all_features.shape != (embed_dim, len(all_names)):
            raise SystemExit(
                f"Existing {args.out_path} has shape {all_features.shape} but expected "
                f"({embed_dim}, {len(all_names)}). Move it aside or pick a fresh --out-path."
            )
    else:
        all_features = np.zeros((embed_dim, len(all_names)), dtype=np.float32)

    batch_size = args.batch_size // len(openai_imagenet_template)
    # Ceiling division so the trailing partial batch is processed.
    num_batches = (len(all_names) + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(num_batches), desc="Extracting text features"):
        start = batch_idx * batch_size
        # Clamp final batch end to len(all_names) to avoid an IndexError on
        # the trailing partial batch.
        end = min(start + batch_size, len(all_names))
        if all_features[:, start:end].any():
            logger.info(
                "Skipping batch %d (%d to %d) because it already exists in the output file.",
                batch_idx, start, end
            )
            continue

        tmp_names = all_names[start:end]
        names = []
        for name in tmp_names:
            if len(name[1]) == 0:
                names.append(' '.join(name[0]))
            else:
                names.append(' '.join(name[0]) + ' with common name ' + name[1])

        txts = [
            template(name) for name in names for template in openai_imagenet_template
        ]
        txts = tokenizer(txts).to(device)
        txt_features = model.encode_text(txts)
        txt_features = torch.reshape(
            txt_features, (len(names), len(openai_imagenet_template), embed_dim)
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
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()),
        help="Shorthand for a known model. Overrides --model / --tokenizer / "
             "--embed-dim when set.")
    parser.add_argument("--model",
        help="open_clip model identifier (e.g. 'hf-hub:imageomics/bioclip-2'). "
             "Required unless --preset is given.")
    parser.add_argument("--tokenizer",
        help="open_clip tokenizer identifier. Defaults to --model when not set.")
    parser.add_argument("--embed-dim", type=int,
        help="Joint embedding dimension. Required unless --preset is given.")
    args = parser.parse_args()

    if args.preset:
        preset = PRESETS[args.preset]
        model_str = preset["model"]
        tokenizer_str = preset["tokenizer"]
        embed_dim = preset["embed_dim"]
    else:
        if not args.model or args.embed_dim is None:
            parser.error("either --preset or both --model and --embed-dim are required")
        model_str = args.model
        tokenizer_str = args.tokenizer or args.model
        embed_dim = args.embed_dim
    logger.info("model=%s  tokenizer=%s  embed_dim=%d", model_str, tokenizer_str, embed_dim)

    model = create_model(model_str, output_dict=True, require_pretrained=True)
    model = model.to(device)
    logger.info("Created model.")
    model = torch.compile(model)
    logger.info("Compiled model.")

    with open(args.names_path) as fd:
        names = json.load(fd)

    tokenizer = get_tokenizer(tokenizer_str)
    write_txt_features(names, embed_dim)
