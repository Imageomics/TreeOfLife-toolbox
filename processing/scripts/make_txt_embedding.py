"""
Makes the entire set of text emebeddings for all possible names in the tree of life. 
Uses the catalog.csv file from TreeOfLife-10M.
"""
import argparse
import csv
import json
import os
import logging

import numpy as np
import torch
import torch.nn.functional as F

from open_clip import create_model, get_tokenizer
from tqdm import tqdm

import lib
from templates import openai_imagenet_template

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger()

model_str = "hf-hub:imageomics/bioclip"
tokenizer_str = "ViT-B-16"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ranks = ("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species")


@torch.no_grad()
def write_txt_features(name_lookup):
    if os.path.isfile(args.out_path):
        all_features = np.load(args.out_path)
    else:
        all_features = np.zeros((512, len(name_lookup)), dtype=np.float32)

    batch_size = args.batch_size // len(openai_imagenet_template)
    for batch, (names, indices) in enumerate(
        tqdm(
            lib.batched(name_lookup.values(), batch_size),
            desc="txt feats",
            total=len(name_lookup) // batch_size,
        )
    ):
        # Skip if any non-zero elements
        if all_features[:, indices].any():
            logger.info(f"Skipping batch {batch}")
            continue

        txts = [
            template(name) for name in names for template in openai_imagenet_template
        ]
        txts = tokenizer(txts).to(device)
        txt_features = model.encode_text(txts)
        txt_features = torch.reshape(
            txt_features, (len(names), len(openai_imagenet_template), 512)
        )
        txt_features = F.normalize(txt_features, dim=2).mean(dim=1)
        txt_features /= txt_features.norm(dim=1, keepdim=True)
        all_features[:, indices] = txt_features.T.cpu().numpy()

        if batch % 100 == 0:
            np.save(args.out_path, all_features)

    np.save(args.out_path, all_features)


def convert_txt_features_to_avgs(name_lookup):
    assert os.path.isfile(args.out_path)

    # Put that big boy on the GPU. We're going fast.
    all_features = torch.from_numpy(np.load(args.out_path)).to(device)
    logger.info("Loaded text features from disk to %s.", device)

    names_by_rank = [set() for rank in ranks]
    for name, index in tqdm(name_lookup.values()):
        i = len(name) - 1
        names_by_rank[i].add((name, index))

    zeroed = 0
    for i, rank in reversed(list(enumerate(ranks))):
        if rank == "Species":
            continue
        for name, index in tqdm(names_by_rank[i], desc=rank):
            species = tuple(
                zip(
                    *(
                        (d, i)
                        for d, i in name_lookup.descendants(prefix=name)
                        if len(d) >= 6
                    )
                )
            )
            if not species:
                logger.warning("No species for %s.", " ".join(name))
                all_features[:, index] = 0.0
                zeroed += 1
                continue

            values, indices = species
            mean = all_features[:, indices].mean(dim=1)
            all_features[:, index] = F.normalize(mean, dim=0)

    out_path, ext = os.path.splitext(args.out_path)
    np.save(f"{out_path}_avgs{ext}", all_features.cpu().numpy())
    if zeroed:
        logger.warning(
            "Zeroed out %d nodes because they didn't have any genus or species-level labels.",
            zeroed,
        )


def convert_txt_features_to_species_only(name_lookup):
    assert os.path.isfile(args.out_path)

    all_features = np.load(args.out_path)
    logger.info("Loaded text features from disk.")

    species = [(d, i) for d, i in name_lookup.descendants() if len(d) == 7]
    species_features = np.zeros((512, len(species)), dtype=np.float32)
    species_names = [""] * len(species)

    for new_i, (name, old_i) in enumerate(tqdm(species)):
        species_features[:, new_i] = all_features[:, old_i]
        species_names[new_i] = name

    out_path, ext = os.path.splitext(args.out_path)
    np.save(f"{out_path}_species{ext}", species_features)
    with open(f"{out_path}_species.json", "w") as fd:
        json.dump(species_names, fd, indent=2)


def get_name_lookup(catalog_path, cache_path):
    if os.path.isfile(cache_path):
        with open(cache_path) as fd:
            lookup = lib.TaxonomicTree.from_dict(json.load(fd))
        return lookup

    lookup = lib.TaxonomicTree()

    with open(catalog_path) as fd:
        reader = csv.DictReader(fd)
        for row in tqdm(reader, desc="catalog"):
            name = [
                row["kingdom"],
                row["phylum"],
                row["class"],
                row["order"],
                row["family"],
                row["genus"],
                row["species"],
            ]
            if any(not value for value in name):
                name = name[: name.index("")]
            lookup.add(name)

    with open(args.name_cache_path, "w") as fd:
        json.dump(lookup, fd, cls=lib.TaxonomicJsonEncoder)

    return lookup


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog-path",
        help="Path to the catalog.csv file from TreeOfLife-10M.",
        required=True,
    )
    parser.add_argument("--out-path", help="Path to the output file.", required=True)
    parser.add_argument(
        "--name-cache-path",
        help="Path to the name cache file.",
        default="name_lookup.json",
    )
    parser.add_argument("--batch-size", help="Batch size.", default=2**15, type=int)
    args = parser.parse_args()

    name_lookup = get_name_lookup(args.catalog_path, cache_path=args.name_cache_path)
    logger.info("Got name lookup.")

    model = create_model(model_str, output_dict=True, require_pretrained=True)
    model = model.to(device)
    logger.info("Created model.")
    model = torch.compile(model)
    logger.info("Compiled model.")

    tokenizer = get_tokenizer(tokenizer_str)
    write_txt_features(name_lookup)
    convert_txt_features_to_avgs(name_lookup)
    convert_txt_features_to_species_only(name_lookup)
