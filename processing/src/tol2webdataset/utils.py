import logging
import os
import re
import shutil
import subprocess
from typing import Sequence, List

import numpy as np
import webdataset as wds
import cv2


def init_logger(logger_name: str, output_path: str = None, logging_level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        filename=output_path,
        level=logging.getLevelName(logging_level),
        format="%(asctime)s - %(levelname)s - %(process)d - %(message)s")
    return logging.getLogger(logger_name)


def resize_image(image_bytes, original_height, original_width, target_size):
    """
    Resize a raw BGR image to the target size and encode it as JPEG.

    Parameters:
    - image_bytes (bytes): The raw BGR image data.
    - original_height (int): The original height of the image (as saved in the Parquet file).
    - original_width (int): The original width of the image (as saved in the Parquet file).
    - target_size (int): The target size for both width and height (square).

    Returns:
    - bytes: The resized image in JPEG format.
    """
    try:
        # Convert raw bytes to NumPy array and reshape to (height, width, 3)
        img = np.frombuffer(image_bytes, dtype=np.uint8).reshape((original_height, original_width, 3))

        # Resize the image to (target_size, target_size)
        # Using INTER_CUBIC for downscaling and INTER_INTER_LANCZOS4 for upscaling
        if target_size < min(original_height, original_width):
            interpolation = cv2.INTER_CUBIC
        else:
            interpolation = cv2.INTER_LANCZOS4
        resized_img = cv2.resize(img, (target_size, target_size), interpolation=interpolation)

        # Encode the resized image to JPEG format
        success, encoded_img = cv2.imencode('.jpg', resized_img)
        if not success:
            raise ValueError("Image encoding failed.")

        return encoded_img.tobytes()
    except Exception as e:
        logging.warning(f"Image resizing failed: {e}")
        return image_bytes  # Return original if resizing fails


def determine_most_specific_known_rank(taxon_dict):
    """
    Determine the most specific known rank that is not 'Unknown'.

    Parameters:
        taxon_dict (dict): Dictionary of ranks and names.

    Returns:
        str: The most specific known rank.
    """
    taxonomic_ranks = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    for rank in reversed(taxonomic_ranks):
        name = taxon_dict.get(rank)
        if name and name.lower() != 'unknown':
            return rank
    return None


def create_taxon_tag_text(taxon_dict, most_specific_rank):
    """
    Create the taxonTag text description based on the taxon dictionary.
    Format: "a photo of kingdom <kingdom> phylum <phylum> ... genus <genus> species <specific epithet>."
    Missing ranks are completely excluded.

    Parameters:
        taxon_dict (dict): Dictionary of ranks and names.
        most_specific_rank (str): The most specific known rank.

    Returns:
        str: The formatted taxonTag text.
    """
    if not most_specific_rank:
        return "a photo of unknown taxonomy."

    taxonomic_ranks = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    text_parts = []

    # Include only known ranks up to and including the most specific rank
    for rank in taxonomic_ranks:
        if rank == most_specific_rank and most_specific_rank == 'species':
            # Handle the special case where species is the most specific rank
            species_name = taxon_dict.get('species', '')
            if species_name and species_name.lower() != 'unknown':
                species_parts = species_name.split()
                if len(species_parts) > 1:
                    text_parts.append(f"{rank} {species_parts[-1]}")
            break

        name = taxon_dict.get(rank, '')
        if name and name.lower() != 'unknown':
            if rank == 'species':
                # For species, only use the specific epithet
                species_parts = name.split()
                if len(species_parts) > 1:
                    name = species_parts[-1]
            text_parts.append(f"{rank} {name}")

        if rank == most_specific_rank:
            break

    return "a photo of " + " ".join(text_parts) + "." if text_parts else "a photo of unknown taxonomy."


def create_taxonomic_name_text(taxon_dict, most_specific_rank):
    """
    Create the taxonomic name text based on the taxon dictionary.
    Format: "<kingdom> <phylum> ... <genus> <specific epithet>"
    Missing ranks are completely excluded.

    Parameters:
        taxon_dict (dict): Dictionary of ranks and names.
        most_specific_rank (str): The most specific known rank.

    Returns:
        str: The taxonomic name text.
    """
    if not most_specific_rank:
        return ""

    taxonomic_ranks = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    name_parts = []

    # Include only known ranks up to and including the most specific rank
    for rank in taxonomic_ranks:
        if rank == most_specific_rank and most_specific_rank == 'species':
            # Handle the special case where species is the most specific rank
            species_name = taxon_dict.get('species', '')
            if species_name and species_name.lower() != 'unknown':
                species_parts = species_name.split()
                if len(species_parts) > 1:
                    name_parts.append(species_parts[-1])
            break

        name = taxon_dict.get(rank, '')
        if name and name.lower() != 'unknown':
            if rank == 'species':
                # For species, only use the specific epithet
                species_parts = name.split()
                if len(species_parts) > 1:
                    name = species_parts[-1]
            name_parts.append(name)

        if rank == most_specific_rank:
            break

    return " ".join(name_parts)


def create_scientific_name_text(taxon_dict, most_specific_rank):
    """
    Create the scientific name text based on either the provided scientific name
    or the most specific known rank.
    Format: "<scientific name>" or "<genus> <specific epithet>"

    Parameters:
        taxon_dict (dict): Dictionary containing taxonomic ranks and names.
        most_specific_rank (str): The most specific known rank.

    Returns:
        str: The scientific name.
    """
    scientific_name = taxon_dict.get('scientific_name')
    if scientific_name:
        return scientific_name.rstrip('.')

    if not most_specific_rank:
        return "Unknown"

    name = taxon_dict.get(most_specific_rank, '')
    if not name or name.lower() == 'unknown':
        return 'Unknown'
    return name


def generate_text_files(taxon_dict, most_specific_rank):
    """
    Generates the content for the additional text files based on the taxonomic information.

    Parameters:
        taxon_dict (dict): Dictionary containing taxonomic ranks and names.
        most_specific_rank (str): The most specific known rank.

    Returns:
        dict: A dictionary with file extensions as keys and their corresponding content as values.
    """
    additional_files = {}

    # Create taxonomic_name.txt
    taxonomic_name = create_taxonomic_name_text(taxon_dict, most_specific_rank)
    additional_files['taxonomic_name.txt'] = taxonomic_name

    # Create scientific_name.txt
    scientific_name = create_scientific_name_text(taxon_dict, most_specific_rank)
    additional_files['scientific_name.txt'] = scientific_name

    # Create sci.txt
    sci_content = f"a photo of {scientific_name}."
    additional_files['sci.txt'] = sci_content

    # Create taxon.txt
    taxon_content = f"a photo of {taxonomic_name}."
    additional_files['taxon.txt'] = taxon_content

    # Create taxonTag.txt
    taxon_tag_text = create_taxon_tag_text(taxon_dict, most_specific_rank)
    additional_files["taxonTag.txt"] = taxon_tag_text

    return additional_files


def init_shard(output_dir: str, shard_index: int) -> wds.TarWriter:
    """
    Initialize a new shard writer.

    Parameters:
        output_dir: Directory where shards are saved
        shard_index: Index number of the shard

    Returns:
        Tuple of (TarWriter)
    """
    shard_name = f"shard-{shard_index:05d}.tar"
    shard_path = os.path.join(output_dir, shard_name)
    if os.path.exists(shard_path):
        os.remove(shard_path)
    writer = wds.TarWriter(shard_path)
    logging.info(f"Creating shard {shard_index}: {shard_path}")
    return writer


def ensure_created(list_of_path: Sequence[str]) -> None:
    for path in list_of_path:
        os.makedirs(path, exist_ok=True)


def truncate_paths(paths: Sequence[str]) -> None:
    for path in paths:
        is_dir = "." not in path.split("/")[-1]
        if is_dir:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
        else:
            open(path, "w").close()


def get_id(output: bytes) -> int:
    return int(output.decode().strip().split(" ")[-1])


def submit_job(submitter_script: str, script: str, *args) -> int:
    output = subprocess.check_output(f"{submitter_script} {script} {' '.join(args)}", shell=True)
    idx = get_id(output)
    return idx


def preprocess_dep_ids(ids: List[int | None]) -> List[str]:
    return [str(_id) for _id in ids if _id is not None]


def logger_folder_count(log_path: str) -> int:
    all_path = os.listdir(log_path)
    i = 0
    for path in all_path:
        if (not os.path.isdir(os.path.join(log_path, path))
                or re.fullmatch(r"\d{4}", path) is not None):
            continue
        i += 1

    return i
