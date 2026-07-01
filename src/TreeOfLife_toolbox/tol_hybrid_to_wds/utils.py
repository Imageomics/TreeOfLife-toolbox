"""
Utility helpers for converting HDF5-backed metadata rows into WebDataset shards.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Dict, Tuple

import webdataset as wds
from PIL import Image


def convert_webp_to_jpeg(webp_bytes: bytes, resize_size: int = 0) -> bytes:
    """
    Decode WebP bytes, optionally resize to a square, and re-encode as JPEG.
    """
    image = Image.open(io.BytesIO(webp_bytes)).convert("RGB")
    if resize_size and resize_size > 0:
        resample = getattr(Image, "Resampling", Image).LANCZOS  # Pillow>=9 compat
        image = image.resize((resize_size, resize_size), resample)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def determine_most_specific_known_rank(taxon_dict: Dict[str, str | None]) -> str | None:
    """
    Return the most specific rank (closest to species) whose value is not unknown.
    """
    taxonomic_ranks = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
    for rank in reversed(taxonomic_ranks):
        value = taxon_dict.get(rank)
        if value and value.strip().lower() != "unknown":
            return rank
    return None


def _clean_species_name(species_value: str | None) -> str | None:
    if not species_value:
        return None
    parts = species_value.split()
    if len(parts) > 1:
        return parts[-1]
    return species_value


def create_taxon_tag_text(taxon_dict: Dict[str, str | None], most_specific_rank: str | None) -> str:
    """
    Compose a descriptive sentence enumerating the known taxonomy ranks.
    """
    if not most_specific_rank:
        return "a photo of unknown taxonomy."

    taxonomic_ranks = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
    text_parts = []
    for rank in taxonomic_ranks:
        value = taxon_dict.get(rank)
        if not value or value.lower() == "unknown":
            if rank == most_specific_rank:
                break
            continue

        if rank == "species":
            cleaned = _clean_species_name(value)
            if cleaned:
                text_parts.append(f"{rank} {cleaned}")
        else:
            text_parts.append(f"{rank} {value}")

        if rank == most_specific_rank:
            break

    return "a photo of " + " ".join(text_parts) + "." if text_parts else "a photo of unknown taxonomy."


def create_taxonomic_name_text(taxon_dict: Dict[str, str | None], most_specific_rank: str | None) -> str:
    """
    Build a space-delimited taxonomy string up to the most specific known rank.
    """
    if not most_specific_rank:
        return ""

    taxonomic_ranks = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
    parts = []
    for rank in taxonomic_ranks:
        value = taxon_dict.get(rank)
        if not value or value.lower() == "unknown":
            if rank == most_specific_rank:
                break
            continue

        if rank == "species":
            cleaned = _clean_species_name(value)
            if cleaned:
                parts.append(cleaned)
        else:
            parts.append(value)

        if rank == most_specific_rank:
            break

    return " ".join(parts)


def create_scientific_name_text(taxon_dict: Dict[str, str | None], most_specific_rank: str | None) -> str:
    """
    Build the scientific name from the most specific known rank.
    """
    if not most_specific_rank:
        return "Unknown"

    value = taxon_dict.get(most_specific_rank)
    if not value or value.lower() == "unknown":
        return "Unknown"
    return value.rstrip(".")


def _get_common_name(taxon_dict: Dict[str, str | None]) -> str:
    value = taxon_dict.get("common_name")
    if value and value.strip():
        return value.strip().rstrip(".")
    scientific = taxon_dict.get("scientific_name")
    if scientific and scientific.strip():
        return scientific.strip().rstrip(".")
    return "Unknown"


def generate_text_files(taxon_dict: Dict[str, str | None]) -> Dict[str, str]:
    """
    Generate the five auxiliary text files used by downstream training pipelines.
    """
    most_specific_rank = determine_most_specific_known_rank(taxon_dict)
    taxonomic_name = create_taxonomic_name_text(taxon_dict, most_specific_rank)
    scientific_name = create_scientific_name_text(taxon_dict, most_specific_rank)
    taxon_for_common = dict(taxon_dict)
    taxon_for_common["scientific_name"] = scientific_name
    common_name = _get_common_name(taxon_for_common)
    taxon_tag_text = create_taxon_tag_text(taxon_dict, most_specific_rank)

    files = {
        "taxonomic_name.txt": taxonomic_name,
        "scientific_name.txt": scientific_name,
        "sci.txt": f"a photo of {scientific_name}.",
        "taxon.txt": f"a photo of {taxonomic_name}.",
        "taxonTag.txt": taxon_tag_text,
        "common_name.txt": common_name,
        "com.txt": f"a photo of {common_name}.",
        "sci_com.txt": f"a photo of {scientific_name} with common name {common_name}.",
        "taxon_com.txt": f"a photo of {taxonomic_name} with common name {common_name}.",
        "taxonTag_com.txt": f"{taxon_tag_text.rstrip('.')} with common name {common_name}.",
    }

    return files


def init_shard_writer(output_dir: str, shard_id: int, prefix: str = "shard") -> Tuple[wds.TarWriter, str]:
    """
    Create (or truncate) the shard tarball and return the TarWriter plus path.
    """
    os.makedirs(output_dir, exist_ok=True)
    shard_name = f"{prefix}-{shard_id:05d}.tar"
    shard_path = os.path.join(output_dir, shard_name)
    if os.path.exists(shard_path):
        os.remove(shard_path)

    logging.info("Creating shard %s at %s", shard_id, shard_path)
    return wds.TarWriter(shard_path), shard_path
