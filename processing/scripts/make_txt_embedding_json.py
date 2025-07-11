#!/usr/bin/env python3
"""
Generate txt_emb_species.json from TreeOfLife-200M catalog data.

This script creates a JSON file for species embeddings by:
- Filtering catalog to only entries with non-null kingdom and non-null species
- For each remaining unique taxonomy, collecting all available common names
- Preferring English common names from GBIF VernacularNames.tsv (from GBIF Backbone Taxonomy), falling back to any language
- Sorting by taxonomy and outputting in [[taxonomy_array], common_name] format

Usage:
    python make_txt_embedding_json.py CATALOG_PATH VERNACULAR_PATH [-o OUTPUT_PATH]

Example:
    python make_txt_embedding_json.py \\
        catalog.parquet \\
        VernacularNames.tsv \\
        -o txt_emb_species.json
"""

import polars as pl
import json
import argparse
from pathlib import Path

def load_vernacular_names(vernacular_path: str) -> tuple[set, set]:
    """Load and return sets of vernacular names from GBIF's TSV file."""
    print(f"Loading vernacular names from: {vernacular_path}")
    
    # Get the vernacular name data
    df_vernacular = pl.read_csv(
        vernacular_path,
        separator="\t",
        quote_char=None,
        ignore_errors=True
    )
    
    # Get English vernacular names (ignoring case)
    english_names = (
        df_vernacular
        .filter(pl.col("language") == "en")
        ["vernacularName"]
        .drop_nulls()
        .str.to_lowercase()
        .unique()
        .to_list()
    )
    
    # Get all vernacular names (ignoring case)
    all_names = (
        df_vernacular["vernacularName"]
        .drop_nulls()
        .str.to_lowercase()
        .unique()
        .to_list()
    )
    
    print(f"\tLoaded {len(english_names)} English and {len(all_names)} total vernacular names")
    return set(english_names), set(all_names)


def select_best_common_name_from_list(names_list, english_names: set, all_names: set) -> str:
    """Select a common name from a list preferring English names."""
    if not names_list:
        return ""
        
    # Try to find English names
    for name in names_list:
        if name and name.lower() in english_names:
            return name
    
    # If no English names found use any language
    for name in names_list:
        if name and name.lower() in all_names:
            return name
            
    # If nothing matches the vernacular names return empty
    return ""


def process_catalog_to_embeddings(catalog_path: str, english_names: set, all_names: set, output_path: str):
    """Process catalog data into embeddings JSON format."""
    print(f"Loading catalog from: {catalog_path}")
    
    # Load catalog  
    df_catalog = pl.read_parquet(catalog_path)
    print(f"\tTotal catalog entries: {len(df_catalog)}")
    
    # Filter to only keep entries with non-null kingdom AND species
    df_filtered = df_catalog.filter(
        (pl.col("kingdom").is_not_null()) & 
        (pl.col("species").is_not_null())
    )
    print(f"\tAfter null kingdom/species filtering: {len(df_filtered)}")
    
    # Get all unique taxonomies with their common names from the catalog
    df_grouped = (
        df_filtered
        .group_by(["kingdom", "phylum", "class", "order", "family", "genus", "species"])
        .agg([
            pl.col("common").drop_nulls().unique().alias("all_common_names")
        ])
    )

    print(f"\tAfter taxonomic deduplication: {len(df_grouped)} unique taxonomies")
    
    # Select 'best' common name for each unique species
    print("Processing common names with English preference")

    # Convert to list of dictionaries for processing
    grouped_data = df_grouped.to_dicts()
    
    # Process each group to select a preferred common name
    processed_rows = []
    for row in grouped_data:
        taxonomy = [row["kingdom"], row["phylum"], row["class"], row["order"], 
                   row["family"], row["genus"], row["species"]]
        common_names = row["all_common_names"] or []
        
        # Select 'best' common name
        best_common = select_best_common_name_from_list(common_names, english_names, all_names)
        
        # Create a fully processed row
        processed_row = {
            "kingdom": row["kingdom"],
            "phylum": row["phylum"], 
            "class": row["class"],
            "order": row["order"],
            "family": row["family"],
            "genus": row["genus"],
            "species": row["species"],
            "verified_common": best_common
        }
        processed_rows.append(processed_row)
    
    # Convert back to Polars DataFrame
    df_processed = pl.DataFrame(processed_rows)
    
    print(f"\tAfter processing: {len(df_processed)}")
    
    # Select only the columns we need for JSON
    df_for_json = df_processed.select([
        "kingdom", "phylum", "class", "order", "family", "genus", "species", "verified_common"
    ])
    # Sort by taxonomy for consistent output
    df_for_json = df_for_json.sort(["kingdom", "phylum", "class", "order", "family", "genus", "species"])
    
    # Convert to JSON format
    print("Converting to JSON format")
    json_data = []
    
    for row in df_for_json.iter_rows():
        # Create taxonomy array (first 7 columns), nulls => empty strings
        taxonomy = [col if col is not None else "" for col in row[:7]]
        
        # Get verified common name
        common_name = row[7] if row[7] is not None else ""
        
        json_data.append([taxonomy, common_name])
    
    # Save compact JSON with UTF-8
    print(f"Saving {len(json_data)} entries to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False)  # Compact, UTF-8
    
    return len(json_data)


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings JSON from TreeOfLife-200M catalog data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "catalog_path",
        help="Path to the catalog.parquet file"
    )
    
    parser.add_argument(
        "vernacular_path", 
        help="Path to the VernacularNames.tsv file (from the GBIF Backbone Taxonomy https://doi.org/10.15468/39omei)"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="txt_emb_species.json",
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    # Check input files exist
    if not Path(args.catalog_path).exists():
        raise FileNotFoundError(f"Catalog file not found: {args.catalog_path}")
    if not Path(args.vernacular_path).exists():
        raise FileNotFoundError(f"VernacularNames file not found: {args.vernacular_path}")
    
    print(f"Catalog: {args.catalog_path}")
    print(f"Vernacular names: {args.vernacular_path}")
    print(f"Output: {args.output}")
    
    # Load vernacular names
    english_names, all_names = load_vernacular_names(args.vernacular_path)
    
    # Process catalog, generate JSON
    entry_count = process_catalog_to_embeddings(args.catalog_path, english_names, all_names, args.output)
    
    print(f"\nEmbeddings JSON complete.")


if __name__ == "__main__":
    main()
