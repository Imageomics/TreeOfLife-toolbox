"""
create_lookup_tbl.py

TOL-200M images are stored across different Parquet files, and the lookup table is created to map the UUIDs to the filepaths.
This script creates lookup tables for a few pre-specified groups by filtering and joining relevant data, and writes the result to Parquet files.
The unique number of image data files in one lookup table could be huge, causing out-of-memory issues when using the lookup table for downstream tasks such as extracting subset from the TOL-200M dataset.
To address this issue, the lookup table is partitioned into batches that contain at most `n_max_files_per_batch` data filepaths. 

Usage:
    python create_lookup_tbl.py <group> <output_path> [--n_max_files_per_batch <n_max_files_per_batch>]

Arguments:
    group (str): The group to create the lookup table for. Must be one of:
        "museum_specimen_all": All museum specimen records.
        "multi_occ_museum_specimen": Museum specimen records with multiple images per occurrence.
        "citizen_science": Citizen science records.
        "citizen_science_fish": Citizen science records with fish images.
        "questagame_birds": Questagame records with taxonRemarks as Birds.
        "questagame_plants_that_flowers": Questagame records with taxonRemarks as Plants that flower.
        "questagame_butterflies_moths": Questagame records with taxonRemarks as Insects - Butterflies and Moths.
        "yale_plants": Yale University Peabody Museum records with taxonRemarks as Animals and Plants: Plants.
    output_path (str): The output path to save the lookup table.
    --n_max_files_per_batch (int, optional): The maximum number of image data files to load per batch. Defaults to 100.

Dependencies:
    - os
    - gc
    - logging
    - argparse
    - pyspark
    - typing

Functions:
    init_spark():
        Initializes and returns a SparkSession.

    write_lookup_tbl(spark, lookup_tbl, output_path, n_max_files_per_batch=100):
        Takes a lookup table, partitions it into batches that contain at most n_max_files_per_batch data filepaths, and writes the result to Parquet.

    create_lookup_tbl(spark, group):
        Creates a lookup table for a pre-specified group by filtering and joining relevant data.

    main(group, output_path):
        Main function to create and save the lookup table for the specified group.

Example:
    python create_lookup_tbl.py multi_occ_museum_specimen /path/to/output --n_max_files_per_batch 50
"""

import os
import gc
import logging
import argparse
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import broadcast, col, countDistinct, row_number, floor
from pyspark.sql.window import Window
from typing import Literal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


N_EXECUTORS = 32

COLS_TAXONOMIC = [
    "kingdom", "phylum", "class", "order", "family", "genus", "species"
]

fish_classes = [
    "Agnatha",                    # Jawless fish
    "Myxini",                     # Hagfish
    "Pteraspidomorphi",           # Early jawless fish (extinct)
    "Thelodonti",                 # Extinct
    "Anaspida",                   # Extinct
    "Petromyzontida",             # Lampreys
    "Hyperoartia",
    "Conodonta",                  # Conodonts (extinct)
    "Cephalaspidomorphi",         # Early jawless fish (extinct)
    "Placodermi",                 # Armoured fish (extinct)
    "Acanthodii",                 # Spiny sharks (extinct)
    "Actinopterygii",             # Ray-finned fish
    "Sarcopterygii"               # Lobe-finned fish
    "Chondrichthyes",             # cartilaginous fish
    "Sarcopterygii"              # lobe-finned fish
]

fish_orders = [
    # Class Myxini
    "Myxiniformes",
    # Class Cephalaspidomorphi
    "Petromyzontiformes",
    
    # Class Chondrichthyes (Cartilaginous Fishes)
    "Selachii",          # Sharks
    "Batoidei",          # Rays, sawfishes, guitarfishes, skates, stingrays
    "Chimaeriformes",    # Chimaeras

    # Class Actinopterygii (Ray-Finned Fishes)
    # Subclass Chondrostei
    "Acipenseriformes",  # Sturgeons and paddlefishes
    "Polypteriformes",   # Bichirs and reedfish

    # Infraclass Holostei
    "Amiiformes",        # Bowfins
    "Semionotiformes",   # Gars

    # Infraclass Teleostei (Advanced Bony Fishes)
    # Superorder Osteoglossomorpha
    "Osteoglossiformes", # Bonytongues, mooneyes, knife fishes, mormyrs

    # Superorder Elopomorpha
    "Elopiformes",       # Ladyfishes and tarpons
    "Albuliformes",      # Bonefishes
    "Anguilliformes",    # eels
    "Saccopharyngiformes",# gulpers

    # Superorder Clupeomorpha
    "Clupeiformes",      # Herrings and anchovies

    # Superorder Ostariophysi
    "Gonorynchiformes",  # Milkfishes
    "Cypriniformes",     # Carps, minnows, loaches
    "Characiformes",     # Characins, tetras, piranhas
    "Siluriformes",      # Catfishes
    "Gymnotiformes",     # Knifefishes, electric eels

    # Superorder Protacanthopterygii
    "Salmoniformes",     # Salmons, trouts, and allies
    "Esociformes",       # Pikes and pickerels
    "Osmeriformes",      # Argentines and smelts

    # Superorder Paracanthopterygii
    "Percopsiformes",    # Trout-perches, pirate perches, cave fishes
    "Gadiformes",        # Cods and allies
    "Lophiiformes",      # Anglerfishes

    "Stomiiformes",
    "Ateleopodiformes",
    "Aulopiformes",
    "Myctophiformes",
    "Lampriformes",
    "Polymixiiformes",
    "Percopsiformes",
    "Gadiformes",
    "Batrachoidiformes",
    "Lophiiformes",
    "Ophidiiformes",
    "Atheriniformes",
    "Cyprinodontiformes",
    "Beloniformes",
    "Mugiliformes",
    "Stephanoberyciformes",
    "Beryciformes",
    "Zeiformes",
    "Gasterosteiformes",
    "Synbranchiformes",
    "Scorpaeniformes",
    "Perciformes",
    "Pleuronectiformes",
    "Tetraodontiformes",
    "Coelacanthiformes",
    "Ceratodontiformes",
    "Lepidosireniformes"
]



def init_spark() -> SparkSession:

    spark = (
        SparkSession.builder
        .appName("GBIF EDA")
        .config("spark.executor.instances", f"{N_EXECUTORS}")
        .config("spark.executor.memory", "75G")
        .config("spark.executor.cores", "12")
        .config("spark.driver.memory", "64G")
        # Additional Tunning
        .config("spark.sql.shuffle.partitions", "1000")
        #.config("spark.sql.files.maxPartitionBytes", "256MB")
        .config("spark.sql.parquet.enableVectorizedReader", "false") 
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )
    
    return spark

def write_lookup_tbl(spark: SparkSession, lookup_tbl: DataFrame, output_path: str, n_max_files_per_batch: int = 100) -> None:
    """
    Takes a lookup table, partitions it into batches that contain at most n_max_files_per_batch data filepaths, and writes the result to Parquet.

    Args:
        spark (SparkSession): The Spark session object.
        lookup_tbl (DataFrame): The lookup table Spark DataFrame containing a "path" column.
        output_path (str): The output path where the partitioned Parquet file will be saved.
        n_max_files_per_batch (int, optional): The maximum number of data files per lookup table batch. Defaults to 100.

    Returns:
        None
    """

    # Group paths into batches of n_max_files_per_batch
    unique_paths = lookup_tbl.select("path").distinct()
    window_spec = Window.orderBy("path")

    unique_paths_with_row = unique_paths.withColumn(
        "row_number", row_number().over(window_spec)
    )

    grouped_paths = unique_paths_with_row.withColumn(
        "group_id", floor((col("row_number") - 1) / n_max_files_per_batch)
    ).drop("row_number")


    # Join grouped paths with the main table
    result_lookup_tbl = lookup_tbl.join(
        grouped_paths, on="path", how="left"
    )

    # Write the result to a partitioned Parquet file
    (
        result_lookup_tbl
        .repartition(1, "group_id")
        .write
        .partitionBy("group_id")
        .mode("overwrite")
        .parquet(output_path)
    )

def create_lookup_tbl(
        spark: SparkSession,
        group: Literal[
            "multi_occ_museum_specimen",
            "citizen_science",
            "questagame_birds",
            "questagame_plants_that_flowers",
            "questagame_butterflies_moths",
            "yale_plants"
        ]
    ) -> DataFrame:
    """
    Creates a lookup table for a pre-specified group by filtering and joining relevant data.

    Args:
        spark (SparkSession): The Spark session object.
        group (Literal): The group for which to create the lookup table. Must be one of:
            "multi_occ_museum_specimen": Museum specimen records with multiple images per occurrence.
            "citizen_science": Citizen science records.
            "questagame_birds": Questagame records with taxonRemarks as Birds.
            "questagame_plants_that_flowers": Questagame records with taxonRemarks as Plants that flower.
            "questagame_butterflies_moths": Questagame records with taxonRemarks as Insects - Butterflies and Moths.
            "yale_plants": Yale University Peabody Museum records with taxonRemarks as Animals and Plants: Plants.

    Returns:
        DataFrame: The resulting lookup table as a Spark DataFrame, containing the `uuid` to `path` mapping and other metadata. 
    """
    

    PATH_COLS_OF_INTEREST = "/fs/scratch/PAS2136/TreeOfLife/attributes/2024-05-01/cols_of_interest"
    PATH_SOURCE_ID_WITH_MULTIPLE_UUIDS = "/fs/scratch/PAS2136/TreeOfLife/attributes/2024-05-01/source_id_with_multiple_uuids"
    PATH_LOOKUP_TABLES = "/fs/scratch/PAS2136/TreeOfLife/lookup_tables/2024-05-01/lookup_tables"
    PATH_OCCURRENCE_CITIZEN_SCIENCE = "/fs/scratch/PAS2136/TreeOfLife/attributes/2024-05-01/occurrence_citizen_science"
    
    spark_df = spark.read.parquet(PATH_COLS_OF_INTEREST)
    lookup_tbl = spark.read.parquet(PATH_LOOKUP_TABLES)

    if group == "multi_occ_museum_specimen":
        # Identify source IDs with multiple UUIDs
        # source_id_with_multiple_uuids = (
        #     spark_df.groupBy("source_id")
        #     .agg(countDistinct("uuid").alias("distinct_uuid_count"))
        #     .filter(col("distinct_uuid_count") > 1)
        # )
        source_id_with_multiple_uuids = spark.read.parquet(PATH_SOURCE_ID_WITH_MULTIPLE_UUIDS)

        # Filter museum specimen records and join with source IDs and lookup table
        result_lookup_tbl = (
            spark_df
            .filter(
                col("basisOfRecord").isin(
                    [
                        "PRESERVED_SPECIMEN", "MATERIAL_SAMPLE", "FOSSIL_SPECIMEN",
                        "LIVING_SPECIMEN", "MATERIAL_CITATION"
                    ]
                )
            )
            .join(source_id_with_multiple_uuids, on="source_id", how="inner")
            .select(["uuid", "source_id", "basisOfRecord"] + COLS_TAXONOMIC)
            .join(lookup_tbl, on="uuid", how="inner")
        )
    
    elif group == "museum_specimen_all":

        result_lookup_tbl = (
            spark_df
            .filter(
                col("basisOfRecord").isin(
                    [
                        "PRESERVED_SPECIMEN", "MATERIAL_SAMPLE", "FOSSIL_SPECIMEN",
                        "LIVING_SPECIMEN", "MATERIAL_CITATION"
                    ]
                )
            )
            .select(["uuid", "source_id", "basisOfRecord", "datasetKey"] + COLS_TAXONOMIC)
            .join(lookup_tbl, on="uuid", how="inner")
        )
    
    elif group == "multi_occ_camera_trap":

        source_id_with_multiple_uuids = spark.read.parquet(PATH_SOURCE_ID_WITH_MULTIPLE_UUIDS)
        cond_remarks = (
            col("eventRemarks").isNotNull() &
            (
                col("eventRemarks").contains("camera trap") |
                col("eventRemarks").contains("Trigger Sensitivity")
            )
        )
        publishers_camera_trap = [
            
            #" National Museum of Nature and Science, Japan", # microscopic plankton images
            "Museums Victoria",
            "Ministerio del Ambiente, Agua y Transición Ecológica de Ecuador - MAATE",
            "Miljøstyrelsen / The Danish Environmental Protection Agency",
            "Burgoigee Creek Landcare Group",
            "Lomonosov Moscow State University"
        ]
        cond_publishers = (
            col("basisOfRecord") == "MACHINE_OBSERVATION"
        ) & (
            col("publisher").isNotNull() & col("publisher").isin(publishers_camera_trap)
        )

        cond_camera_trap = (cond_remarks | cond_publishers)

        result_lookup_tbl = (
            spark_df
            .filter(cond_camera_trap)
            .join(source_id_with_multiple_uuids, on="source_id", how="inner")
            .select(["uuid", "source_id", "basisOfRecord"] + COLS_TAXONOMIC)
            .join(lookup_tbl, on="uuid", how="inner")
        )



    elif group == "citizen_science":
        # Filter citizen science records and join with lookup table
        
        # source_id_with_multiple_uuids = spark.read.parquet(PATH_SOURCE_ID_WITH_MULTIPLE_UUIDS)
        # Include all citizen science images metadata
        occurrence_df_citizen_science = spark.read.parquet(PATH_OCCURRENCE_CITIZEN_SCIENCE)

        result_lookup_tbl = (
            occurrence_df_citizen_science
            .select(["uuid", "source_id", "basisOfRecord"] + COLS_TAXONOMIC)
            .join(lookup_tbl, on="uuid", how="inner")
        )
    
    elif group == "citizen_science_fish":

        occurrence_df_citizen_science = spark.read.parquet(PATH_OCCURRENCE_CITIZEN_SCIENCE)

        result_lookup_tbl = (
            occurrence_df_citizen_science
            .select(["uuid", "source_id", "basisOfRecord"] + COLS_TAXONOMIC)
            .filter(
                (col("phylum").isNotNull()) & 
                (col("order").isNotNull()) &
                (col("phylum") == "Chordata") &
                (col("order").isin(fish_orders))
            )
            .join(lookup_tbl, on="uuid", how="inner")
        )
        

    elif group == "questagame_birds":
        # Filter Condition:
        # - Published by Questagame
        # - taxonRemarks as Birds
        # - phylum not labeled as Chordata

        filter_cond = (
            (col("taxonRemarks") == "Birds") &
            (col("phylum") != "Chordata") &
            (col("publisher") == "Questagame")
        )

        result_lookup_tbl = (
            spark_df
            .filter(filter_cond)
            .select(["uuid", "source_id", "publisher", "taxonRemarks", "taxonomicStatus", "issue"] + COLS_TAXONOMIC)
            .join(lookup_tbl, on="uuid", how="inner")
        )
    elif group == "questagame_plants_that_flowers":
        # Filter Condition:
        # - Published by Questagame
        # - taxonRemarks as Plants that flower
        # - kingdom not labeled as Plantae, or incertae sedis (1653)

        filter_cond = (
            (col("taxonRemarks") == "Plants that flower") &
            (~col("kingdom").isin(["Plantae", "incertae sedis"])) &
            (col("publisher") == "Questagame")
        )

        result_lookup_tbl = (
            spark_df
            .filter(filter_cond)
            .select(["uuid", "source_id", "publisher", "taxonRemarks", "taxonomicStatus", "issue"] + COLS_TAXONOMIC)
            .join(lookup_tbl, on="uuid", how="inner")
        )
    elif group == "questagame_butterflies_moths":
        # Filter Condition:
        # - Published by Questagame
        # - taxonRemarks as Insects - Butterflies and Moths
        # - kingdom not labeled as incertae sedis (7088)
        # - phylum not labeled as Arthropoda

        filter_cond = (
            (col("taxonRemarks") == "Insects - Butterflies and Moths") &
            (col("kingdom")!="incertae sedis") &
            (col("phylum")!="Arthropoda") &
            (col("publisher") == "Questagame")
        )

        result_lookup_tbl = (
            spark_df
            .filter(filter_cond)
            .select(["uuid", "source_id", "publisher", "taxonRemarks", "taxonomicStatus", "issue"] + COLS_TAXONOMIC)
            .join(lookup_tbl, on="uuid", how="inner")
        )   
    elif group == "yale_plants":
        # Filter Condition:
        # - Published by Yale University Peabody Museum
        # - taxonRemarks as Animals and Plants: Plants
        # - kingdom not in Plantae or incertae sedis (16)

        filter_cond = (
            (col("taxonRemarks") == "Animals and Plants: Plants") &
            (~col("kingdom").isin(["Plantae", "incertae sedis"])) &
            (col("publisher") == "Yale University Peabody Museum")
        )

        sample_stratey = {
            "sample_col": "kingdom",
            "sample_prop": {
                "Chromista": 0.05,
                "Bacteria": 0.1,
                "Fungi": 0.05,
                "Animalia": 1,
                "Protozoa": 1
            }   
        }

        result_lookup_tbl = (
            spark_df
            .filter(filter_cond)
            .select(["uuid", "source_id", "publisher", "taxonRemarks", "taxonomicStatus", "issue"] + COLS_TAXONOMIC)
            .join(lookup_tbl, on="uuid", how="inner")
            .sampleBy(
                sample_stratey["sample_col"],
                sample_stratey["sample_prop"],
                seed=614
            )
        )
    else:
        raise ValueError(f"Invalid group: {group}")           

    return result_lookup_tbl

    
def main(
        group: Literal[
            "multi_occ_museum_specimen",
            "museum_specimen_all",
            "citizen_science",
            "citizen_science_fish",
            "questagame_birds",
            "questagame_plants_that_flowers",
            "questagame_butterflies_moths",
            "yale_plants"
        ],
        output_path: str,
        n_max_files_per_batch: int = 100
):

    spark = init_spark()

    write_lookup_tbl(
        spark,
        create_lookup_tbl(spark, group),
        output_path,
        n_max_files_per_batch = n_max_files_per_batch
    )

    logging.info(f"Lookup table for group {group} saved to {output_path}")

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create lookup tables for different groups.")
    parser.add_argument("group", type=str, help="The group to create the lookup table for.")
    parser.add_argument("output_path", type=str, help="The output path to save the lookup table.")
    parser.add_argument("--n_max_files_per_batch", type=int, default=100, help="The maximum number of image data files load per batch.")
    args = parser.parse_args()

    main(args.group, args.output_path, args.n_max_files_per_batch)








