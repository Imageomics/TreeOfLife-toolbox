# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: PySpark
#     language: python
#     name: pyspark
# ---

# %% [markdown]
# # Env

# %%
# Set base path
BASE_PATH = ""

# %%
import os
import sys
sys.path.append(os.path.abspath("..")) # TreeOfLife-dev/notebooks
# OSC's interactive session's PySpark Kernel set the current working directory to notebook directory

from helpers.data_analysis import init_spark, create_freq, view_freq, check_sparsity
from helpers.image import show_image_table, show_images, decode_image, find_image, show_image_interact
from helpers.variables import COLS_TAXONOMIC
import matplotlib.pyplot as plt

from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.functions import col, countDistinct, count, desc, broadcast, lower

# %%
from pyspark.sql import SparkSession

# Check if there is an active Spark session
spark= SparkSession.getActiveSession()

# %%
# Columns of Interest
spark_df = spark.read.parquet(f"{BASE_PATH}/gbif/attributes/cols_of_interest")

# TOL-200M Metadata
#spark_df_tol = spark.read.parquet(f"{BASE_PATH}/TreeOfLife/metadata")

# Lookup Tables
# Contain uuid and image file path
# lookup_tbl = spark.read.parquet(f"{BASE_PATH}/TreeOfLife/lookup_tables")

# %%
occ_df = spark.read.parquet(f"{BASE_PATH}/gbif/processed/2024-05-01/occurrence_parquets")

# %%
# bash:
# mongod --dbpath="${BASE_PATH}/TreeOfLife/gbif_mongo" --fork --logpath="${BASE_PATH}/mongo_logs/gbif_mongo.log"
# Check process status
# ps aux | grep mongod

from pymongo import MongoClient

# Step 1: Connect to the MongoDB server
client = MongoClient("mongodb://localhost:27017/")

# %% [markdown]
# # Identify Camera-Trap Data within the Multi-Image Subset
#
# We use `basisOfRecord` and `eventRemarks` to identify camera-trap data entries.
#
# ## Criteria for Camera-Trap Data
#
# 1. Records with `eventRemarks` containing:
#   - `'camera trap'`
#   - `'Trigger Sensitivity'`
# Most of these records also have `basisOfRecord` labeled as `MACHINE_OBSERVATION`.
#     - **Exception**: 31 entries from Sibecocenter LLC are labeled as `HUMAN_OBSERVATION`.
#
# 2. A list of publishers from `MACHINE_OBSERVATION` that don't have `eventRemarks`
#
# ## `MACHINE_OBSERVATION` Data Breakdown
# **Number of `MACHINE_OBSERVATION` Records:**
# - Total: 235,632
#   - `eventRemarks` is null: 98,835
#   - `eventRemarks` contains `'camera trap'` or `'Trigger Sensitivity'`: 136,705
#   - `eventRemarks` contains weather-related information: 92
#
# **Note**:
# - Entries with weather-related `eventRemarks` are confirmed **not** to be camera-trap data.
# - Entries with `'camera trap'` or `'Trigger Sensitivity'` in `eventRemarks` are confirmed as camera-trap data.
#
# ### Records with Null `eventRemarks`
# For records without `eventRemarks`, camera-trap data can be inferred from their images. I've reviewed them by publishers
#
# ---
#
# #### Camera Trap Data Publishers
# - **National Museum of Nature and Science, Japan**
#   - *[Plankton image dataset](https://www.gbif.org/dataset/2b9b1484-431b-4dae-94bb-2259e6c1fdc6) from a cabled observatory system (JEDI System/OCEANS) deployed at coastal area of Oshima island, Tokyo, Japan*
# - **Museums Victoria**
# - **Ministerio del Ambiente, Agua y Transición Ecológica de Ecuador - MAATE**
# - **Miljøstyrelsen / The Danish Environmental Protection Agency**
# - **Burgoigee Creek Landcare Group**
# - **Lomonosov Moscow State University**
#
# ---
#
# #### Non-Camera Trap Data Publishers
# - **Xeno-canto Foundation for Nature Sounds**
#   - *Audio files attached with images, not useful for training ([example](https://www.gbif.org/occurrence/4508370762))*
# - **Florida Museum of Natural History**
#   - *Citizen science, phone, archive ([example](https://www.gbif.org/occurrence/gallery?dataset_key=832b8188-f762-11e1-a439-00145eb45e9))*
# - **University of North Carolina at Chapel Hill Herbarium (NCU)**
#   - *Not camera trap data*
# - **Jardín Botánico de Quito**
#   - *Cell phone camera, not camera trap data*
# - **Negrita Films SAS**
#   - *High-res images, not camera trap data*
# - **University of Alaska Museum of the North**
#   - *Not camera trap data*
# - **University of Colorado Museum of Natural History**
#   - *High-res images, not camera trap data*
#
# ---
#
# #### Unknown Publisher
# - **Área Metropolitana del Valle de Aburrá**: *Records could not be found.*
#
# ---

# %% [markdown]
# Find subset of source ID that contains multiple images

# %%
col_list = ["source_id", "eventRemarks", "publisher", "datasetKey", "basisOfRecord", "taxonomicStatus", "scientificName"] + COLS_TAXONOMIC
source_id_with_multiple_uuids = (
    spark_df.groupBy(*col_list)
    .agg(
        F.countDistinct("uuid").alias("distinct_uuid_count"),
    )
    .filter(col("distinct_uuid_count") > 1)
)
source_id_with_multiple_uuids.cache()

source_id_with_multiple_uuids.printSchema()

# %%
cond_remarks = (
    col("eventRemarks").isNotNull() &
    (
        col("eventRemarks").contains("camera trap") |
        col("eventRemarks").contains("Trigger Sensitivity")
    )
)
publishers_camera_trap = [
    #"National Museum of Nature and Science, Japan", # microscopic plankton images, excluded
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

# %%
assert (
    source_id_with_multiple_uuids.filter(cond_camera_trap).count() + source_id_with_multiple_uuids.filter(~cond_camera_trap).count() == source_id_with_multiple_uuids.count()
), "The conditions do not fully partition the dataset."

# %%
assert (
    spark_df.filter(cond_camera_trap).count() + spark_df.filter(~cond_camera_trap).count() == spark_df.count()
), "The conditions do not fully partition the dataset."

# %%
source_id_with_multiple_uuids.filter(cond_camera_trap).count()

# %%
view_freq(source_id_with_multiple_uuids.filter(cond_camera_trap), "basisOfRecord", truncate=False)

# %%
(
    spark_df
    .join(
        source_id_with_multiple_uuids.filter(cond_camera_trap).select("source_id"),
        on = "source_id",
        how = "inner"
    )
    .select(["uuid", "source_id", "datasetKey", "basisOfRecord", "publisher"])
    .repartition(4)
    .write
    .mode("overwrite")
    .parquet(f"{BASE_PATH}/gbif/attributes/occurrence_camera_trap_multi_images")
)

# %% [markdown]
# # Invesitgate Dataset Metadata

# %%
df_camera_trap = spark.read.parquet(f"{BASE_PATH}/gbif/attributes/occurrence_camera_trap_multi_images")

# %%
data_camera_trap = spark.read.parquet(f"{BASE_PATH}/gbif/image_lookup/multi_images_camera_trap/*")

# %%
data_camera_trap_filtered = data_camera_trap.filter(col("publisher")!="National Museum of Nature and Science, Japan")

# %%
data_camera_trap.printSchema()

# %%
df_camera_trap.printSchema()

# %% [markdown]
# Total images count

# %%
df_camera_trap.count()

# %% [markdown]
# Total occurrence count

# %%
df_camera_trap.select("source_id").distinct().count()

# %%
filtered_df = (
    spark_df
    .join(
        df_camera_trap,
        on = "uuid",
        how = "inner"
    )
)

# %%
# Generate boolean columns and cast to integers for aggregation
spark_df_with_flags = filtered_df.withColumn(
    "is_taxon_higher_rank", 
    when(col("issue").contains("TAXON_MATCH_HIGHERRANK"), 1).otherwise(0)
).withColumn(
    "is_taxon_match_none", 
    when(col("issue").contains("TAXON_MATCH_NONE"), 1).otherwise(0)
).withColumn(
    "is_taxon_match_fuzzy", 
    when(col("issue").contains("TAXON_MATCH_FUZZY"), 1).otherwise(0)
).withColumn(
    "is_any", 
    when((col("issue").contains("TAXON_MATCH_FUZZY")) | (col("issue").contains("TAXON_MATCH_NONE")) | (col("issue").contains("TAXON_MATCH_HIGHERRANK")), 1).otherwise(0)
)

# Calculate sum and mean for each numeric column
result_df = spark_df_with_flags.agg(
    spark_sum("is_taxon_higher_rank").alias("sum_taxon_higher_rank"),
    mean("is_taxon_higher_rank").alias("mean_taxon_higher_rank"),
    spark_sum("is_taxon_match_none").alias("sum_taxon_match_none"),
    mean("is_taxon_match_none").alias("mean_taxon_match_none"),
    spark_sum("is_taxon_match_fuzzy").alias("sum_taxon_match_fuzzy"),
    mean("is_taxon_match_fuzzy").alias("mean_taxon_match_fuzzy"),
    spark_sum("is_any").alias("sum_is_any"),
    mean("is_any").alias("mean_is_any")
)


# %%
result_df.show(truncate=False)

# %% [markdown]
# | Category               | Count  | Percentage  |
# |------------------------|--------|-------------|
# | Taxon Higher Rank      | 8,692  | 0.36%       |
# | Taxon Match None       | 69,279 | 2.85%       |
# | Taxon Match Fuzzy      | 355    | 0.01%       |
# | Contains Any           | 78,326 | 3.22%       |
#

# %% [markdown]
# # Distribution Analysis

# %%
lookup_tbl_camera_trap = spark.read.parquet(f"{BASE_PATH}/gbif/lookup_tables/2024-05-01/lookup_multi_images_camera_trap")
lookup_tbl = spark.read.parquet(f"{BASE_PATH}/gbif/lookup_tables/2024-05-01/lookup_tables")

# %%
lookup_tbl_camera_trap = lookup_tbl_camera_trap.filter(col("publisher") != "National Museum of Nature and Science, Japan")

# %%
COLS_TAXONOMIC_KEY = [x + "Key" for x in COLS_TAXONOMIC]

lookup_tbl_camera_trap = (
    lookup_tbl_camera_trap
    .join(
        spark_df.select(["uuid"] + COLS_TAXONOMIC_KEY),
        on = "uuid",
        how = "inner"
    )
)

spark_df = (
    spark_df
    .join(
        lookup_tbl.select("uuid"),
        on = "uuid",
        how = "inner"
    )
)

# %%
agg_df = (
    lookup_tbl_camera_trap
    .groupBy(["source_id"] + COLS_TAXONOMIC_KEY)
    .count()
)

# %%
agg_df.printSchema()

# %%
threshold = 15

occ_keep_df = agg_df.filter(col("count")<=threshold)
occ_omit_df = agg_df.filter(col("count")>threshold)

# %%
taxon_level = "species"
taxon_level_key = taxon_level + "Key"

taxon_freq = (
    spark_df
    .join(
        occ_omit_df.select(["source_id"]),
        on = "source_id",
        how = "left_anti"
    )
    .select([taxon_level, taxon_level_key])
    .groupBy([taxon_level, taxon_level_key])
    .count().withColumnRenamed("count", "img_count")
    .filter(col(taxon_level).isNotNull())
)

taxon_freq_camera_trap_keep = (
    lookup_tbl_camera_trap
    .join(
        occ_omit_df.select(["source_id"]),
        on = "source_id",
        how = "left_anti"
    )
    .select([taxon_level, taxon_level_key])
    .groupBy([taxon_level, taxon_level_key])
    .count().withColumnRenamed("count", "img_count")
    .filter(col(taxon_level).isNotNull())
)

taxon_freq_camera_trap_omit = (
    lookup_tbl_camera_trap
    .join(
        occ_omit_df.select(["source_id"]),
        on = "source_id",
        how = "inner"
    )
    .select([taxon_level, taxon_level_key])
    .groupBy([taxon_level, taxon_level_key])
    .count().withColumnRenamed("count", "img_count")
    .filter(col(taxon_level).isNotNull())
)

# %%
taxon_freq_camera_trap_omit.printSchema()

# %%
lookup_tbl_camera_trap_omit = (
    lookup_tbl_camera_trap
    .join(
        occ_omit_df.select(["source_id"]),
        on = "source_id",
        how = "inner"
    )
)

n_omit_occ = lookup_tbl_camera_trap_omit.select(["source_id"]).distinct().count()
n_omit_occ_species_null = lookup_tbl_camera_trap_omit.filter(col("species").isNull()).select(["source_id"]).distinct().count()

print(f"Removed occ: {n_omit_occ}")
print(f"Removed occ with no species label: {n_omit_occ_species_null}")

# %%
occ_omit_df.printSchema()

# %%
check_sparsity(occ_omit_df.filter(col("speciesKey").isNull()).select(COLS_TAXONOMIC_KEY)).show(truncate=False)

# %% [markdown]
# If species label is missing, roll-up a level to genus

# %%
taxon_level = "genusKey"
taxon_level_key = "genusKey"

taxon_freq = (
    spark_df
    .join(occ_omit_df.filter((col("speciesKey").isNull())&(col("genusKey").isNotNull())).select(["source_id"]), on="source_id", how="left_anti")
    .select([taxon_level, taxon_level_key])
    .groupBy([taxon_level, taxon_level_key])
    .count().withColumnRenamed("count", "img_count")
    .filter(col(taxon_level).isNotNull())
)

# Camera trap - Kept occurrences
taxon_freq_camera_trap_keep = (
    lookup_tbl_camera_trap
    .join(occ_omit_df.filter((col("speciesKey").isNull())&(col("genusKey").isNotNull())).select(["source_id"]), on="source_id", how="left_anti")
    .select([taxon_level, taxon_level_key])
    .groupBy([taxon_level, taxon_level_key])
    .count().withColumnRenamed("count", "img_count")
    .filter(col(taxon_level).isNotNull())
)

# Camera trap - Omitted occurrences
taxon_freq_camera_trap_omit = (
    lookup_tbl_camera_trap
    .join(occ_omit_df.filter((col("speciesKey").isNull())&(col("genusKey").isNotNull())).select(["source_id"]), on="source_id", how="inner")
    .select([taxon_level, taxon_level_key])
    .groupBy([taxon_level, taxon_level_key])
    .count().withColumnRenamed("count", "img_count")
    .filter(col(taxon_level).isNotNull())
)

# %%
taxon_loss_camera_trap_kept = (
    taxon_freq_camera_trap_omit
    .join(taxon_freq_camera_trap_keep, on=taxon_level_key, how="left_anti")
).count()

taxon_loss_gbif = (
    taxon_freq_camera_trap_omit
    .join(taxon_freq, on=taxon_level_key, how="left_anti")
).count()

print(f"Genus loss in the camera-trap subset: {taxon_loss_camera_trap_kept}")
print(f"Genus loss in the GBIF OCC: {taxon_loss_gbif}")

# %% [markdown]
# With the discarded subset:
# - How many unique `taxon_level` can't found in the kept subset
# - How many unique `taxon_level` can't found in the GBIF occurrence (successfully downloaded)

# %%
n_camera_trap_kept = (
    taxon_freq_camera_trap_omit
    .join(
        taxon_freq_camera_trap_keep,
        on = "speciesKey",
        how = "left_anti"
    )
).count()

n_total = (
    taxon_freq_camera_trap_omit
    .join(
        taxon_freq,
        on = "speciesKey",
        how = "left_anti"
    )
).count()

# %% [markdown]
# Run simulations by changing the threshold

# %%
occ_omit_df.printSchema()

# %%
lookup_tbl_camera_trap = spark.read.parquet(f"{BASE_PATH}/gbif/lookup_tables/2024-05-01/lookup_multi_images_camera_trap")
lookup_tbl = spark.read.parquet(f"{BASE_PATH}/gbif/lookup_tables/2024-05-01/lookup_tables")

COLS_TAXONOMIC_KEY = [x + "Key" for x in COLS_TAXONOMIC]

lookup_tbl_camera_trap = (
    lookup_tbl_camera_trap
    .join(
        spark_df.select(["uuid"] + COLS_TAXONOMIC_KEY),
        on = "uuid",
        how = "inner"
    )
)

spark_df = (
    spark_df
    .join(
        lookup_tbl.select("uuid"),
        on = "uuid",
        how = "inner"
    )
)

def run_simulation(spark_df, lookup_tbl_camera_trap, threshold_values, taxon_level="species"):
    """
    Runs a simulation to assess taxon loss at different occurrence count thresholds.
    
    Parameters:
        spark_df (DataFrame): Original occurrence data with taxon information.
        lookup_tbl_camera_trap (DataFrame): Camera trap lookup table.
        taxon_level (str): Taxonomic level to analyze (default is "species").
    
    Returns:
        DataFrame: Spark DataFrame containing simulation results.
    """
    #threshold_values = list(range(60, 10, -10)) + [15, 12, 10, 8, 5, 2]
    taxon_level_key = taxon_level + "Key"
    simulation_results = []

    agg_df = (
        lookup_tbl_camera_trap
        .groupBy(["source_id"] + COLS_TAXONOMIC_KEY)
        .count()
    )
        
    for threshold in tqdm(threshold_values, desc="Running Simulations"):
        occ_keep_df = agg_df.filter(col("count") <= threshold)
        occ_omit_df = agg_df.filter(col("count") > threshold)

        n_occ_omit = occ_omit_df.select("source_id").distinct().count()
        n_img_omit = (
            lookup_tbl_camera_trap
            .join(occ_omit_df.select(["source_id"]), on="source_id", how="inner")
            .select("uuid").distinct().count()
        )
        
        # Calculate taxon frequencies for kept occurrences
        taxon_freq = (
            spark_df
            .join(occ_omit_df.select(["source_id"]), on="source_id", how="left_anti")
            .select([taxon_level, taxon_level_key])
            .groupBy([taxon_level, taxon_level_key])
            .count().withColumnRenamed("count", "img_count")
            .filter(col(taxon_level).isNotNull())
        )

        # Camera trap - Kept occurrences
        taxon_freq_camera_trap_keep = (
            lookup_tbl_camera_trap
            .join(occ_omit_df.select(["source_id"]), on="source_id", how="left_anti")
            .select([taxon_level, taxon_level_key])
            .groupBy([taxon_level, taxon_level_key])
            .count().withColumnRenamed("count", "img_count")
            .filter(col(taxon_level).isNotNull())
        )

        # Camera trap - Omitted occurrences
        taxon_freq_camera_trap_omit = (
            lookup_tbl_camera_trap
            .join(occ_omit_df.select(["source_id"]), on="source_id", how="inner")
            .select([taxon_level, taxon_level_key])
            .groupBy([taxon_level, taxon_level_key])
            .count().withColumnRenamed("count", "img_count")
            .filter(col(taxon_level).isNotNull())
        )

        taxon_loss_camera_trap_kept = (
            taxon_freq_camera_trap_omit
            .join(taxon_freq_camera_trap_keep, on=taxon_level_key, how="left_anti")
        ).count()

        taxon_loss_gbif = (
            taxon_freq_camera_trap_omit
            .join(taxon_freq, on=taxon_level_key, how="left_anti")
        ).count()

        # Store results
        simulation_results.append((
            threshold, taxon_loss_camera_trap_kept, taxon_loss_gbif,
            n_occ_omit, n_img_omit
        ))
    
    simulation_df = spark.createDataFrame(
        simulation_results, 
        ["threshold", "taxon_loss_camera_trap_kept", "taxon_loss_gbif", "n_occ_omit", "n_img_omit"]
    )
    
    return simulation_df


# %%
# %%time
result_df = run_simulation(
    spark_df, lookup_tbl_camera_trap,
    threshold_values = list(range(60, 10, -10)) + [15, 12, 10, 8, 5, 2],
    taxon_level="species"
)

# %%
result_df.show(truncate=False)

# %% [markdown]
# TODO: Are they `taxon_loss_camera_trap_kept` in lila?
#
# Check species NULL, set threshold to 15+ order

# %%
# %%time
result_df_genus = run_simulation(
    spark_df, lookup_tbl_camera_trap,
    threshold_values = list(range(60, 10, -10)) + [15, 12, 10, 8, 5, 2],
    taxon_level="genus"
)

# %%
result_df_genus.show(truncate=False)

# %%
# %%time
result_df_family = run_simulation(
    spark_df, lookup_tbl_camera_trap,
    threshold_values = list(range(60, 10, -10)) + [15, 12, 10, 8, 5, 2],
    taxon_level="family"
)

# %%
result_df_family.show(truncate=False)

# %%
# %%time
result_df_order = run_simulation(
    spark_df, lookup_tbl_camera_trap,
    threshold_values = list(range(60, 10, -10)) + [15, 12, 10, 8, 5, 2],
    taxon_level="order"
)

# %%
result_df_order.show(truncate=False)

# %%
occ_keep_df = agg_df.filter(col("count") <= 15)
occ_omit_df = agg_df.filter(col("count") > 15)

n_occ_omit = occ_omit_df.select("source_id").distinct().count()
n_img_omit = (
    lookup_tbl_camera_trap
    .join(occ_omit_df.select(["source_id"]), on="source_id", how="inner")
    .select("uuid").distinct().count()
)

# Calculate taxon frequencies for kept occurrences
taxon_freq = (
    spark_df
    .join(occ_omit_df.select(["source_id"]), on="source_id", how="left_anti")
    .select([taxon_level, taxon_level_key])
    .groupBy([taxon_level, taxon_level_key])
    .count().withColumnRenamed("count", "img_count")
    .filter(col(taxon_level).isNotNull())
)

# Camera trap - Kept occurrences
taxon_freq_camera_trap_keep = (
    lookup_tbl_camera_trap
    .join(occ_omit_df.select(["source_id"]), on="source_id", how="left_anti")
    .select([taxon_level, taxon_level_key])
    .groupBy([taxon_level, taxon_level_key])
    .count().withColumnRenamed("count", "img_count")
    .filter(col(taxon_level).isNotNull())
)

# Camera trap - Omitted occurrences
taxon_freq_camera_trap_omit = (
    lookup_tbl_camera_trap
    .join(occ_omit_df.select(["source_id"]), on="source_id", how="inner")
    .select([taxon_level, taxon_level_key])
    .groupBy([taxon_level, taxon_level_key])
    .count().withColumnRenamed("count", "img_count")
    .filter(col(taxon_level).isNotNull())
)

taxon_loss_camera_trap_kept = (
    taxon_freq_camera_trap_omit
    .join(taxon_freq_camera_trap_keep, on=taxon_level_key, how="left_anti")
)

taxon_loss_gbif = (
    taxon_freq_camera_trap_omit
    .join(taxon_freq, on=taxon_level_key, how="left_anti")
)

# %%
occ_omit_species_loss_df = (
    occ_omit_df
    .join(
        taxon_loss_camera_trap_kept.select("speciesKey"),
        on = "speciesKey",
        how = "inner"
    )
    .select("source_id")
    .join(
        lookup_tbl_camera_trap.select(["source_id"] + COLS_TAXONOMIC).distinct(),
        on = "source_id",
        how = "inner"
    )
)

# %%
occ_omit_species_loss_df.cache()

# %%
df_camera_trap = spark.read.parquet(f"{BASE_PATH}/gbif/image_lookup/multi_images_camera_trap")

# %%
df_subset = (
    df_camera_trap
    .join(
        occ_omit_species_loss_df.select("source_id"),
        on = "source_id",
        how = "inner"
    )
)

# %%
df_subset.write.parquet(f"{BASE_PATH}/gbif/image_lookup/multi_images_camera_trap_below_15_species_loss")

# %%
df_subset = spark.read.parquet(f"{BASE_PATH}/gbif/image_lookup/multi_images_camera_trap_below_15_species_loss")

# %%
df_subset.drop("image", "original_size", "resized_size").toPandas().to_csv(
    f"{BASE_PATH}/gbif/image_lookup/multi_images_camera_trap_below_15_species_loss.csv",
    index=False
)

# %%
metadata_all = spark.read.parquet(f"{BASE_PATH}/TreeOfLife/metadata")
metadata_lila = metadata_all.filter(col("source").isin(["lila-bc-multi-label", "lila-bc-single-label"]))

# %%
(
    df_subset
    .select("species").distinct()
    .join(
        metadata_lila.select("species").distinct(),
        on = "species",
        how = "inner"
    )
).count()

# %% [markdown]
# # Filtering

# %%
threshold = 15

occ_keep_df = agg_df.filter(col("count")<=threshold)
occ_omit_df = agg_df.filter(col("count")>threshold)

# %%
lookup_tbl_camera_trap.count()

# %%
lookup_tbl_camera_trap_15 = (
    lookup_tbl_camera_trap
    .join(
        occ_keep_df.select("source_id"),
        on = "source_id",
        how = "inner"
    )
)
lookup_tbl_camera_trap_15.count()

# %%
df_camera_trap = spark.read.parquet(f"{BASE_PATH}/gbif/image_lookup/multi_images_camera_trap")

# %%
df_camera_trap_15 = (
    df_camera_trap
    .join(
        occ_keep_df.select("source_id"),
        on = "source_id",
        how = "inner"
    )
)

# %%
df_camera_trap_15.write.mode("overwrite").parquet(f"{BASE_PATH}/gbif/image_lookup/multi_images_camera_trap_below_15")

# %% [markdown]
#
