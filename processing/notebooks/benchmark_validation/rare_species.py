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
from pathlib import Path
sys.path.append(os.path.abspath(".."))


from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.functions import col, countDistinct, count, desc, input_file_name, regexp_replace, when, lit
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, BooleanType, DoubleType, ArrayType, LongType, BinaryType
import pandas as pd

from pyspark.sql import SparkSession
# Check if there is an active Spark session
spark= SparkSession.getActiveSession()

# %% [markdown]
# # Objective
#
# [Rare Species](https://huggingface.co/datasets/imageomics/rare-species) is a benchmark from v1 that we will continue to use for v2. 
#
# We want to exclude these rare species images from TOL-200M EOL download. This notebook document the mapping process from Rare Species to TOL-200M.

# %% [markdown]
# # Data Description
#
# Data used in this analysis
# - EOL manifest
# - Rare Species benchmark
# - TOL-200M

# %% [markdown]
# ## EOL
#
# The Encyclopedia of Life (EOL) dataset is a comprehensive collection of biological data, aggregating species information, images, and metadata from various sources to support biodiversity research and education.
#
# The `media_manifest.csv` file within EOL serves as an index of the media assets.
#
# The media manifest used for the TOL-200M curation is combined at:

# %%
eol_media_manifest = spark.read.csv(
    f"{BASE_PATH}/gbif/processed/EoL/metadata/media_manifest.csv",
    header = True
)

# %% [markdown]
# Check schema:

# %%
eol_media_manifest.printSchema()

# %% [markdown]
# ## Rare Species
#
# [Rare Species](https://huggingface.co/datasets/imageomics/rare-species) was generated alongside TreeOfLife-10M; data (images and text) were pulled from Encyclopedia of Life (EOL) to generate a dataset consisting of rare species for zero-shot-classification and more refined image classification tasks.
# - `rarespecies-catalog.csv`: contains the following metadata associated with each image in the dataset
# - `licenses.csv`: File with license, source, and copyright holder associated to each image listed in rarespecies-catalog.csv; rarespecies_id is the shared unique identifier to link the two files
#
# There are some overlapping information across these two metadata files. We'll be mostly using `licenses.csv` for our tasks.

# %%
# Pull directly from HF

metadata_rare_species = spark.createDataFrame(
    pd.read_csv("https://huggingface.co/datasets/imageomics/rare-species/resolve/main/metadata.csv?download=true")
)

liscenses_rare_species = spark.createDataFrame(
    pd.read_csv("https://huggingface.co/datasets/imageomics/rare-species/resolve/main/metadata/licenses.csv?download=true")
)

# %% [markdown]
# Check the schema:

# %%
metadata_rare_species.printSchema()

# %%
liscenses_rare_species.printSchema()

# %% [markdown]
# ## TOL-200M
#
# **Data location:** `{BASE_PATH}/TreeOfLife/data/source=eol`
#
# **Error download location:** `{BASE_PATH}/TreeOfLife/logs/errors/source=eol`

# %%
eol_data_schema = StructType([
    StructField("uuid", StringType(), True),
    StructField("source_id", StringType(), True),
    StructField("identifier", StringType(), True),  # Changed from DoubleType to StringType
    StructField("is_license_full", BooleanType(), True),
    StructField("license", StringType(), True),
    StructField("source", StringType(), True),
    StructField("title", StringType(), True),  # Changed from DoubleType to StringType
    StructField("hashsum_original", StringType(), True),
    StructField("hashsum_resized", StringType(), True),
    StructField("original_size", ArrayType(LongType(), True), True),
    StructField("resized_size", ArrayType(LongType(), True), True),
    StructField("image", BinaryType(), True),
    StructField("server", StringType(), True)
])

eol_error_schema = StructType([
    StructField("uuid", StringType(), True),
    StructField("identifier", StringType(), True),
    StructField("retry_count", LongType(), True),
    StructField("error_code", LongType(), True),
    StructField("error_msg", StringType(), True)
])

data_eol = spark.read.schema(eol_data_schema).parquet(f"{BASE_PATH}/TreeOfLife/data/source=eol")
error_eol = spark.read.schema(eol_error_schema).parquet(f"{BASE_PATH}/TreeOfLife/logs/errors/source=eol")

# %%
data_eol.printSchema()

# %% [markdown]
# - `source_id` is curated using **EOL content ID**
# - `identifier` is curated from **EOL Full-Size Copy URL** and it's used for TOL-200M downloads

# %%
error_eol.printSchema()

# %% [markdown]
# # Mapping
#
# We attempt to use **EOL Full-Size Copy URL** and **MD5 Hashsum** to create a mapping between Rare Species images and the **successfully downloaded** TOL-200M images.

# %% [markdown]
# Check attribute uniqueness

# %%
n_images = liscenses_rare_species.count()
n_unique_urls = liscenses_rare_species.select("eol_full_size_copy_url").distinct().count()
n_unique_md5s = liscenses_rare_species.select("md5").distinct().count()

print(f"Rare Species total images: {n_images}")
print(f"Rare Species unique content URLs: {n_unique_urls}")
print(f"Rare Species unique MD5s: {n_unique_md5s}")

# %%
n_images = data_eol.count()
n_unique_urls = data_eol.select("identifier").distinct().count()
n_unique_md5s = data_eol.select("hashsum_original").distinct().count()

print(f"TOL EOL total images: {n_images}")
print(f"TOL EOL unique content URLs: {n_unique_urls}")
print(f"TOL EOL unique MD5s: {n_unique_md5s}")

# %%
df = liscenses_rare_species

df = (
    df.join(
        data_eol.select("hashsum_original"),
        (data_eol.hashsum_original == df.md5),
        how="left"
    ).withColumn(
        "md5_matched", when(col("hashsum_original").isNotNull(), lit(True)).otherwise(lit(False))
    ).drop("hashsum_original")
)

df = (
    df.join(
        data_eol.select("identifier"),
        (data_eol.identifier == df.eol_full_size_copy_url),
        how="left"
    ).withColumn(
        "url_matched", when(col("identifier").isNotNull(), lit(True)).otherwise(lit(False))
    ).drop("identifier")
)

# %%
df.groupBy("md5_matched", "url_matched").count().show(truncate=False)

# %% [markdown]
# **There 487 images from Rare Species benchmark dataset that can't be matched to the TOL-200M successful image downloads using MD5.** 
#
# **These images can't be matched using content URL used for download either.**
#
#
#
# The error log tables keep track of the information of the unsuccessful download attempt for the EOL images. Let's check the unsuccessful download logs to see if we could find these unmatched images. Since there are no MD5 available, we can only use the content URL as the matching key.

# %%
unmatched_by_md5 = df.filter(~col("md5_matched"))

# %%
unmatched_by_md5.printSchema()

# %%
unmatched_by_md5 = (
    unmatched_by_md5.join(
        error_eol.select("identifier"),
        (error_eol.identifier == unmatched_by_md5.eol_full_size_copy_url),
        how="left"
    ).withColumn(
        "error_url_matched", when(col("identifier").isNotNull(), lit(True)).otherwise(lit(False))
    ).drop("identifier")
)

unmatched_by_md5.cache()

# %%
unmatched_by_md5.filter(col("error_url_matched")).count()

# %% [markdown]
# **There are 114 MD5-unmatched Rare Species images founded in the error logs.** The downloader failed to download these images during the TOL-200M curation process. 

# %%
unmatched_by_all = unmatched_by_md5.filter(~col("error_url_matched"))
unmatched_by_all.count()

# %% [markdown]
# The remaining 373 Rare Species images cannot be matched using MD5 or content URL and are not listed in the error log files.
#
# **Therefore, based on MD5 hashes and EOL content URLs, there is NO evidence that these images exist in the TOL-200M dataset or were included in the TOL-200M download process.**

# %% [markdown]
# # Review Unmatched Images

# %% [markdown]
# We'd like to check if these unmatched Rare Species images are included in the `media_manifest.csv`. And also investigate whether they could be successfully downloaded. 

# %% [markdown]
# ## URL Validation

# %%
unmatched_by_all.select("eol_full_size_copy_url").show(10, truncate=False)

# %% [markdown]
# We manually checked the URLs, and the majority of them appear to be valid.

# %% [markdown]
# ## Presence in EOL Media Manifest

# %%
eol_media_manifest.printSchema()

# %%
unmatched_by_all.printSchema()

# %%
eol_media_manifest = eol_media_manifest.withColumnsRenamed(
    {
        "EOL Full-Size Copy URL": "manifest_eol_full_size_copy_url",
        "Medium Source URL": "manifest_medium_source_url",
        "EOL content ID": "manifest_eol_content_id",
        "EOL page ID": "manifest_eol_page_id"
    }
)
unmatched_by_all = (
    unmatched_by_all
    # Check matching by EOL full size copy URL
    .join(
        eol_media_manifest.select("manifest_eol_full_size_copy_url"),
        eol_media_manifest.manifest_eol_full_size_copy_url == unmatched_by_all.eol_full_size_copy_url,
        how = "left"
    )
    .withColumn(
        "manifest_content_url_matched",
        when(col("manifest_eol_full_size_copy_url").isNotNull(), lit(True)).otherwise(lit(False))
    )
    .drop("manifest_eol_full_size_copy_url")
    
    # Check matching by Medium Source URL
    .join(
        eol_media_manifest.select("manifest_medium_source_url"),
        eol_media_manifest.manifest_medium_source_url == unmatched_by_all.medium_source_url,
        how = "left"
    )
    .withColumn(
        "manifest_medium_source_url_matched",
        when(col("manifest_medium_source_url").isNotNull(), lit(True)).otherwise(lit(False))
    )
    .drop("manifest_medium_source_url")

    # Check matching by eol_content_id & eol_page_id
    .join(
        eol_media_manifest.select("manifest_eol_content_id", "manifest_eol_page_id"),
        (eol_media_manifest.manifest_eol_content_id == unmatched_by_all.eol_content_id) &
        (eol_media_manifest.manifest_eol_page_id == unmatched_by_all.eol_page_id),
        how = "left"
    )
    .withColumn(
        "manifest_content_id_matched",
        when(
            (col("manifest_eol_content_id").isNotNull()) & (col("manifest_eol_page_id").isNotNull()),
             lit(True)
        ).otherwise(lit(False))
    )
    .drop("manifest_eol_content_id", "manifest_eol_page_id")
)

# %%
(
    unmatched_by_all
    .groupBy(
        "manifest_content_url_matched",
        "manifest_medium_source_url_matched",
        "manifest_content_id_matched"
    )
    .count()
).show(truncate=False)

# %% [markdown]
# Among all of the 373 unmatched images
# - 314 can't be found in the EOL manifest.
# - 59 could be matched using content URL or medium source URL

# %% [markdown]
# # Create Lookup Table

# %%
data_eol_rare_species = (
    data_eol
    .withColumn(
        "path", 
        regexp_replace(input_file_name(), "^file://", "")
    )
    .join(
        liscenses_rare_species.select("md5").withColumnRenamed("md5", "hashsum_original"),
        on = "hashsum_original",
        how = "inner"
    )
)

# %%
data_eol_rare_species.count()

# %%
lookup_tbl_rare_species = (
    data_eol_rare_species
    .select("uuid", "path")
)

# %%
lookup_tbl_rare_species.printSchema()

# %%
lookup_tbl_rare_species.show(5, truncate=False)

# %%
lookup_tbl_rare_species = lookup_tbl_rare_species.coalesce(1)

# %%
lookup_tbl_rare_species.write.mode("overwrite").parquet(f"{BASE_PATH}/TreeOfLife/lookup_tables/2024-05-01/eol/lookup_rare_species")

# %%
data_eol.count()
