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
sys.path.append(os.path.abspath(".."))

from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.functions import col, countDistinct, count, desc, broadcast, lower, sum, row_number, floor
from pyspark.sql.window import Window

from helpers.variables import COLS_TAXONOMIC
from helpers.data_analysis import init_spark, create_freq, view_freq, check_sparsity
from helpers.gbif import fetch_gbif_chunk, fetch_gbif_iter, retry_failed_chunks, insert_records_to_mongo, fetch_publisher_key
#from helpers.text_search import flatten_dict, full_text_search_rdd, flatten_list_to_string, extract_fields

import pandas as pd

# %%
# bash:
# mongod --dbpath="${BASE_PATH}/gbif/gbif_mongo" --fork --logpath="${BASE_PATH}/mongo_logs/gbif_mongo.log"
# Check process status
# ps aux | grep mongod

from pymongo import MongoClient

# Connect to the MongoDB server
client = MongoClient("mongodb://localhost:27017/")

db = client["gbif"] 
collection_registry = db["registry"]  
collection_grscicoll_collection = db["grscicoll_collection"]  
collection_grscicoll_institution = db["grscicoll_institution"] 
collection_organization = db["organization"] 

# %% [markdown]
# # Download GRSciColl Collection

# %%
base_url = "https://api.gbif.org/v1/grscicoll/collection"
all_records_collection, all_logs_collection = fetch_gbif_iter(base_url, params=None, limit=500)

# %%
insert_records_to_mongo(
    collection_grscicoll_collection,
    all_records_collection,
    unique_key = "key"
)

# %%
base_url = "https://api.gbif.org/v1/grscicoll/institution"
all_records_institution, all_logs_institution = fetch_gbif_iter(base_url, params=None, limit=500)

# %%
insert_records_to_mongo(
    collection_grscicoll_institution,
    all_records_institution,
    unique_key = "key"
)

# %% [markdown]
# # Download organization

# %%
base_url = "https://api.gbif.org/v1/organization"
all_records_organization, all_logs_organization = fetch_gbif_iter(base_url, params=None, limit=500)

# %%
insert_records_to_mongo(
    collection_organization,
    all_records_organization,
    unique_key = "key"
)

# %% [markdown]
# # Download occurrence registery

# %%
# %%time
base_url = "https://api.gbif.org/v1/dataset"
all_records, all_logs = fetch_gbif_iter(base_url, params = None, limit=500)

# %% [markdown]
# MongoDB uses the `_id` field as a unique identifier for each document. 

# %%
collection.count_documents({})

# %% [markdown]
# # Approach Summary

# %% [markdown]
# Marie Grosjean is the Data Adminstrator from GBIF. She developed an [NLP approach](https://data-blog.gbif.org/post/gbif-citizen-science/) to identify and automatically label datasets as citizen science (CS) using the metadata available via the [GBIF Dataset API](https://techdocs.gbif.org/en/openapi/v1/registry#/). Marie tagged these datasets in the API using a `machineTag` and there are [638 identified dataset matches](http://api.gbif.org/v1/dataset?machineTagNamespace=citizenScience.gbif.org).
#
# We've noticed that the NLP model has failed to capture many citizen science datasets and also mislabeled some non-CS dataset, such as [AntWeb](https://www.gbif.org/dataset/13b70480-bd69-11dd-b15f-b8a03c50a862) ([JSON](https://api.gbif.org/v1/dataset/13b70480-bd69-11dd-b15f-b8a03c50a862)).
#
# We developed an rule-based full-text search approach to increase the precision and coverage:
#
# - **Scope of search:** 
#     - Specific multilingual (English, Spanish, Portuguese, French) keywords (singular & plural) for "citizen" and "citizen science"
#     - Known and verified CS publishers such as iNaturalist, Observation.org, etc.
#     - Known and verified CS datasets
# - **Preprocessing**
#     - Remove all machineTags to prevent introducing results from Marie's NLP model's prior prediction
#     - No additional text cleaning steps applied
# - **Pros & Cons**
#     - Pros: Fast, scalable, reproducible, deterministic
#     - Cons: May introduce recall by missing datasets that do not contain the predefined search terms. May also introduce non-CS datasets that contains seach terms in the metadata.

# %%
# %%time
base_url = "https://api.gbif.org/v1/dataset?machineTagNamespace=citizenScience.gbif.org"
all_records, all_logs = fetch_gbif_iter(base_url, params = None, limit=500)

# %%
key_values = [record['key'] for record in all_records if 'key' in record]
matching_df_nlp = spark.createDataFrame([(value,) for value in key_values], ["datasetKey"])

# %% [markdown]
# # Spark full-text search

# %%
from pyspark.sql import SparkSession

# Check if there is an active Spark session
spark= SparkSession.getActiveSession()

# %%
spark.stop()

# %%
spark = (SparkSession.builder
             .appName("GBIF EDA")
             .config("spark.executor.instances", "64")
             .config("spark.executor.memory", "75G")
             .config("spark.executor.cores", "12")
             .config("spark.sql.parquet.enableVectorizedReader", "false") 
             .getOrCreate())

# %%
spark.sparkContext.addPyFile(os.path.abspath("../helpers/text_search.py"))

from text_search import flatten_dict, full_text_search_rdd, flatten_list_to_string, extract_fields

# %%
publisher_citizen_sci = [
    "iNaturalist.org",
    "Observation.org",
    "naturgucker.de",
    "Questagame",
    "Pl@ntNet",
    "NatureMapr",
    "Citizen Science - ALA Website",
    "BioCollect",
    "Tweed Koala Sightings",
    "Koala Action Group",
    "TilapiaMap",
    "Blauwtipje.nl",
    "Great Koala Count 2",
    #"myFOSSIL eMuseum",
    "Superb Parrot Monitoring project",
    "SLU Artdatabanken",
    "Sibecocenter LLC"

]

publisher_citizen_sci_df = pd.DataFrame(fetch_publisher_key(publisher_citizen_sci))

# %%
datasetKey_citizen_sci = [
    "84a649ce-ff81-420d-9c41-aa1de59e3766", # Citizen Science - ALA Website
    "cca13f2c-0d2c-4c2f-93b9-4446c0cc1629"  # BugGuide published by United States Geological Survey
]

# %%
search_terms = [
    # 'citizen' in different languages, in singular and plural
    "citizen", "citizens",
    # ciencia ciencias
    "ciudadana", "ciudadano", # feminine masculine
    "ciudadanas", "ciudadanos",
    "cidadã", "cidadãs",
    "citoyenne", "citoyennes",
    
    #"inaturalist", "observation.org",
    
    # 'citizen science' in different languages, in singular and plural
    "citizen science", "citizen-science",
    "ciencia ciudadana", # Spanish
    "ciência cidadã",    # Portuguese
    "science citoyenne"  # French
]

search_terms.extend(list(pd.DataFrame(publisher_citizen_sci_df)["key"])) # keys of the known citizen science publishers
search_terms.extend(datasetKey_citizen_sci)                              # keys of citizen science datasets

# %%
# %%time
rdd = spark.sparkContext.parallelize(
    list(collection_registry.find({}))
    # list(collection.find({}, {"machineTags": 0})) # Exclude the machineTag from the downstream full text search
)

rdd_filtered = rdd.map(lambda record: {k: v for k, v in record.items() if k != "machineTags"})

# %%
rdd = None

# %% [markdown]
# **Without `machineTags`**

# %%
# %%time
term_counts = []
for term in search_terms:
    matching_rdd = full_text_search_rdd(rdd_filtered, term)
    count = matching_rdd.count() 
    print(f"Term: {term}, Count: {count}")
    term_counts.append({"Term": term, "Count": count})  

# %%
# Perform search all terms
matching_rdd = full_text_search_rdd(rdd_filtered, search_terms)

match_count = matching_rdd.count()
#results = matching_rdd.collect()
match_count

# %%
# %%time
fields_to_extract = ["_id", "publishingOrganizationKey"]
extract_fields_func = extract_fields(fields_to_extract)

matching_df = spark.createDataFrame(matching_rdd.map(extract_fields_func)).distinct()
matching_df.cache()

matching_df.count()

# %%
# Free memory
rdd_filtered = None 
matching_rdd = None

# %%
from pyspark.sql.types import StructField, StringType, StructType
rdd_grscicoll_institution_all = spark.sparkContext.parallelize(
    list(collection_grscicoll_institution.find({}))
)

fields_to_extract = [
    "_id", 
    "name",
    "code",
    "types",
    "institutionalGovernances",      # Instutional governance of a GrSciColl institution
    "disciplines",                   # Discipline of a GrSciColl institution. Accepts multiple values, for example
    "masterSource"                   # DATASET, ORGANIZATION
]

extract_fields_func = extract_fields(fields_to_extract, fields_to_flatten = ["types", "institutionalGovernances", "disciplines"])

schema = StructType([
    StructField("_id", StringType(), True),
    StructField("name", StringType(), True),
    StructField("code", StringType(), True),
    StructField("types", StringType(), True),
    StructField("institutionalGovernances", StringType(), True),
    StructField("disciplines", StringType(), True),  # Flattened to String
    StructField("masterSource", StringType(), True)
])

grscicoll_institution_df = spark.createDataFrame(
    rdd_grscicoll_institution_all.map(extract_fields_func),
    schema
)

grscicoll_institution_type_list = rdd_grscicoll_institution_all.map(extract_fields(["types"]))

# %%
grscicoll_institution_df.printSchema()

# %%
check_sparsity(grscicoll_institution_df).show(truncate=False)

# %%
create_freq_rdd(grscicoll_institution_type_list, key="types")

# %%
rdd_grscicoll_collection_all = spark.sparkContext.parallelize(
    list(collection_grscicoll_collection.find({}))
)

fields_to_extract = [
    "_id", 
    "name",
    "code",
    "contentTypes",
    "preservationTypes",
    "institutionKey",      
    "institutionName",                 
    "institutionCode",
    "occurrenceCount"
]

extract_fields_func = extract_fields(fields_to_extract, fields_to_flatten = ["contentTypes", "preservationTypes"])

grscicoll_collection_df = spark.createDataFrame(
    rdd_grscicoll_collection_all.map(extract_fields_func)
)

grscicoll_collection_contentTypes_list = rdd_grscicoll_institution_all.map(extract_fields(["contentTypes"]))
grscicoll_collection_preservationTypes_list = rdd_grscicoll_institution_all.map(extract_fields(["preservationTypes"]))
rdd_grscicoll_collection_all = None

# %%
grscicoll_collection_df.printSchema()

# %%
check_sparsity(grscicoll_collection_df).show(truncate=False)

# %% [markdown]
# # Find match in occurrence

# %%
spark_df = spark.read.parquet(f"{BASE_PATH}/gbif/attributes/cols_of_interest")

# %%
filtered_df = (
    spark_df
    .join(
        broadcast(matching_df), 
        matching_df._id == spark_df.datasetKey,
        how = "inner"
    )
)

# %% [markdown]
# ## Match count stats

# %%
filtered_df.select("datasetKey").distinct().count()

# %%
filtered_df.select("publisher").distinct().count()

# %%
filtered_df.select("source_id").distinct().count()

# %%
spark_df.groupBy("basisOfRecord").count().orderBy(desc("count")).show()

# %%
filtered_df.groupBy("basisOfRecord").count().orderBy(desc("count")).show()

# %% [markdown]
# ## Source ID multiple images subset

# %% [markdown]
# Check on subset source ID with multiple images

# %%
col_list = ["source_id", "publisher", "basisOfRecord", "datasetKey"]
source_id_with_multiple_uuids = (
    spark_df.groupBy(*col_list)
    .agg(
        F.countDistinct("uuid").alias("distinct_uuid_count"),
    )
    .filter(col("distinct_uuid_count") > 1)
)
#source_id_with_multiple_uuids.cache()
#source_id_with_multiple_uuids.count()

# %%
source_id_with_multiple_uuids.agg(sum("distinct_uuid_count").alias("sum_value")).show(truncate=False)

# %%
filtered_df = (
    source_id_with_multiple_uuids
    .join(
        broadcast(matching_df), 
        matching_df._id == source_id_with_multiple_uuids.datasetKey,
        how = "inner"
    )
)

# %%
filtered_df.select("source_id").distinct().count()

# %%
view_freq(filtered_df, "basisOfRecord", truncate=False)

# %%
(
    spark_df
    .join(
        source_id_with_multiple_uuids.filter(cond_camera_trap),
        on = "source_id",
        how = "inner"
    )
    .count()
)

# %%
source_id_with_multiple_uuids.groupBy("basisOfRecord").count().orderBy(desc("count")).show()

# %%
filtered_df.groupBy("basisOfRecord").count().orderBy(desc("count")).show()

# %% [markdown]
# ### HUMAN_OBSERVATION

# %% [markdown]
# Most of the `HUMAN_OBSERVATION` are identified as citizen science

# %%
human_observation_unmatched_df = (
    spark_df
    .filter(col("basisOfRecord")=="HUMAN_OBSERVATION")
    .join(
        broadcast(matching_df), 
        matching_df._id == spark_df.datasetKey,
        how = "left_anti"
    )
)

# %%
view_freq(
    human_observation_unmatched_df,
    ["publisher", "datasetKey"],
    30,
    truncate = False
)
create_freq(human_observation_unmatched_df, ["publisher", "datasetKey"]).count()

# %% [markdown]
# ### MATERIAL_SAMPLE

# %%
view_freq(
    filtered_df.filter(col("basisOfRecord").isin(["MATERIAL_SAMPLE"])),
    ["publisher", "datasetKey"],
    truncate = False
)

# %% [markdown]
# ## Exclude Museum Specimens & GRSciColl

# %%
basisOfRecord_specimen_list=["PRESERVED_SPECIMEN", "FOSSIL_SPECIMEN", "MATERIAL_SAMPLE", "LIVING_SPECIMEN", "MATERIAL_CITATION"]

# %% [markdown]
# ### Find GRSciColl Dataset Collection Match

# %%
dataset_collection_match_df = (
    filtered_df
    .join(
        broadcast(grscicoll_collection_df),
        filtered_df.collectionCode == grscicoll_collection_df.code,
        how="inner"
    )
    .filter(
        col("basisOfRecord").isin(basisOfRecord_specimen_list)
    )
    .groupBy(["basisOfRecord", "publisher", "datasetKey"])
    .count()
    .orderBy(desc("count"))
    .distinct()
)
dataset_collection_match_df.show(truncate=False)

# %% [markdown]
# ### Find GRSciColl Dataset Institute Match

# %%
dataset_institution_match_df = (
    filtered_df
    .join(
        broadcast(grscicoll_institution_df),
        filtered_df.institutionCode == grscicoll_institution_df.code,
        how="inner"
    )
    .filter(
        col("basisOfRecord").isin(basisOfRecord_specimen_list)
    )
    .groupBy(["basisOfRecord", "publisher", "datasetKey"])
    .count()
    .orderBy(desc("count"))
    .distinct()
)
dataset_institution_match_df.show(truncate=False)

# %%
dataset_grscicoll_match_df = dataset_institution_match_df.union(dataset_collection_match_df).distinct().orderBy(desc("count"))
dataset_grscicoll_match_df.show(50, truncate=False)

# %% [markdown]
# ### Find GRSciColl Records Match

# %%
# Collection 
record_collection_match_df = (
    filtered_df
    .join(
        broadcast(grscicoll_collection_df),
        filtered_df.collectionCode == grscicoll_collection_df.code,
        how="inner"
    )
    .filter(
        col("basisOfRecord").isin(basisOfRecord_specimen_list)
    )
    .select("uuid")
    .distinct()
)

# Institution
record_institution_match_df = (
    filtered_df
    .join(
        broadcast(grscicoll_institution_df),
        filtered_df.institutionCode == grscicoll_institution_df.code,
        how="inner"
    )
    .filter(
        col("basisOfRecord").isin(basisOfRecord_specimen_list)
    )
    .select("uuid")
    .distinct()
)

record_specimen_match_df = record_collection_match_df.union(record_institution_match_df).distinct()

# %% [markdown]
# ### Remove Matched Datasets & Records

# %%
filtered_df_x_grscicoll = (
    filtered_df
    .join(
        broadcast(record_specimen_match_df),
        on = "uuid",
        how="left_anti"
    )
    .join(
        broadcast(dataset_grscicoll_match_df.select("datasetKey")),
        on = "datasetKey",
        how = "left_anti"
    )
)

# %%
view_freq(filtered_df_x_grscicoll, "basisOfRecord")

# %%
filtered_df_x_grscicoll.count()

# %%

# %% [markdown]
# # Generate Lookup Table for Multi-Images Subset

# %%
from pyspark.sql import SparkSession

# Check if there is an active Spark session
spark= SparkSession.getActiveSession()

# %%
(
    filtered_df_x_grscicoll
    .repartition(10)
    .write
    .mode("overwrite")
    .parquet(f"{BASE_PATH}/gbif/attributes/occurrence_citizen_science")
)

# %%
filtered_df_x_grscicoll = spark.read.parquet(f"{BASE_PATH}/gbif/attributes/occurrence_citizen_science")
lookup_tbl = spark.read.parquet(f"{BASE_PATH}/gbif/lookup_tables/2024-05-01/lookup_tables")

# %%
lookup_tbl_citizen_science = (
    filtered_df_x_grscicoll
    .join(source_id_with_multiple_uuids.select("source_id"), on="source_id", how="inner")
    .select(["uuid", "source_id", "basisOfRecord", "publisher", "datasetKey", "scientificName", "taxonRank"] + COLS_TAXONOMIC)
    .join(lookup_tbl, on="uuid", how="inner")
)

# %%
spark_df = spark.read.parquet(f"{BASE_PATH}/gbif/attributes/cols_of_interest")

filtered_df = (
    spark_df
    .join(
        filtered_df_x_grscicoll.select("uuid"),
        on = "uuid",
        how = "inner"
    )
)

# %%
from pyspark.sql.functions import col, when, mean, sum as spark_sum

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

result_df.show(truncate=False)

# %% [markdown]
# ## N_MAX_FILES: 100

# %%
N_MAX_FILES = 100

unique_paths = lookup_tbl_citizen_science.select("path").distinct()
window_spec = Window.orderBy("path")
unique_paths_with_row = unique_paths.withColumn(
    "row_number", row_number().over(window_spec)
)

grouped_paths = unique_paths_with_row.withColumn(
    "group_id", floor((col("row_number") - 1) / N_MAX_FILES)
).drop("row_number")

result_lookup_tbl = (
    lookup_tbl_citizen_science
    .join(grouped_paths, on="path", how="left")
    .select(["uuid", "source_id", "basisOfRecord", "publisher", "datasetKey", "scientificName", "taxonRank"] + COLS_TAXONOMIC+ ["path", "group_id"])
    .repartition(1, "group_id")
)

# %%
(
    result_lookup_tbl
    .write.partitionBy("group_id").mode("overwrite")
    .parquet(f"{BASE_PATH}/gbif/lookup_tables/2024-05-01/lookup_multi_images_citizen_science")
)

# %%
result_lookup_tbl.printSchema()

# %% [markdown]
# ## N_MAX_FILES: 300

# %%
N_MAX_FILES = 300

unique_paths = lookup_tbl_citizen_science.select("path").distinct()
window_spec = Window.orderBy("path")
unique_paths_with_row = unique_paths.withColumn(
    "row_number", row_number().over(window_spec)
)

grouped_paths = unique_paths_with_row.withColumn(
    "group_id", floor((col("row_number") - 1) / N_MAX_FILES)
).drop("row_number")

result_lookup_tbl = (
    lookup_tbl_citizen_science
    .join(grouped_paths, on="path", how="left")
    .select(["uuid", "source_id", "basisOfRecord", "publisher", "datasetKey", "scientificName", "taxonRank"] + COLS_TAXONOMIC+ ["path", "group_id"])
    .repartition(1, "group_id")
)

# %%
(
    result_lookup_tbl
    .write.partitionBy("group_id").mode("overwrite")
    .parquet(f"{BASE_PATH}/gbif/lookup_tables/2024-05-01/lookup_multi_images_citizen_science_300")
)


# %%
def process_group(spark, base_input_path, base_output_path, group_id):

    # Construct paths for the current group
    group_input_path = f"{base_input_path}/group_id={group_id}"
    group_output_path = f"{base_output_path}/group_id={group_id}"


    filtered_df = spark.read.parquet(group_input_path)
    unique_paths = [row['path'] for row in filtered_df.select("path").distinct().collect()]
    
    # Read the combined DataFrame from unique paths
    combined_df = spark.read.parquet(*unique_paths).select(["uuid", "original_size", "resized_size", "image"])

    result_df = combined_df.join(broadcast(filtered_df), on="uuid", how="inner")
    result_df = result_df.dropDuplicates(["uuid"]).repartition(100)
    
    # Write the result to the output path
    result_df.write.mode("overwrite").parquet(group_output_path)
    print(f"Processed and saved results for group_id={group_id} to {group_output_path}")


# %%
base_input_path = f"{BASE_PATH}/gbif/lookup_tables/2024-05-01/lookup_multi_images_citizen_science"
base_output_path = f"{BASE_PATH}/gbif/image_lookup/multi_images_citizen_science"

# %%
spark.stop()


# %%
def init_spark() -> SparkSession:
    spark = (SparkSession.builder
             .appName("GBIF EDA")
             .config("spark.executor.instances", "80")
             .config("spark.executor.memory", "75G")
             .config("spark.executor.cores", "12")
             .config("spark.sql.parquet.enableVectorizedReader", "false") 
             .getOrCreate())
    
    return spark

spark = init_spark()

# %%
process_group(spark, base_input_path, base_output_path, "0")

# %% [markdown]
# # Comparison & Summary

# %%
filtered_df_nlp = (
    spark_df
    .join(
        broadcast(matching_df_nlp),
        on = "datasetKey",
        how="inner"
    )
)
(
    create_freq(spark_df, "basisOfRecord").selectExpr("basisOfRecord", "count AS n_occurrence")
    .join(
        create_freq(filtered_df_nlp, "basisOfRecord").selectExpr("basisOfRecord", "count AS n_CS_nlp"),
        on = "basisOfRecord",
        how="left"
    )
    .join(
        create_freq(filtered_df, "basisOfRecord").selectExpr("basisOfRecord", "count AS n_CS_matched"),
        on = "basisOfRecord",
        how="left"
    )
    .join(
        create_freq(filtered_df_x_grscicoll, "basisOfRecord").selectExpr("basisOfRecord", "count AS n_CS_matched_x_grscicoll"),
        on = "basisOfRecord",
        how="left"
    )
).show(truncate=False)

# %%
print(f"Occurrence dataset count: {spark_df.select('datasetKey').distinct().count()}")

print(f"NLP dataset count: {filtered_df_nlp.select('datasetKey').distinct().count()}")

print(f"Full-text search dataset count: {filtered_df.select('datasetKey').distinct().count()}")

print(f"Full-text search dataset count (exclude grscicoll): {filtered_df_x_grscicoll.select('datasetKey').distinct().count()}")

# %%
print(f"Occurrence `source_id` count: {spark_df.select('source_id').distinct().count()}")

print(f"NLP `source_id` count: {filtered_df_nlp.select('source_id').distinct().count()}")

print(f"Full-text search `source_id` count: {filtered_df.select('source_id').distinct().count()}")

print(f"Full-text search `source_id` count (exclude grscicoll): {filtered_df_x_grscicoll.select('source_id').distinct().count()}")

# %% [markdown]
# | **Type**                                | **Dataset Count** | **`source_id` Count** |
# |-----------------------------------------|------------------:|----------------------:|
# | Occurrence                              |              1873 |            169,112,381 |
# | NLP                                     |                63 |            109,356,415 |
# | Full-text search                        |               107 |            110,323,916 |
# | Full-text search (exclude grscicoll)    |                93 |            110,286,547 |
#

# %%
view_freq(
    filtered_df_x_grscicoll.filter(
        (col("basisOfRecord") != "HUMAN_OBSERVATION")
    ),
    ["publisher", "basisOfRecord"],
    truncate = False
)

# %%
filtered_df_x_grscicoll.select("source_id").distinct().count()
