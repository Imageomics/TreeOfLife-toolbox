# GBIF Exploratory Data Analysis (EDA)

# Files

The GBIF data falls into three different source categories, each of which calls for a different method of processing for multi-image occurrences, as images within these categories have similar formats.
To address the multi-image per occurrence and invalid entries issues, we thus extract images into these three groups for additional processing. The notebooks below document the identification strategy and the preparation process for data extraction. 
- Citizen Science: `Identify_Citizen_Science.ipynb` (Separation: Issue #51, resolution: #53)
- Museum Specimen: `Identify_Museum_Specimen.ipynb` (Separation: Issue #50, resolution: #52)
- Camera Trap: `Identify_Camera_Trap.ipynb` (Issue #47)

To facilitate collaboration and version control, we pair each notebook with a corresponding `{notebook_name}.py` file generated using `Jupytext`.

All notebooks use helper functions from [`notebooks/helpers`](../helpers).
Be sure to set the appropriate `BASE_PATH` value at the top of each notebook.

# Working Environment

## Spark
The notebooks use `pyspark` to perform analytics on OSC (High-Performance Computing Cluster (HPC)). Here are the steps to use `pyspark` interface for Spark through Jupyter notebook on OSC Ondemand. 

- Log on to https://ondemand.osc.edu/ with your OSC credentials. Choose **Jupyter+Spark** app from the **Interactive Apps** option.

- Provide job submission parameters
    ```
    - Cluster: OSC Cardinal
    - 4 nodes (384 cores)
    - 8 worker per node
    - Spark 3.5.1
    ```
- Click **Launch**. Open notebook and select `PySpark` as the notebook kernel. 

- Obtain the active Spark session

    ```
    from pyspark.sql import SparkSession
    spark= SparkSession.getActiveSession()
    ```
- Start new Spark session (Optional)
    ```
    spark.stop()
    spark = (
        SparkSession.builder
        .appName("GBIF EDA")
        # ...
        # Specify configurations
        # ...
        .getOrCreate()
    )
    ```

## MongoDB
In `Identify_Citizen_Science.ipynb` we had to use [GBIF Registry](https://www.gbif.org/article/5FlXBKbirSiq0ascKYiA8q/gbif-infrastructure-registry) to identify citizen science publishers and datasets. GBIF Registry is much smaller than GBIF Occurrence but it consists of unstructured data. We pulled all data instances using the [GBIF Registry API](https://techdocs.gbif.org/en/openapi/v1/registry#/) iteratively and stored them in a mongoDB:
```
{BASE_PATH}/gbif/gbif_mongo
```
If you are working on OSC, then you can set up mongoDB using the installation scripts:
- `scripts/mongo/install_mongodb.sh` 
- `scripts/mongo/install_mongosh.sh`

To run the mongoDB as a background process in bash:
```
mongod --dbpath="BASE_PATH/gbif/gbif_mongo" --fork --logpath="~/YOUR_LOG_PATH.log"
```

# Data

## Image Data
We store images and image metadata in `{BASE_PATH}/TreeOfLife/data/data*.parquet`. 

```
root
 |-- uuid: string (nullable = true)
 |-- source_id: string (nullable = true)
 |-- identifier: string (nullable = true)
 |-- is_license_full: boolean (nullable = true)
 |-- license: string (nullable = true)
 |-- source: string (nullable = true)
 |-- title: string (nullable = true)
 |-- hashsum_original: string (nullable = true)
 |-- hashsum_resized: string (nullable = true)
 |-- original_size: array (nullable = true)
 |    |-- element: long (containsNull = true)
 |-- resized_size: array (nullable = true)
 |    |-- element: long (containsNull = true)
 |-- image: binary (nullable = true)
 |-- server: string (nullable = true)

```


## Darwin Core Archive Download
The occurrence data in GBIF change over time due to updates, corrections, or new records. We download snapshot that preserves the exact dataset version as it was at download time. Darwin Core Archive (DwC-A) Format is a standardized format for sharing biodiversity data. It uses a compressed ZIP file containing
- `occurrence.txt`: contains species occurrence records, which document the information about the species observed, collected, or recorded.
- `multimedia.txt`: contains references to images, audio, video, or other media related to an occurrence.
- `verbatim.txt`: contains the original, unprocessed information about the species as provided by the data publisher

The raw DwC-A occurrence snapshot download is from https://doi.org/10.15468/dl.bfv433. 

We also converted original `.txt` files into `.parquet` format for faster querying and processing performance:
- Occurrence: `{BASE_PATH}/gbif/processed/2024-05-01/occurrence_parquets`
- Verbatim: `{BASE_PATH}/gbif/processed/2024-05-01/verbatim_parquets`

## Columns of Interest
The `cols_of_interest` dataset is sourced from the GBIF `occurrence.txt`. It includes selected columns we considered potentially useful (e.g., taxa info, source info for filtering, etc.) and is joined with a filtered `multimedia.txt` on `source_id`. Additional filtering was applied to remove non-image entries. The data is stored in scratch storage at 

```
{BASE_PATH}/gbif/attributes/cols_of_interest
```
Here's the schema
```
root
 |-- source_id: string (nullable = true)
 |-- uuid: string (nullable = true)
 |-- source: string (nullable = true)
 |-- basisOfRecord: string (nullable = true)
 |-- sex: string (nullable = true)
 |-- lifeStage: string (nullable = true)
 |-- reproductiveCondition: string (nullable = true)
 |-- behavior: string (nullable = true)
 |-- habitat: string (nullable = true)
 |-- fieldNotes: string (nullable = true)
 |-- eventRemarks: string (nullable = true)
 |-- occurrenceRemarks: string (nullable = true)
 |-- organismRemarks: string (nullable = true)
 |-- locationRemarks: string (nullable = true)
 |-- georeferenceRemarks: string (nullable = true)
 |-- identificationRemarks: string (nullable = true)
 |-- taxonRemarks: string (nullable = true)
 |-- taxonomicStatus: string (nullable = true)
 |-- identificationVerificationStatus: string (nullable = true)
 |-- issue: string (nullable = true)
 |-- publisher: string (nullable = true)
 |-- kingdom: string (nullable = true)
 |-- phylum: string (nullable = true)
 |-- class: string (nullable = true)
 |-- order: string (nullable = true)
 |-- family: string (nullable = true)
 |-- genus: string (nullable = true)
 |-- species: string (nullable = true)
 |-- scientificName: string (nullable = true)
 |-- taxonRank: string (nullable = true)
 |-- vernacularName: string (nullable = true)
 |-- previousIdentifications: string (nullable = true)
 |-- verbatimTaxonRank: string (nullable = true)
 |-- verbatimScientificName: string (nullable = true)
 |-- verbatimIdentification: string (nullable = true)
 |-- datasetKey: string (nullable = true)
 |-- datasetName: string (nullable = true)
 |-- individualCount: string (nullable = true)
 |-- organismQuantity: string (nullable = true)
 |-- organismQuantityType: string (nullable = true)
 |-- institutionID: string (nullable = true)
 |-- institutionCode: string (nullable = true)
 |-- collectionID: string (nullable = true)
 |-- collectionCode: string (nullable = true)
 |-- datasetID: string (nullable = true)
 |-- materialSampleID: string (nullable = true)
```

## Lookup Tables

The directory `{BASE_PATH}/gbif/lookup_tables` contains lookup tables that store the 1-to-1 mapping between each image's `uuid` to its file download location in `/TreeOfLife/data`.

```
{BASE_PATH}/gbif/lookup_tables/2024-05-01
├── lookup_multi_images_camera_trap              # Multi-images per occurrence: Camera Trap
├── lookup_multi_images_citizen_science          # Multi-images per occurrence: Citizen Science
├── lookup_multi_images_museum_specimen          # Multi-images per occurrence: Museum Specimen
└── lookup_tables                                # Complete lookup tables for all successfully downloaded GBIF occurrence images
```

## Image Extraction
The subsets of images are extracted to `{BASE_PATH}/gbif/image_lookup` for additional analysis. 

```
{BASE_PATH}/gbif/image_lookup
├── multi_images_camera_trap                     # Multi-images per occurrence: Camera Trap
├── multi_images_citizen_science                 # Multi-images per occurrence: Citizen Science
└── multi_images_museum_specimen                 # Multi-images per occurrence: Museum Specimen
```
