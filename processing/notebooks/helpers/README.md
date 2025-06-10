# `notebook/helpers`

This directory contains reusable Python modules designed to facilitate exploratory data analysis.

## Directory Contents

### **`__init__.py`**

Initializes the module and exports key functions and variables.

**Exports**:

- Functions: `init_spark`, `create_freq`, `view_freq`, `check_sparsity`, `decode_image`, `show_image_table`, `show_image`, `show_images`, `fetch_gbif_occurrence`, `fetch_grscicoll_institutions`, `resolve_gbif_taxon_id`

- Variables: `COLS_OF_INTEREST`, `COLS_TAXONOMIC`.

---

### **`data_analysis.py`**

Provides tools for data exploration and processing using PySpark.

**Key Functions**:
- `init_spark`: Initializes a SparkSession with optimal configurations for large-scale data processing.
- `create_freq`: Generates a frequency table for specified columns in a DataFrame.
- `view_freq`: Displays the frequency table for quick inspection. Wrapper of `create_freq`.
- `check_sparsity`: Analyzes sparsity (null values) for each column in a DataFrame. Sparsity ranges from 0 to 1: 0 means no null values and 1 means all null.

---

### **`gbif.py`**

Integrates with the GBIF (Global Biodiversity Information Facility) API

**Key Functions**:
- `fetch_gbif_occurrence`: Retrieves occurrence data for a specific `gbif_id`, with options to extract all or just selected fields.
- `resolve_gbif_taxon_id`: Resolve the GBIF taxon ID into its actual taxon label, i.e., retrieves the taxon label (Kingdom, Phylum, ..., species) associated to the GBIF taxon ID.
- `fetch_publisher_key`: Retrieves publisher key from the GBIF API by providing the name, only keep exact matches. The list of known citizen science publishers we use is in `notebook/gbif/Identify_Citizen_Science.ipynb`. During citizen science images identification, we obtain the keys of the pre-specified publishers and identify occurrence associated with those publishers using the keys. 
- `fetch_gbif_chunk`: With a given GBIF API endpoint, offset, and limit, retrieves data in JSON format (e.g., from [GBIF Registry](https://techdocs.gbif.org/en/openapi/v1/registry)).
- `fetch_gbif_iter`: Retrieves all data iteratively calling `fetch_gbif_chunk`
- `retry_failed_chunks`: Retries failed GBIF data fetch attempts.
- `insert_records_to_mongo`: Inserts downloaded GBIF records into a MongoDB collection.
---

### **`image.py`**

Handles binary processing and visualization of the image data.

**Key Functions**:
- `find_image`: Find image data based on `uuid`. Returns a Spark DataFrame that contains the image and the essential metadata. **The collect evaluation could be very slow to run during interactive sessions. For batch inquiries, consider submitting jobs to worker nodes and write result to disk.**
- `decode_image`: Converts raw binary image data to RGB Numpy array format for visualization.
- `decode_image_to_pil`: Converts raw binary image data to PIL image format.
- `save_images_partition`: Saves a batch of binary image data to disks using Spark. The image format to use for saving choose between 'jpeg' or 'png' (default 'jpeg').
- `show_image_table`: Displays an image alongside its metadata in a tabular format. 
- `show_images`: Plots images in a grid layout for quick inspection.
- `show_image_interact`: view images interactively by specifying the index and size within a Jupyter Notebook

---
### **`text_search.py`**
Handles full-text search using Spark

**Key Functions**:
- `full_text_search_rdd`: Performs full-text search on Spark RDDs.
- `flatten_dict`: Flattens nested dictionaries
- `flatten_list_to_string`: Converts a list into a single string representation.
- `extract_fields`: Returns a function that extracts specified fields from records. Use with Spark Map-Reduce

---

### **`variables.py`**

Contains predefined global constants used throughout the modules.

**Key Variables**:
- `COLS_OF_INTEREST`: A list of columns of interest from the GBIF `occurrence` dataset. These columns contain information related to taxonomic labels and other identifying information that we explored.
- `COLS_TAXONOMIC`: Taxonomic columns: kingdom, phylum, ..., species.
---
