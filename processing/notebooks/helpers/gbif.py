import requests
import pandas as pd
from pyspark.sql import SparkSession
from tqdm import tqdm
import pymongo

def fetch_gbif_count(base_url, params=None):
    """
    Fetch the total count of records from a GBIF API endpoint.

    Args:
        base_url (str): The base API URL.
        params (dict): Optional query parameters.

    Returns:
        int: The total number of records.
    """
    params = params or {}
    params = params.copy()
    params["limit"] = 1  
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json().get('count', 0)
    except requests.RequestException as e:
        print(f"Error fetching count: {e}")
        return 0

def fetch_gbif_chunk(base_url, params=None, offset=0, limit=500):
    """
    Fetch a chunk of records from a GBIF API endpoint.

    Args:
        base_url (str): The base API URL.
        params (dict): Optional query parameters.
        offset (int): The starting record offset.
        limit (int): The number of records to fetch.

    Returns:
        tuple: A tuple containing:
               - records: List of records fetched.
               - log: A dictionary log with status and details.
    """
    params = params or {}
    params.update({"offset": offset, "limit": limit})  # Add pagination to params
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        records = response.json().get('results', [])
        log = {"offset": offset, "status": "success", "record_count": len(records)}
        return records, log
    except requests.RequestException as e:
        print(f"Failed to fetch chunk at offset {offset}: {e}")
        log = {"offset": offset, "status": "failed", "error": str(e)}
        return [], log

def fetch_gbif_iter(base_url, params=None, limit=500):
    """
    Fetch all records iteratively from a GBIF API endpoint in chunks.

    Args:
        base_url (str): The base API URL.
        params (dict): Optional query parameters.
        limit (int): The maximum number of records to fetch per request.

    Returns:
        tuple: A tuple containing:
               - all_records: List of all records.
               - all_logs: List of log dictionaries for each chunk.
    """
    # Fetch total count of records
    total_count = fetch_gbif_count(base_url, params)
    print(f"Total count of records: {total_count}")
    if total_count == 0:
        return [], []

    all_records = []
    all_logs = []

    num_chunks = (total_count + limit - 1) // limit  # Calculate total chunks
    with tqdm(total=num_chunks, desc="Fetching chunks", unit="chunk") as pbar:
        for offset in range(0, total_count, limit):
            records, log = fetch_gbif_chunk(base_url, params, offset, limit)
            all_records.extend(records)
            all_logs.append(log)
            pbar.update(1)  # Update progress bar
    print("All chunks fetched.")
    return all_records, all_logs

def retry_failed_chunks(base_url, failed_logs, params=None, limit=500, retries=3):
    """
    Retry fetching the failed chunks of records from a GBIF API endpoint.

    Args:
        base_url (str): The base API URL.
        failed_logs (list): List of log dictionaries with failed chunks.
        params (dict): Optional query parameters.
        limit (int): The number of records to fetch per request.
        retries (int): Number of retry attempts for each failed chunk.

    Returns:
        tuple: A tuple containing:
               - retry_records: List of successfully fetched records.
               - retry_logs: Updated list of logs for retry attempts.
    """
    retry_records = []
    retry_logs = []

    with tqdm(total=len(failed_logs), desc="Retrying failed chunks", unit="chunk") as pbar:
        for log in failed_logs:
            offset = log["offset"]
            success = False
            for attempt in range(retries):
                records, retry_log = fetch_gbif_chunk(base_url, params, offset, limit)
                if retry_log["status"] == "success":
                    retry_records.extend(records)
                    retry_logs.append(retry_log)
                    success = True
                    break  # Exit retry loop on success
                else:
                    print(f"Retry {attempt + 1} failed for offset {offset}")
            if not success:
                retry_logs.append({"offset": offset, "status": "failed", "error": "Max retries reached"})
            pbar.update(1)
    return retry_records, retry_logs

def insert_records_to_mongo(collection, records, unique_key="key"):
    """
    Insert records into a MongoDB collection, ensuring key uniqueness.

    Args:
        collection (pymongo.collection.Collection): The MongoDB collection to insert records into.
        records (list): List of records to insert.
        unique_key (str): The field in the record to use as the unique `_id`.

    Returns:
        None
    """
    for record in tqdm(records, desc="Inserting records", unit="doc"):
        # Ensure key uniqueness by setting it as the `_id`
        record["_id"] = record.pop(unique_key, None)  # Set 'key' as '_id' and remove 'key' field
        if record["_id"] is None:
            tqdm.write(f"Record missing '{unique_key}' field. Skipping...")
            continue
        
        try:
            collection.insert_one(record)
        except pymongo.errors.DuplicateKeyError:
            tqdm.write(f"Document with key {record['_id']} already exists. Skipping...")

def fetch_publisher_key(publisher_names):
    """
    Fetch publisher keys from the GBIF API based on publisher name(s), 
    returning only exact title matches (case-insensitive).
    
    Args:
        publisher_names (str or list of str): The publisher name(s) to search for.
    
    Returns:
        list: A list of dictionaries containing publisher keys and titles with exact matches.
    """
    if isinstance(publisher_names, str):
        publisher_names = [publisher_names]  # Ensure input is a list

    results = []
    for name in publisher_names:
        url = f"https://api.gbif.org/v1/organization/suggest?q={name}"
        try:
            response = requests.get(url)
            response.raise_for_status()  
            suggestions = response.json()
            
            # Filter for exact title matches (case-insensitive)
            filtered_results = [
                suggestion for suggestion in suggestions 
                if suggestion.get('title', '').strip().lower() == name.strip().lower()
            ]
            results.extend(filtered_results)
        except requests.RequestException as e:
            print(f"An error occurred while fetching publisher key for '{name}': {e}")
    
    return results


def get_vernacular_names(species_key):
    # TODO: Fix this function
    # GBIF Species API endpoint for vernacular names
    vernacular_names_url = f"https://api.gbif.org/v1/species/{species_key}/vernacularNames"

    try:
        # Send GET request to retrieve vernacular names
        response = requests.get(vernacular_names_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        vernacular_names_data = response.json()

        # Return the list of vernacular names
        return vernacular_names_data.get('results', [])

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

def fetch_gbif_occurrence(gbif_id, full=False):
    
    url = f"https://api.gbif.org/v1/occurrence/{gbif_id}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        
        if full:
            return data
        
        # Extract the desired fields into a dictionary
        extracted_fields = {
            "gbifID": data.get("gbifID"),
            "Scientific Name": data.get("scientificName"),
            "Accepted Scientific Name": data.get("acceptedScientificName"),
            "Kingdom": data.get("kingdom"),
            "Phylum": data.get("phylum"),
            "Class": data.get("class"),
            "Order": data.get("order"),
            "Family": data.get("family"),
            "Genus": data.get("genus"),
            "Species": data.get("species"),
            "Generic Name": data.get("genericName"),
            "Specific Epithet": data.get("specificEpithet"),
            "Taxon Rank": data.get("taxonRank"),
            "Taxonomic Status": data.get("taxonomicStatus"),
        }

        return extracted_fields
    else:
        print(f"Error: Unable to fetch data, status code {response.status_code}")
        return None    

def resolve_gbif_taxon_id(taxon_id):
    url = f"https://api.gbif.org/v1/species/{taxon_id}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get("scientificName")
    return None
