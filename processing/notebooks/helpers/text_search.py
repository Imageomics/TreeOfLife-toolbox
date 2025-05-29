def flatten_list_to_string(lst, sep=" "):
    """
    Flattens a list into a single string with a given separator.
    Args:
        lst (list): Input list to flatten.
        sep (str): Separator for joining list elements.

    Returns:
        str: Flattened string or None if list is empty.
    """
    if not lst:  # Check for None or empty list
        return None
    return sep.join(map(str, lst))  # Join elements into a string


def flatten_dict(d, parent_key='', sep='.'):
    """
    Recursively flattens a nested dictionary.
    Args:
        d (dict): Input dictionary to flatten.
        parent_key (str): Key from parent dict (for recursion).
        sep (str): Separator for nested keys.

    Returns:
        dict: Flattened dictionary where all keys are at the top level.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Use flatten_list_to_string or safely handle lists
            items.append((new_key, flatten_list_to_string(v) or ""))
        elif v is None:  # Handle None values
            items.append((new_key, ""))
        else:
            items.append((new_key, str(v)))  # Convert all values to string
    return dict(items)


def full_text_search_rdd(rdd, search_terms, exclude_fields=None):
    """
    Perform a full-text search on a Spark RDD for one or more terms, excluding specified fields.
    Args:
        rdd (RDD): The input RDD containing records (as dictionaries).
        search_terms (str or list): A single search term (str) or multiple terms (list of strings).
        exclude_fields (list): List of field names to exclude from the search.

    Returns:
        RDD: Filtered RDD containing records matching any of the search terms.
    """
    # Ensure search_terms is a list for consistent handling
    if isinstance(search_terms, str):
        search_terms = [search_terms]

    # Normalize all search terms to lowercase for case-insensitive search
    search_terms = [term.lower() for term in search_terms]

    # Ensure exclude_fields is a set for faster lookups
    exclude_fields = set(exclude_fields or [])

    def record_matches(record):
        # Flatten the record to search across all fields
        flattened_record = flatten_dict(record)

        # Remove excluded fields from the flattened dictionary
        filtered_record = {
            k: v for k, v in flattened_record.items() if k not in exclude_fields
        }

        # Check if any term matches any field value
        return any(
            term in value.lower() for value in filtered_record.values() for term in search_terms
        )

    # Filter RDD based on matching records
    return rdd.filter(record_matches)


def extract_fields(fields, fields_to_flatten=None):
    """
    Generate a function to extract specific fields from a record, 
    with the option to flatten list fields into a string.

    Args:
        fields (list): List of field names to extract from the record.
        fields_to_flatten (list): List of fields to flatten if they contain lists.

    Returns:
        function: A function that extracts the specified fields from a record.
    """
    
    fields_to_flatten = fields_to_flatten or []
    
    if isinstance(fields, str):
        fields = [fields]

    def extractor(record):
        extracted = {}
        for field in fields:
            if field in fields_to_flatten:
                extracted[field] = flatten_list_to_string(record.get(field, []), sep="|")
            else:
                extracted[field] = record.get(field)
        return extracted

    return extractor

def create_freq_rdd(rdd, key):
    """
    Get all unique values from a list field in an RDD of dictionaries and their counts.
    
    Args:
        rdd (RDD): An RDD of dictionaries.
        key (str): The key whose list values need to be processed and counted.
    
    Returns:
        list: A list of tuples with unique values and their counts.
    """
    # Step 1: Filter out records where the key is missing, None, or an empty list
    filtered_rdd = rdd.filter(lambda x: x.get(key) not in (None, []) and isinstance(x[key], list))
    
    # Step 2: Flatten the lists and map each value to a key-value pair (value, 1)
    value_pairs_rdd = filtered_rdd.flatMap(lambda x: x[key]) \
                                  .filter(lambda value: value is not None) \
                                  .map(lambda value: (value, 1))
    
    # Step 3: Reduce by key to get the count of each unique value
    value_counts_rdd = value_pairs_rdd.reduceByKey(lambda a, b: a + b)
    
    # Step 4: Collect the results
    return value_counts_rdd.collect()
