import sys
import logging
from typing import Iterable
import json, re

from VirtualHavruta.document import ChunkDocument

from langchain_core.documents import Document

def create_logger(name='virtual-havruta'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(sys.stdout)
    formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    return logger

def part_res(input_res, sep=''):
    if isinstance(input_res, list):
        input_res = ' '.join(input_res)
    if sep:
        return input_res.partition(sep)[2].strip()
    return input_res.strip()

def min_max_scaling(data: Iterable, offset: float = 1e-09) -> list:
    """
    Perform min-max scaling on a list or numpy array of numerical data.

    Parameters:
    -----------
        data
            The input data to be scaled.
        offset
            to avoid returning zero for minimum value.

    Returns:
    --------
        The scaled data.
    """
    data = list(data)
    if not data:
        return data
    
    min_val = min(data)
    max_val = max(data)

    if min_val == max_val:
        return [0.5] * len(data)  # All values are the same, return 0.5

    scaled_data = [(x - min_val + offset) / (max_val - min_val) for x in data]

    return scaled_data

def dict_to_yaml_str(input_dict: dict, indent: int = 0) -> str:
    """
    Convert a dictionary to a YAML-like string without using external libraries.

    Parameters:
        input_dict: The dictionary to convert.
        indent: The current indentation level.

    Returns:
    The YAML-like string representation of the input dictionary.
    """
    yaml_str = ""
    for key, value in input_dict.items():
        padding = "  " * indent
        if isinstance(value, dict):
            yaml_str += f"{padding}{key}:\n{dict_to_yaml_str(value, indent + 1)}"
        elif isinstance(value, list):
            yaml_str += f"{padding}{key}:\n"
            for item in value:
                yaml_str += f"{padding}- {item}\n"
        else:
            yaml_str += f"{padding}{key}: {value}\n"
    return yaml_str

def get_node_data(node: "Node") -> dict:
    """Given a node from the graph database, return the data of the node.

    Parameters
    ----------
    node
        from the graph database

    Returns
    -------
        data of the node
    """
    try:
        record = node.data()
    except AttributeError:
        record = node._properties
    else:
        assert len(record) == 1
        record: dict = next(iter(record.values()))
    return record

def convert_node_to_doc(node: "Node", base_url: str= "https://www.sefaria.org/") -> Document:
    """
    Convert a node from the graph database to a Document object.

    Parameters:
        node (Node): The node from the graph database.

    Returns:
        Document: The Document object created from the node.
    """
    node_data: dict = get_node_data(node)
    metadata = {k:v for k, v in node_data.items() if not k.startswith("content")}
    new_reference_part = metadata["url"].replace(base_url, "")
    new_category = metadata["primaryDocCategory"]
    metadata["source"] = f"Reference: {new_reference_part}. Version Title: -, Document Category: {new_category}, URL: {metadata['url']}"

    page_content = dict_to_yaml_str(node_data.get("content")) if isinstance(node_data.get("content"), dict) else node_data.get("content", "")
    return ChunkDocument(
        page_content=page_content,
        metadata=metadata
    )

def convert_vector_db_record_to_doc(record) -> ChunkDocument:
    assert len(record) == 1
    record: dict = next(iter(record.values()))
    page_content = record.pop("text", None)
    return ChunkDocument(
        page_content=dict_to_yaml_str(page_content)
        if isinstance(page_content, dict)
        else page_content,
        metadata=record
    )

def load_selected_keys(file_path: str, selected_keys: list) -> dict:
    """
    Loads specific keys from a JSON file and returns them as a dictionary.

    This function reads a JSON file from the given file path and filters the data 
    to include only the keys specified in the `selected_keys` list. It then returns 
    a dictionary containing these key-value pairs.

    Parameters:
    file_path (str): The path to the JSON file to be loaded.
    selected_keys (list): A list of keys to be extracted from the JSON file.

    Returns:
    dict: A dictionary containing only the key-value pairs corresponding to the 
          specified `selected_keys`. If a key is not found in the JSON data, it is 
          ignored.

    Example:
    If the JSON file contains:
    {
        "name": "John",
        "age": 30,
        "city": "New York"
    }

    And `selected_keys` is ["name", "city"], the function will return:
    {
        "name": "John",
        "city": "New York"
    }
    """
    # Open and load the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Create a new dictionary with only the selected keys
    filtered_data = {key: data[key] for key in selected_keys if key in data}
    
    return filtered_data
    
def find_matched_filters(query: str, metadata_ranges: dict) -> dict:
    """
    Scans the input query to find matches for strings in the provided JSON data. 
    Only the longest, non-overlapping matches are returned for each key.
    
    Parameters:
    query (str): The input string to be searched.
    metadata_ranges (dict): A dictionary where keys map to lists of strings. 
                      The function will check if any strings from these lists are present in the query.

    Returns:
    dict: A dictionary where each key corresponds to those in the input metadata_ranges 
          and the values are lists of matched strings found in the query.
    
    Example:
    metadata_ranges = {
        "key1": ["foo", "bar", "multi word", "multi", ""],
        "key2": ["another phrase", "phrase", "baz", None, "ample"]
    }

    query = "This is an example with multi word and foo inside another phrase."

    The result will be:
    {
        "key1": ["multi word", "foo"],
        "key2": ["another phrase"]
    }
    """  
    # Dictionary to store matches
    matched_filters = {}
    
    # Iterate over the keys and lists in the JSON data
    for key, string_list in metadata_ranges.items():
        matched_strings = []
        # Sort the list by length of the strings in descending order
        sorted_string_list = sorted([s for s in string_list if s and s.strip()], key=len, reverse=True)
        # Track the parts of the query that have already been matched
        matched_query = query.lower()
        
        # Iterate over each string in the list
        for s in sorted_string_list:    
            # Check if the string (one-word or multi-word) is found in the query
            if s and s.strip():
                # Use regex to match whole words
                pattern = r'\b' + re.escape(s.lower()) + r'\b'
                if re.search(pattern, matched_query):
                    matched_strings.append(s)
                    # Replace the matched portion with a placeholder to avoid overlapping matches
                    matched_query = re.sub(pattern, '', matched_query, count=1)

        # If any matches are found, add them to the matches dictionary
        if matched_strings:
            matched_filters[key] = matched_strings
    
    return matched_filters

def construct_db_filter(matched_filters: dict) -> dict:
    """
    Constructs a database filter string based on the given matched_filters.

    Parameters:
    matched_filters (dict): A dictionary containing optional keys 'primaryDocCategory', 
                     'authorNames', etc., whose values are lists of strings.

    Returns:
    dict: A dictionary representing the DB filter string for querying.

    Example:
    If matched_filters is {'authorNames': ['Rashi']}, the function will return:
    {"authorNames": {"$in": ['Rashi']}}
    
    If matched_filters is {'primaryDocCategory': ['String A', 'String B'], 'authorNames': ['String C']}, 
    the function will return:
    {"$or": [{"primaryDocCategory": {"$in": ['String A', 'String B']}}, 
             {"authorNames": {"$in": ['String C']}}]}
    """
    
    # Create filter conditions for each field
    filter_conditions = []
    for k, v in matched_filters.items():
        filter_conditions.append({k: {"$in": v}})
    
    # If there are multiple conditions, use the $or operator
    if len(filter_conditions) > 1:
        return {"$or": filter_conditions}
    elif filter_conditions:
        return filter_conditions[0]  # Return the single condition without $or
    else:
        return {}  # Return an empty dict if no conditions are provided
    
def merge_topics(extraction: str = '', matched_topics: dict = {}) -> str:
    """
    Merges a comma-separated string of topics with an existing dictionary of lists of topics,
    avoiding duplicates.

    Parameters:
    - extraction (str): A string of topics separated by commas (e.g., 'topic1, topic2, topic3').
    - matched_topics (dict): A dictionary with lists of topics that have already been matched.

    Returns:
    - str: A string of combined topics, separated by commas with no duplicates.
    """
    # Split the extraction string into a list of topics
    extraction_lst = extraction.split(', ')

    # Create a list and a set of topics from all lists within the dictionary for quick membership testing
    joint_matched_topics = []
    joint_matched_topics_set = set()

    for topics in matched_topics.values():
        for topic in topics:
            if topic.lower() not in joint_matched_topics_set:
                joint_matched_topics.append(topic)
                joint_matched_topics_set.add(topic.lower())
    
    # Iterate over each topic in the extracted list
    for topic in extraction_lst:
        # Convert topic to lowercase
        lower_topic = topic.lower()
        # Check if the lowercased topic is not already in the set
        if lower_topic not in joint_matched_topics_set:
            # If the topic is new, append it to the matched_topics list
            joint_matched_topics.append(topic)
            # Add the lowercased topic to the set to keep track of seen topics
            joint_matched_topics_set.add(lower_topic)

    # Join the updated matched topics list into a string separated by commas
    return ', '.join(joint_matched_topics)