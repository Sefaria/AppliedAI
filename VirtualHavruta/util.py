import sys
import logging
from typing import Iterable

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