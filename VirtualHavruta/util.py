import logging
from logging import handlers
from typing import Iterable

def create_logger(f='virtual-havruta.log', name='virtual-havruta', mb=1*1024*1024, bk=0):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    file_handler = logging.handlers.RotatingFileHandler(f, mode='a', maxBytes=mb, backupCount=bk, encoding=None, delay=0)
    formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def part_res(input_res, sep=''):
    if isinstance(input_res, list):
        input_res = ' '.join(input_res)
    if sep:
        return input_res.partition(sep)[2].strip()
    return input_res.strip()

def min_max_scaling(data: Iterable, offset: float = 1e-05) -> list:
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