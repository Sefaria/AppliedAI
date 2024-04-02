import logging

def create_logger(f='virtual-havruta.log', name='virtual-havruta', mb=1*1024*1024, bk=0):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    file_handler = logging.handlers.RotatingFileHandler(f, mode='a', maxBytes=mb, backupCount=bk, encoding=None, delay=0)
    formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def part_res(input_res, sep=''):
    if sep:
        return input_res.partition(sep)[2].strip()
    return input_res.strip()
