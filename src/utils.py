import logging
import os
import pickle
from . import config

def get_logger(name):
    """
    Sets up a logger with a standard formatter.
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler
        fh = logging.FileHandler(os.path.join(config.LOG_DIR, 'app.log'))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

def save_object(obj, filepath):
    """
    Saves a python object securely using pickle.
    """
    logger = get_logger(__name__)
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        logger.info(f"Successfully saved object to {filepath}")
    except Exception as e:
        logger.error(f"Error saving object to {filepath}: {e}")
        raise e

def load_object(filepath):
    """
    Loads a python object from a filepath using pickle.
    """
    logger = get_logger(__name__)
    try:
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        logger.info(f"Successfully loaded object from {filepath}")
        return obj
    except Exception as e:
        logger.error(f"Error loading object from {filepath}: {e}")
        raise e
