import os
import sys
import numpy as np
import dill
import yaml
from pandas import DataFrame

from us_visa.exception import USvisaException
from us_visa.logger import logging

import mlflow
import mlflow.sklearn


def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns its content as a dictionary.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: The contents of the YAML file.

    Raises:
        USvisaException: If an error occurs while reading the file.
    """
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise USvisaException(e, sys) from e


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Writes content to a YAML file.

    Args:
        file_path (str): Path where the YAML file should be saved.
        content (object): Data to write into the file.
        replace (bool, optional): If True, existing file is replaced. Defaults to False.

    Raises:
        USvisaException: If an error occurs while writing the file.
    """
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            yaml.dump(content, file)

    except Exception as e:
        raise USvisaException(e, sys) from e


def load_object(file_path: str) -> object:
    """
    Loads a serialized object from a file using dill.

    Args:
        file_path (str): Path to the saved object file.

    Returns:
        object: The deserialized object.

    Raises:
        USvisaException: If an error occurs while loading the object.
    """
    logging.info("Entered the load_object method of utils")

    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)

        logging.info("Exited the load_object method of utils")
        return obj

    except Exception as e:
        raise USvisaException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    """
    Saves an object to a file using dill.

    Args:
        file_path (str): Path to save the object.
        obj (object): The object to serialize and save.

    Raises:
        USvisaException: If an error occurs while saving the object.
    """
    logging.info("Entered the save_object method of utils")

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Exited the save_object method of utils")

    except Exception as e:
        raise USvisaException(e, sys) from e


def save_numpy_array_data(file_path: str, array: np.array) -> None:
    """
    Saves a NumPy array to a file.

    Args:
        file_path (str): Path to save the array.
        array (np.array): The NumPy array to save.

    Raises:
        USvisaException: If an error occurs while saving the array.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)

    except Exception as e:
        raise USvisaException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    Loads a NumPy array from a file.

    Args:
        file_path (str): Path to load the array from.

    Returns:
        np.array: The loaded NumPy array.

    Raises:
        USvisaException: If an error occurs while loading the array.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)

    except Exception as e:
        raise USvisaException(e, sys) from e


def drop_columns(df: DataFrame, cols: list) -> DataFrame:
    """
    Drops specified columns from a Pandas DataFrame.

    Args:
        df (DataFrame): The DataFrame from which columns should be removed.
        cols (list): List of column names to drop.

    Returns:
        DataFrame: DataFrame with specified columns removed.

    Raises:
        USvisaException: If an error occurs while dropping columns.
    """
    logging.info("Entered drop_columns method of utils")

    try:
        df = df.drop(columns=cols, axis=1)

        logging.info("Exited the drop_columns method of utils")
        return df

    except Exception as e:
        raise USvisaException(e, sys) from e
    