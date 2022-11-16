import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from typing import Tuple

def _load_data(
        path: str, test_size: float, target_column: str
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:

    """
    Load a dataset from the given path and split it into train/val/test sets.
    :param path: dataset path
    :param test_size: test set ratio
    :param target_column: dataset target column name
    :return: train/val/test sets
    """

    # read a CSV file and drop columns with missing values
    df = pd.read_csv(path).dropna(axis=1)

    # split the dataframe into train/val/test sets
    assert 0. < test_size < 1.
    test_size = int(test_size * len(df))
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=42)  # the test set should be the same
    df_train, df_val = train_test_split(df_train, test_size=test_size)

    # divide the train/val/test dataframes into feature and target arrays
    y_train, x_train = df_train.pop(target_column).values.reshape(-1, 1), df_train.values
    y_val, x_val = df_val.pop(target_column).values.reshape(-1, 1), df_val.values
    y_test, x_test = df_test.pop(target_column).values.reshape(-1, 1), df_test.values

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def load_california_housing(
        path: str="./data/california_housing.csv",
        test_size: float=0.2
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:

    """
    Load the California Housing dataset.
    :param path: dataset path
    :param test_size: test set ratio
    :return: train/val/test sets of the California Housing dataset
    """

    return _load_data(path, test_size, target_column="median_house_value")
