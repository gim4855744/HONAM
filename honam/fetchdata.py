"""Fetch datasets."""

import base64
import bz2
import gzip
import os
import shutil

from os.path import join as pjoin, exists as pexists
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from tqdm import tqdm

__all__ = [
    'fetch_ca_housing', 'fetch_insurance', 'fetch_house_prices', 'fetch_bikeshare', 'fetch_year',
    'DATASET_MAP'
]


def _download(url, filename, delete_if_interrupted=True, chunk_size=4096):

    """It saves file from url to filename with a fancy progressbar."""

    try:

        with open(filename, "wb") as f:
            print("Downloading {} > {}".format(url, filename))
            response = requests.get(url, stream=True)
            total_length = response.headers.get('content-length')
            if total_length is None:  # no content length header
                f.write(response.content)
            else:
                total_length = int(total_length)
                with tqdm(total=total_length) as progressbar:
                    for data in response.iter_content(chunk_size=chunk_size):
                        if data:  # filter-out keep-alive chunks
                            f.write(data)
                            progressbar.update(len(data))

    except Exception as e:

        if delete_if_interrupted:
            print("Removing incomplete download {}.".format(filename))
            os.remove(filename)
        raise e
    
    return filename


def _create_onedrive_directdownload(onedrive_link):
    """See https://towardsdatascience.com/how-to-get-onedrive-direct-download-link-ecb52a62fee4."""
    data_bytes64 = base64.b64encode(bytes(onedrive_link, 'utf-8'))
    data_bytes64_String = data_bytes64.decode('utf-8').replace('/','_').replace('+','-').rstrip("=")
    resultUrl = f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64_String}/root/content"
    return resultUrl


def _download_file_from_onedrive(onedrive_link, destination):
    """Download file from onedrive."""
    _download(_create_onedrive_directdownload(onedrive_link), destination)


def _download_file_from_google_drive(id, destination):

    """See https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url."""

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


def fetch_ca_housing(
    path: str = './data/',
    fold: int = 0
) -> dict:

    """Download California Housing dataset.

    Args:
        path: where the data should be stored.
        fold: data fold number.

    Returns:
        data: it contains the following keys
            - 'x_train', 'y_train', 'x_test', 'y_test': train/test sets. x is a pandas dataframe and y is a numpy array.
            - 'categorical_features', 'continuous_features': which features are categorical/continuous.
            - 'task': which target task type. either 'reg', 'bincls', or 'multicls'.
            - 'n_features': the number of input features.
            - 'n_outputs': the number of outputs.
    """

    assert 0 <= fold <= 4, 'invalid fold number.'

    data_path = pjoin(path, 'cahousing', 'california_housing_prices.csv')
    if not pexists(data_path):
        os.makedirs(pjoin(path, 'cahousing'), exist_ok=True)
        file_id = "1L-mAY0PBJZ7SFEDaAUYfWa4ckJetD-hc"
        _download_file_from_google_drive(file_id, data_path)

    categorical_features = ['ocean_proximity']
    continuous_features = [
        'longitude', 'latitude', 'housing_median_age', 'total_rooms', 'population', 'households',
        'median_income'
    ]
    targets = ['median_house_value']
    usecols = categorical_features + continuous_features + targets

    df = pd.read_csv(data_path, usecols=usecols)
    x = df.drop(targets, axis=1)
    y = df.get(targets).values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return {
        'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test,
        'categorical_features': categorical_features, 'continuous_features': continuous_features,
        'task': 'regression', 'n_features': len(categorical_features + continuous_features), 'n_outputs': 1
    }


def fetch_insurance(
    path: str = './data/',
    fold: int = 0
) -> dict:

    """Download Insurance dataset.

    Args:
        path: where the data should be stored.
        fold: data fold number.

    Returns:
        data: it contains the following keys
            - 'x_train', 'y_train', 'x_test', 'y_test': train/test sets. x is a pandas dataframe and y is a numpy array.
            - 'categorical_features', 'continuous_features': which features are categorical/continuous.
            - 'task': which target task type. either 'reg', 'bincls', or 'multicls'.
            - 'n_features': the number of input features.
            - 'n_outputs': the number of outputs.
    """

    assert 0 <= fold <= 4, 'invalid fold number.'

    data_path = pjoin(path, 'insurance', 'insurance.csv')
    if not pexists(data_path):
        os.makedirs(pjoin(path, 'insurance'), exist_ok=True)
        file_id = "1hhsC8aRXQS-TLqEk9hvKXIW3m3voeaz0"
        _download_file_from_google_drive(file_id, data_path)

    categorical_features = ['sex', 'smoker', 'region']
    continuous_features = ['age', 'bmi', 'children']
    targets = ['charges']
    usecols = categorical_features + continuous_features + targets

    df = pd.read_csv(data_path, usecols=usecols)
    x = df.drop(targets, axis=1)
    y = df.get(targets).values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return {
        'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test,
        'categorical_features': categorical_features, 'continuous_features': continuous_features,
        'task': 'regression', 'n_features': len(categorical_features + continuous_features), 'n_outputs': 1
    }


def fetch_house_prices(
    path: str = './data/',
    fold: int = 0
) -> dict:

    """Download House Prices dataset.

    Args:
        path: where the data should be stored.
        fold: data fold number.

    Returns:
        data: it contains the following keys
            - 'x_train', 'y_train', 'x_test', 'y_test': train/test sets. x is a pandas dataframe and y is a numpy array.
            - 'categorical_features', 'continuous_features': which features are categorical/continuous.
            - 'task': which target task type. either 'reg', 'bincls', or 'multicls'.
            - 'n_features': the number of input features.
            - 'n_outputs': the number of outputs.
    """

    assert 0 <= fold <= 4, 'invalid fold number.'

    data_path = pjoin(path, 'house', 'house_prices.csv')
    if not pexists(data_path):
        os.makedirs(pjoin(path, 'house'), exist_ok=True)
        file_id = "15gTfFGFm3V31xGv5q1wASYtV0wqBmirH"
        _download_file_from_google_drive(file_id, data_path)

    categorical_features = [
        'MSSubClass', 'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
        'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
        'Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'Heating', 'HeatingQC',
        'CentralAir', 'KitchenQual', 'Functional', 'PavedDrive', 'SaleType', 'SaleCondition'
    ]
    continuous_features = [
        'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2',
        'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
        'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
        'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
        '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'
    ]
    targets = ['SalePrice']
    usecols = categorical_features + continuous_features + targets

    df = pd.read_csv(data_path, usecols=usecols)
    x = df.drop(targets, axis=1)
    y = df.get(targets).values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return {
        'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test,
        'categorical_features': categorical_features, 'continuous_features': continuous_features,
        'task': 'regression', 'n_features': len(categorical_features + continuous_features), 'n_outputs': 1
    }


def fetch_bikeshare(
    path: str = './data/',
    fold: int = 0
) -> dict:

    """Download Bikeshare dataset.

    Args:
        path: where the data should be stored.
        fold: data fold number.

    Returns:
        data: it contains the following keys
            - 'x_train', 'y_train', 'x_test', 'y_test': train/test sets. x is a pandas dataframe and y is a numpy array.
            - 'categorical_features', 'continuous_features': which features are categorical/continuous.
            - 'task': which target task type. either 'reg', 'bincls', or 'multicls'.
            - 'n_features': the number of input features.
            - 'n_outputs': the number of outputs.
    """

    assert 0 <= fold <= 4, 'invalid fold number.'

    data_path = pjoin(path, 'bikeshare', 'hour.csv')
    if not pexists(data_path):
        os.makedirs(pjoin(path, 'bikeshare'), exist_ok=True)
        _download_file_from_onedrive('https://1drv.ms/u/s!ArHmmFHCSXTIg8gDTrnCO2vDulTygA?e=wCHjgF', pjoin(path, 'bikeshare.zip'))
        with ZipFile(pjoin(path, 'bikeshare.zip'), 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(path)
        os.remove(pjoin(path, 'bikeshare.zip'))

    categorical_features = None
    continuous_features = [
        'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp',
        'hum', 'windspeed'
    ]
    targets = ['cnt']
    usecols = continuous_features + targets

    df = pd.read_csv(data_path, usecols=usecols)
    x = df.drop(targets, axis=1)
    y = df.get(targets).values

    train_idx = pd.read_csv(pjoin(path, 'bikeshare', 'train%d.txt' % fold), header=None)[0].values
    test_idx = pd.read_csv(pjoin(path, 'bikeshare', 'test%d.txt' % fold), header=None)[0].values
    
    return {
        'x_train': x.iloc[train_idx], 'y_train': y[train_idx], 'x_test': x.iloc[test_idx], 'y_test': y[test_idx],
        'categorical_features': categorical_features, 'continuous_features': continuous_features,
        'task': 'regression', 'n_features': len(continuous_features), 'n_outputs': 1
    }


def fetch_year(
    path: str = './data/',
    test_size: int = 51630,
    fold: int = 0
) -> dict:

    """Download Year dataset.

    Args:
        path: where the data should be stored.
        test_size: teset set size.
        fold: data fold number.

    Returns:
        data: it contains the following keys
            - 'x_train', 'y_train', 'x_test', 'y_test': train/test sets. x is a pandas dataframe and y is a numpy array.
            - 'categorical_features', 'continuous_features': which features are categorical/continuous.
            - 'task': which target task type. either 'reg', 'bincls', or 'multicls'.
            - 'n_features': the number of input features.
            - 'n_outputs': the number of outputs.
    """

    assert 0 <= fold <= 4, 'invalid fold number.'

    path = pjoin(path, 'year')

    data_path = pjoin(path, 'data.csv')
    if not pexists(data_path):
        os.makedirs(path, exist_ok=True)
        _download('https://www.dropbox.com/s/l09pug0ywaqsy0e/YearPredictionMSD.txt?dl=1', data_path)
    n_features = 91
    types = {i: (np.float32 if i != 0 else np.int32) for i in range(n_features)}

    categorical_features = None
    continuous_features = list(range(1, n_features))
    targets = [0]

    df = pd.read_csv(data_path, header=None, dtype=types)
    x = df.drop(targets, axis=1)
    y = df.get(targets).values

    return {
        'x_train': x.iloc[:-test_size], 'y_train': y[:-test_size], 'x_test': x.iloc[-test_size:], 'y_test': y[-test_size:],
        'categorical_features': categorical_features, 'continuous_features': continuous_features,
        'task': 'regression', 'n_features': len(continuous_features), 'n_outputs': 1
    }


DATASET_MAP = {
    'ca_housing': fetch_ca_housing,
    'insurance': fetch_insurance,
    'house_prices': fetch_house_prices,
    'bikeshare': fetch_bikeshare,
    'year': fetch_year
}
