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


def fetch_california_housing(
    path: str = './data/',
    fold: int = 0
) -> dict:

    """Download California Housing dataset.

    Args:
        path: where the data should be stored.
        fold: data fold number.

    Returns:
        data: it contains the following items
            - 'x_train', 'y_train', 'x_test', 'y_test': train/test sets. x and y are pandas dataframes.
            - 'cat_feats', 'num_feats': which features are categorical/numerical.
            - 'task': which target task type. either 'reg', 'bincls', or 'multicls'.
    """

    assert 0 <= fold <= 4, 'invalid fold number.'

    dataset_name = 'california_housing'
    data_dir = pjoin(path, dataset_name)
    data_path = pjoin(data_dir, f'{dataset_name}.csv')
    if not pexists(data_path):
        os.makedirs(data_dir, exist_ok=True)
        _download_file_from_google_drive('1L-mAY0PBJZ7SFEDaAUYfWa4ckJetD-hc', data_path)

    cat_cols = ['ocean_proximity']
    num_cols = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'population', 'households', 'median_income']
    target_cols = ['median_house_value']
    usecols = cat_cols + num_cols + target_cols

    df = pd.read_csv(data_path, usecols=usecols)
    x = df.drop(target_cols, axis=1)
    y = df.get(target_cols)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return {
        'x_train': x_train, 'y_train': y_train,
        'x_test': x_test, 'y_test': y_test,
        'cat_feats': cat_cols, 'num_feats': num_cols,
        'task': 'reg'
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
        data: it contains the following items
            - 'x_train', 'y_train', 'x_test', 'y_test': train/test sets. x and y are pandas dataframes.
            - 'cat_feats', 'num_feats': which features are categorical/numerical.
            - 'task': which target task type. either 'reg', 'bincls', or 'multicls'.
    """

    assert 0 <= fold <= 4, 'invalid fold number.'

    dataset_name = 'insurance'
    data_dir = pjoin(path, dataset_name)
    data_path = pjoin(data_dir, f'{dataset_name}.csv')
    if not pexists(data_path):
        os.makedirs(data_dir, exist_ok=True)
        _download_file_from_google_drive('1hhsC8aRXQS-TLqEk9hvKXIW3m3voeaz0', data_path)

    cat_cols = ['sex', 'smoker', 'region']
    num_cols = ['age', 'bmi', 'children']
    target_cols = ['charges']
    usecols = cat_cols + num_cols + target_cols

    df = pd.read_csv(data_path, usecols=usecols)
    x = df.drop(target_cols, axis=1)
    y = df.get(target_cols)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return {
        'x_train': x_train, 'y_train': y_train,
        'x_test': x_test, 'y_test': y_test,
        'cat_feats': cat_cols, 'num_feats': num_cols,
        'task': 'reg'
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
        data: it contains the following items
            - 'x_train', 'y_train', 'x_test', 'y_test': train/test sets. x and y are pandas dataframes.
            - 'cat_feats', 'num_feats': which features are categorical/numerical.
            - 'task': which target task type. either 'reg', 'bincls', or 'multicls'.
    """

    assert 0 <= fold <= 4, 'invalid fold number.'

    dataset_name = 'house_prices'
    data_dir = pjoin(path, dataset_name)
    data_path = pjoin(data_dir, f'{dataset_name}.csv')
    if not pexists(data_path):
        os.makedirs(data_dir, exist_ok=True)
        _download_file_from_google_drive('15gTfFGFm3V31xGv5q1wASYtV0wqBmirH', data_path)

    cat_cols = [
        'MSSubClass', 'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
        'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
        'Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'Heating', 'HeatingQC',
        'CentralAir', 'KitchenQual', 'Functional', 'PavedDrive', 'SaleType', 'SaleCondition'
    ]
    num_cols = [
        'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2',
        'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
        'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
        'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
        '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'
    ]
    target_cols = ['SalePrice']
    usecols = cat_cols + num_cols + target_cols

    df = pd.read_csv(data_path, usecols=usecols)
    x = df.drop(target_cols, axis=1)
    y = df.get(target_cols)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return {
        'x_train': x_train, 'y_train': y_train,
        'x_test': x_test, 'y_test': y_test,
        'cat_feats': cat_cols, 'num_feats': num_cols,
        'task': 'reg'
    }


def fetch_fico(
    path: str = './data/',
    fold: int = 0
) -> dict:
    
    """Download FICO dataset.

    Args:
        path: where the data should be stored.
        fold: data fold number.

    Returns:
        data: it contains the following items
            - 'x_train', 'y_train', 'x_test', 'y_test': train/test sets. x and y are pandas dataframes.
            - 'cat_feats', 'num_feats': which features are categorical/numerical.
            - 'task': which target task type. either 'reg', 'bincls', or 'multicls'.
    """

    assert 0 <= fold <= 4, 'invalid fold number.'

    dataset_name = 'fico'
    data_dir = pjoin(path, dataset_name)
    data_path = pjoin(data_dir, f'{dataset_name}.csv')
    if not pexists(data_path):
        os.makedirs(data_dir, exist_ok=True)
        _download_file_from_google_drive('16keHIu0OSwi9v-7mlJYdsdyL-GYBetym', data_path)

    cat_cols = None
    num_cols = [
        'ExternalRiskEstimate', 'MSinceOldestTradeOpen', 'MSinceMostRecentTradeOpen', 'AverageMInFile', 'NumSatisfactoryTrades',
        'NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec', 'PercentTradesNeverDelq', 'MSinceMostRecentDelq',
        'MaxDelq2PublicRecLast12M', 'MaxDelqEver', 'NumTotalTrades', 'NumTradesOpeninLast12M', 'PercentInstallTrades',
        'MSinceMostRecentInqexcl7days', 'NumInqLast6M', 'NumInqLast6Mexcl7days', 'NetFractionRevolvingBurden', 'NetFractionInstallBurden',
        'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance', 'NumBank2NatlTradesWHighUtilization', 'PercentTradesWBalance'
    ]
    target_cols = ['RiskPerformance']
    usecols = num_cols + target_cols

    df = pd.read_csv(data_path, usecols=usecols)
    x = df.drop(target_cols, axis=1)
    y = df.get(target_cols)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return {
        'x_train': x_train, 'y_train': y_train,
        'x_test': x_test, 'y_test': y_test,
        'cat_feats': cat_cols, 'num_feats': num_cols,
        'task': 'bincls'
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
        data: it contains the following items
            - 'x_train', 'y_train', 'x_test', 'y_test': train/test sets. x and y are pandas dataframes.
            - 'cat_feats', 'num_feats': which features are categorical/numerical.
            - 'task': which target task type. either 'reg', 'bincls', or 'multicls'.
    """

    assert 0 <= fold <= 4, 'invalid fold number.'

    dataset_name = 'bikeshare'
    data_dir = pjoin(path, dataset_name)
    data_path = pjoin(data_dir, 'hour.csv')
    if not pexists(data_path):
        os.makedirs(data_dir, exist_ok=True)
        data_zip_path = pjoin(path, f'{dataset_name}.zip')
        _download_file_from_onedrive('https://1drv.ms/u/s!ArHmmFHCSXTIg8gDTrnCO2vDulTygA?e=wCHjgF', data_zip_path)
        with ZipFile(data_zip_path, 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(path)
        os.remove(data_zip_path)

    cat_cols = None
    num_cols = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
    target_cols = ['cnt']
    usecols = num_cols + target_cols

    df = pd.read_csv(data_path, usecols=usecols)
    x = df.drop(target_cols, axis=1)
    y = df.get(target_cols)

    train_idx = pd.read_csv(pjoin(data_dir, f'train{fold}.txt'), header=None)[0].values
    test_idx = pd.read_csv(pjoin(data_dir, f'test{fold}.txt'), header=None)[0].values
    
    return {
        'x_train': x.iloc[train_idx], 'y_train': y.iloc[train_idx],
        'x_test': x.iloc[test_idx], 'y_test': y.iloc[test_idx],
        'cat_feats': cat_cols, 'num_feats': num_cols,
        'task': 'reg'
    }


def fetch_year(
    path: str = './data/',
    test_size: int = 51630,
    fold: int = 0
) -> dict:

    """Download Year dataset.

    Args:
        path: where the data should be stored.
        fold: data fold number.

    Returns:
        data: it contains the following items
            - 'x_train', 'y_train', 'x_test', 'y_test': train/test sets. x and y are pandas dataframes.
            - 'cat_feats', 'num_feats': which features are categorical/numerical.
            - 'task': which target task type. either 'reg', 'bincls', or 'multicls'.
    """

    assert 0 <= fold <= 4, 'invalid fold number.'

    dataset_name = 'year'
    data_dir = pjoin(path, dataset_name)
    data_path = pjoin(data_dir, 'data.csv')
    if not pexists(data_path):
        os.makedirs(data_dir, exist_ok=True)
        _download('https://www.dropbox.com/s/l09pug0ywaqsy0e/YearPredictionMSD.txt?dl=1', data_path)
    n_features = 91
    types = {i: (np.float32 if i != 0 else np.int32) for i in range(n_features)}

    cat_cols = None
    num_cols = list(range(1, n_features))
    target_cols = [0]
    usecols = num_cols + target_cols

    df = pd.read_csv(data_path, header=None, dtype=types, usecols=usecols)
    x = df.drop(target_cols, axis=1)
    y = df.get(target_cols)

    return {
        'x_train': x.iloc[:-test_size], 'y_train': y.iloc[:-test_size],
        'x_test': x.iloc[-test_size:], 'y_test': y.iloc[-test_size:],
        'cat_feats': cat_cols, 'num_feats': num_cols,
        'task': 'reg'
    }


def fetch_credit(
    path: str = './data/',
    fold: int = 0
) -> dict:

    """Download Credit dataset.

    Args:
        path: where the data should be stored.
        fold: data fold number.

    Returns:
        data: it contains the following items
            - 'x_train', 'y_train', 'x_test', 'y_test': train/test sets. x and y are pandas dataframes.
            - 'cat_feats', 'num_feats': which features are categorical/numerical.
            - 'task': which target task type. either 'reg', 'bincls', or 'multicls'.
    """

    assert 0 <= fold <= 4, 'invalid fold number.'

    dataset_name = 'credit'
    data_dir = pjoin(path, dataset_name)
    data_path = pjoin(data_dir, 'creditcard.csv')
    if not pexists(data_path):
        os.makedirs(data_dir, exist_ok=True)
        data_zip_path = pjoin(path, f'{dataset_name}.zip')
        _download_file_from_onedrive('https://1drv.ms/u/s!ArHmmFHCSXTIg8d_VZKCFHe7_uNkpw?e=vOSV3S', data_zip_path)
        with ZipFile(data_zip_path, 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(path)
        os.remove(data_zip_path)

    cat_cols = None
    num_cols = [
        'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14',
        'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
    ]
    target_cols = ['Class']
    usecols = num_cols + target_cols

    df = pd.read_csv(data_path, usecols=usecols)
    x = df.drop(target_cols, axis=1)
    y = df.get(target_cols)

    train_idx = pd.read_csv(pjoin(data_dir, f'train{fold}.txt'), header=None)[0].values
    test_idx = pd.read_csv(pjoin(data_dir, f'test{fold}.txt'), header=None)[0].values
    
    return {
        'x_train': x.iloc[train_idx], 'y_train': y.iloc[train_idx],
        'x_test': x.iloc[test_idx], 'y_test': y.iloc[test_idx],
        'cat_feats': cat_cols, 'num_feats': num_cols,
        'task': 'bincls'
    }


def fetch_support2(
    path: str = './data/',
    fold: int = 0
) -> dict:

    """Download SUPPORT2 dataset.

    Args:
        path: where the data should be stored.
        fold: data fold number.

    Returns:
        data: it contains the following items
            - 'x_train', 'y_train', 'x_test', 'y_test': train/test sets. x and y are pandas dataframes.
            - 'cat_feats', 'num_feats': which features are categorical/numerical.
            - 'task': which target task type. either 'reg', 'bincls', or 'multicls'.
    """

    assert 0 <= fold <= 4, 'invalid fold number.'

    dataset_name = 'support2'
    data_dir = pjoin(path, dataset_name)
    data_path = pjoin(data_dir, 'support2.csv')
    if not pexists(data_path):
        os.makedirs(data_dir, exist_ok=True)
        data_zip_path = pjoin(path, f'{dataset_name}.zip')
        _download_file_from_onedrive('https://1drv.ms/u/s!ArHmmFHCSXTIg8d74X-u6bwQjhJTIA?e=DV7GIK', data_zip_path)
        with ZipFile(data_zip_path, 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(path)
        os.remove(data_zip_path)

    cat_cols = ['sex', 'dzclass', 'race', 'ca', 'income']
    num_cols = [
        'age', 'num.co', 'edu', 'scoma', 'sps', 'hday', 'diabetes', 'dementia', 'meanbp', 'wblc', 'hrt', 'resp',
        'temp', 'pafi', 'alb', 'bili', 'crea', 'sod', 'ph', 'glucose', 'bun', 'urine', 'adlp', 'adls', 'adlsc'
    ]
    target_cols = ['hospdead']
    usecols = cat_cols + num_cols + target_cols

    df = pd.read_csv(data_path, usecols=usecols)
    x = df.drop(target_cols, axis=1)
    y = df.get(target_cols)

    x[num_cols] = x[num_cols].fillna(0.)
    x.loc[x['income'].isna(), 'income'] = 'NaN'
    x.loc[x['income'] == 'under $11k', 'income'] = ' <$11k'
    x.loc[x['race'].isna(), 'race'] = 'NaN'

    train_idx = pd.read_csv(pjoin(data_dir, f'train{fold}.txt'), header=None)[0].values
    test_idx = pd.read_csv(pjoin(data_dir, f'test{fold}.txt'), header=None)[0].values
    
    return {
        'x_train': x.iloc[train_idx], 'y_train': y.iloc[train_idx],
        'x_test': x.iloc[test_idx], 'y_test': y.iloc[test_idx],
        'cat_feats': cat_cols, 'num_feats': num_cols,
        'task': 'bincls'
    }


def fetch_mimic3(
    path: str = './data/',
    fold: int = 0
) -> dict:

    """Download MIMIC-III dataset.

    Args:
        path: where the data should be stored.
        fold: data fold number.

    Returns:
        data: it contains the following items
            - 'x_train', 'y_train', 'x_test', 'y_test': train/test sets. x and y are pandas dataframes.
            - 'cat_feats', 'num_feats': which features are categorical/numerical.
            - 'task': which target task type. either 'reg', 'bincls', or 'multicls'.
    """

    assert 0 <= fold <= 4, 'invalid fold number.'

    dataset_name = 'mimic3'
    data_dir = pjoin(path, dataset_name)
    data_path = pjoin(data_dir, 'adult_icu.gz')
    if not pexists(data_path):
        os.makedirs(data_dir, exist_ok=True)
        data_zip_path = pjoin(path, f'{dataset_name}.zip')
        _download_file_from_onedrive('https://1drv.ms/u/s!ArHmmFHCSXTIg8d6o6icAULya24iyw?e=s7TNxa', data_zip_path)
        with ZipFile(data_zip_path, 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(path)
        os.remove(data_zip_path)

    cat_cols = None
    num_cols = [
        'age', 'first_hosp_stay', 'first_icu_stay', 'adult_icu', 'eth_asian', 'eth_black', 'eth_hispanic', 'eth_other', 'eth_white',
        'admType_ELECTIVE', 'admType_EMERGENCY', 'admType_NEWBORN', 'admType_URGENT', 'heartrate_min', 'heartrate_max', 'heartrate_mean',
        'sysbp_min', 'sysbp_max', 'sysbp_mean', 'diasbp_min', 'diasbp_max', 'diasbp_mean', 'meanbp_min', 'meanbp_max', 'meanbp_mean',
        'resprate_min', 'resprate_max', 'resprate_mean', 'tempc_min', 'tempc_max', 'tempc_mean', 'spo2_min', 'spo2_max', 'spo2_mean',
        'glucose_min', 'glucose_max', 'glucose_mean', 'aniongap', 'albumin', 'bicarbonate', 'bilirubin', 'creatinine', 'chloride', 'glucose',
        'hematocrit', 'hemoglobin', 'lactate', 'magnesium', 'phosphate', 'platelet', 'potassium', 'ptt', 'inr', 'pt', 'sodium', 'bun', 'wbc'
    ]
    target_cols = ['mort_icu']
    usecols = num_cols + target_cols

    df = pd.read_csv(data_path, compression='gzip', usecols=usecols)
    x = df.drop(target_cols, axis=1)
    y = df.get(target_cols)

    train_idx = pd.read_csv(pjoin(data_dir, f'train{fold}.txt'), header=None)[0].values
    test_idx = pd.read_csv(pjoin(data_dir, f'test{fold}.txt'), header=None)[0].values
    
    return {
        'x_train': x.iloc[train_idx], 'y_train': y.iloc[train_idx],
        'x_test': x.iloc[test_idx], 'y_test': y.iloc[test_idx],
        'cat_feats': cat_cols, 'num_feats': num_cols,
        'task': 'bincls'
    }


def fetch_click(
    path: str = './data/',
    test_size: int = 100000,
    fold: int = 0):

    """Download Click dataset.

    Args:
        path: where the data should be stored.
        fold: data fold number.

    Returns:
        data: it contains the following items
            - 'x_train', 'y_train', 'x_test', 'y_test': train/test sets. x and y are pandas dataframes.
            - 'cat_feats', 'num_feats': which features are categorical/numerical.
            - 'task': which target task type. either 'reg', 'bincls', or 'multicls'.
    """

    dataset_name = 'click'
    data_dir = pjoin(path, dataset_name)
    data_path = pjoin(data_dir, 'click.csv')
    if not pexists(data_path):
        os.makedirs(data_dir, exist_ok=True)
        _download('https://www.dropbox.com/s/w43ylgrl331svqc/click.csv?dl=1', data_path)

    cat_cols = ['url_hash', 'ad_id', 'advertiser_id', 'query_id', 'keyword_id', 'title_id', 'description_id', 'user_id']
    num_cols = ['impression', 'depth', 'position']
    target_cols = ['target']
    usecols = cat_cols + num_cols + target_cols

    df = pd.read_csv(data_path, index_col=0, usecols=usecols)
    x = df.drop(target_cols, axis=1)
    y = df.get(target_cols)
    
    x_train, x_test = x.iloc[:-test_size].copy(), x.iloc[-test_size:].copy()
    y_train, y_test = y.iloc[:-test_size].copy(), y.iloc[-test_size:].copy()

    return {
        'x_train': x_train, 'y_train': y_train,
        'x_test': x_test, 'y_test': y_test,
        'cat_feats': cat_cols, 'num_feats': num_cols,
        'task': 'bincls'
    }


DATASET_MAP = {
    'california_housing': fetch_california_housing,
    'insurance': fetch_insurance,
    'house_prices': fetch_house_prices,
    'fico': fetch_fico,
    'bikeshare': fetch_bikeshare,
    'year': fetch_year,
    'credit': fetch_credit,
    'support2': fetch_support2,
    'mimic3': fetch_mimic3,
    'click': fetch_click
}
