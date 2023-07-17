import random
import pickle

import numpy as np
import torch

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from torch.backends import cudnn

def set_global_seed(seed: int) -> None:

    """
    Set the global random seed for reproducibility.
    :param seed: a random seed number
    """

    cudnn.deterministic = True
    cudnn.benchmark = False

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_r_squared(y_true: np.ndarray, y_predict: np.ndarray):
    mean = np.mean(y_true)
    sst = np.square(y_true - mean).sum()
    ssr = np.square(y_true - y_predict).sum()
    score = 1 - (ssr / sst)
    return score

def get_r_absolute(y_true: np.ndarray, y_predict: np.ndarray):
    mean = np.mean(y_true)
    sat = np.abs(y_true - mean).sum()
    sar = np.abs(y_true - y_predict).sum()
    score = 1 - (sar / sat)
    return score

def evaluate(y_true: np.ndarray, y_predict: np.ndarray, task: str):
    if task == "regression":
        r_squared = get_r_squared(y_true, y_predict)
        r_absolute = get_r_absolute(y_true, y_predict)
        rmse = np.sqrt(mean_squared_error(y_true, y_predict))
        mae = mean_absolute_error(y_true, y_predict)
        print("R-squared: {:.7f}, R-absolute: {:.7f}, RMSE: {:.7f}, MAE: {:.7f}".format(r_squared, r_absolute, rmse, mae))
    elif task == "classification":
        auroc = roc_auc_score(y_true, y_predict)
        auprc = average_precision_score(y_true, y_predict)
        y_predict = (y_predict >= 0.5).astype('int')
        acc = accuracy_score(y_true, y_predict)
        f1 = f1_score(y_true, y_predict)
        print("AUROC: {:.7f}, AUPRC: {:.7f}, Acc: {:.7f}, F1: {:.7f}".format(auroc, auprc, acc, f1))


def save_pickle(obj, path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)


def load_pickle(path):
    with open(path, 'rb') as file:
        obj = pickle.load(file)
    return obj
