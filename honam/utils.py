import pickle
from typing import Union, Any

import torch
import torchmetrics
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from honam._types import _TASK_INPUT

__all__ = ['get_loader']


def get_loader(
    x: np.ndarray,
    y: Union[None, np.ndarray] = None,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0
) -> DataLoader:
    
    """Get torch dataloader.

    Args:
        x: input features.
        y: targets. None indicates no target values
        batch_size: how many samples per batch to load.
        shuffle: set to True to have the data reshuffled at every epoch.
        num_workers: how many subprocesses to use for data loading.

    Returns:
        data loader.
    """

    x = torch.tensor(x, dtype=torch.float32)
    if y is not None:
        y = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(x, y)
    else:
        dataset = TensorDataset(x)

    loader = DataLoader(dataset, batch_size, shuffle, num_workers=num_workers)

    return loader


def get_r_squared(
    y_predict: torch.Tensor,
    y_true: torch.Tensor
):
    mean = y_true.mean(dim=0)
    sst = torch.square(y_true - mean).sum(dim=0)
    ssr = torch.square(y_true - y_predict).sum(dim=0)
    score = 1 - (ssr / sst)
    return score.mean()

def get_r_absolute(
    y_predict: torch.Tensor,
    y_true: torch.Tensor
):
    mean = y_true.mean(dim=0)
    sat = torch.abs(y_true - mean).sum(dim=0)
    sar = torch.abs(y_true - y_predict).sum(dim=0)
    score = 1 - (sar / sat)
    return score.mean()

def evaluate(
    y_predict: torch.Tensor,
    y_true: torch.Tensor,
    task: _TASK_INPUT
):
    if task == "regression":
        r_squared = get_r_squared(y_true, y_predict)
        r_absolute = get_r_absolute(y_true, y_predict)
        rmse = np.sqrt(torchmetrics.functional.mean_squared_error(y_predict, y_true))
        mae = torchmetrics.functional.mean_absolute_error(y_predict, y_true)
        print("R-squared: {:.7f}, R-absolute: {:.7f}, RMSE: {:.7f}, MAE: {:.7f}".format(r_squared, r_absolute, rmse, mae))
    else:
        auroc = torchmetrics.functional.auroc(y_predict, y_true, task)
        auprc = torchmetrics.functional.average_precision(y_predict, y_true, task)
        y_predict = (y_predict >= 0.5).astype('int')
        acc = torchmetrics.functional.accuracy(y_predict, y_true, task)
        f1 = torchmetrics.functional.f1_score(y_predict, y_true, task)
        print("AUROC: {:.7f}, AUPRC: {:.7f}, Acc: {:.7f}, F1: {:.7f}".format(auroc, auprc, acc, f1))


def save_pickle(
    obj: Any,
    path: str
) -> None:
    with open(path, mode='wb') as file:
        pickle.dump(obj, file)


def load_pickle(
    path: str
) -> Any:
    with open(path, mode='rb') as file:
        obj = pickle.load(file)
    return obj
