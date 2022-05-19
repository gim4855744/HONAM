import pandas as pd
import numpy as np
import random
import torch
import os

from torch.utils.data.dataloader import DataLoader
from torch.backends import cudnn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from absl import app, flags

from data.load_data import california_housing_prices, insurance, house_prices, fico, credit, mimic, compas
from data.dataset import TensorDataset

from model.lr import LR
from model.crossnet import CrossNet
from model.mlp import MLP
from model.nam import NAM
from model.namcross import NAMWithCrossNet
from model.honam import HONAM

from eval import eval_regression_task, eval_binary_classification_task

cudnn.deterministic = True
cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 1, "")
flags.DEFINE_string("dataset", "", "")
flags.DEFINE_string("model", "", "")
flags.DEFINE_integer("order", 0, "the order of feature interactions")


def main(argv):

    seed = FLAGS.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset = FLAGS.dataset
    if dataset == "ca_housing":
        data = california_housing_prices()
    elif dataset == "insurance":
        data = insurance()
    elif dataset == "house_prices":
        data = house_prices()
    elif dataset == "fico":
        data = fico()
    elif dataset == "credit":
        data = credit()
    elif dataset == "mimic":
        data = mimic()
    elif dataset == "compas":
        data = compas()
    else:
        raise Exception

    if dataset == "ca_housing" or dataset == "insurance" or dataset == "house_prices":
        task = "regression"
    elif dataset == "fico" or dataset == "credit" or dataset == "mimic" or dataset == "compas":
        task = "classification"
    else:
        raise Exception

    # get numerical feature columns
    feature_names = data.drop("target", axis=1).columns
    feature_names = [name for name in feature_names if data[name].dtype != "object"]  # only numerical features

    # make dummy features
    data = pd.get_dummies(data)

    # set data info
    num_samples = data.shape[0]
    num_features = data.shape[1] - 1
    out_size = 1

    # set hyper-parameters
    batch_size = num_samples // 100

    # train, val, test split
    test_size = len(data) // 5
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=test_size)

    # feature scaling
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    train_data[feature_names] = feature_scaler.fit_transform(train_data[feature_names].values)
    val_data[feature_names] = feature_scaler.transform(val_data[feature_names].values)
    test_data[feature_names] = feature_scaler.transform(test_data[feature_names].values)

    train_data["target"] = target_scaler.fit_transform(train_data["target"].values.reshape(-1, 1))
    val_data["target"] = target_scaler.transform(val_data["target"].values.reshape(-1, 1))
    test_data["target"] = target_scaler.transform(test_data["target"].values.reshape(-1, 1))

    # torch data module
    test_dataset = TensorDataset(test_data, device)
    test_loader = DataLoader(test_dataset, batch_size)

    # model define

    model_name = FLAGS.model
    order = FLAGS.order
    model_dict = {"lr": LR,
                  "crossnet": CrossNet,
                  "mlp": MLP,
                  "nam": NAM,
                  "namcross": NAMWithCrossNet,
                  "honam": HONAM}

    model = model_dict[model_name](num_features, out_size, order=order).to(device)

    save_dir = "/home/minkyu/Projects/HONAMv2/save/{}/{}/".format(dataset, model.__class__.__name__)
    save_path = os.path.join(save_dir, "order{}_seed{}.ckpt".format(order, seed))

    # test

    model.load_state_dict(torch.load(save_path))
    total_predicts, total_targets = [], []

    model.eval()

    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            predicts = model(batch_features)
            total_predicts.extend(predicts.detach().cpu().tolist())
            total_targets.extend(batch_targets.detach().cpu().tolist())

    total_predicts = target_scaler.inverse_transform(total_predicts)
    total_targets = target_scaler.inverse_transform(total_targets)

    if task == "regression":
        eval_regression_task(total_targets, total_predicts, num_samples, num_features)
    else:
        eval_binary_classification_task(total_targets, total_predicts)


if __name__ == '__main__':
    app.run(main)
