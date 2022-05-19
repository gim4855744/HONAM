import pandas as pd
import numpy as np
import xgboost
import random
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from absl import app, flags

from data.load_data import california_housing_prices, insurance, house_prices, fico, credit, mimic

from eval import eval_regression_task, eval_binary_classification_task

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 1, "")
flags.DEFINE_string("dataset", "", "")
flags.DEFINE_string("model", "", "")
flags.DEFINE_integer("order", 0, "the order of feature interactions")
flags.DEFINE_bool("verbose", False, " if True, print train and val loss every epoch")


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
    else:
        raise Exception

    if dataset == "ca_housing" or dataset == "insurance" or dataset == "house_prices":
        task = "regression"
    elif dataset == "fico" or dataset == "credit" or dataset == "mimic":
        task = "classification"
    else:
        raise Exception

    # get numerical feature columns
    feature_names = data.drop("target", axis=1).columns
    numeric_feature_names = [name for name in feature_names if data[name].dtype != "object"]  # only numerical features

    # make dummy features
    data = pd.get_dummies(data)

    print(data["target"].sum() / len(data))
    exit()

    # set data info
    num_samples = data.shape[0]
    num_features = data.shape[1] - 1

    # train, val, test split
    test_size = len(data) // 5
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=test_size)

    # feature scaling
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    train_data[numeric_feature_names] = feature_scaler.fit_transform(train_data[numeric_feature_names].values)
    val_data[numeric_feature_names] = feature_scaler.transform(val_data[numeric_feature_names].values)
    test_data[numeric_feature_names] = feature_scaler.transform(test_data[numeric_feature_names].values)

    train_data["target"] = target_scaler.fit_transform(train_data["target"].values.reshape(-1, 1))
    val_data["target"] = target_scaler.transform(val_data["target"].values.reshape(-1, 1))
    test_data["target"] = target_scaler.transform(test_data["target"].values.reshape(-1, 1))

    train_x, train_y = train_data.drop("target", axis=1).values, train_data["target"].values
    val_x, val_y = val_data.drop("target", axis=1).values, val_data["target"].values
    test_x, test_y = test_data.drop("target", axis=1).values, test_data["target"].values

    train_data = xgboost.DMatrix(train_x, label=train_y)
    val_data = xgboost.DMatrix(val_x, label=val_y)
    test_data = xgboost.DMatrix(test_x, label=test_y)

    if task == "regression":
        params = {"objective": "reg:squarederror"}
        eval_metric = "rmse"
    else:
        params = {"objective": "binary:logistic"}
        eval_metric = "logloss"

    xgboost.set_config(verbosity=0)
    eval_set = [(val_data, eval_metric)]
    bst = xgboost.train(params, train_data,
                        num_boost_round=1000, evals=eval_set, verbose_eval=FLAGS.verbose, early_stopping_rounds=10)

    total_predicts = bst.predict(test_data).reshape(-1, 1)
    total_targets = test_y.reshape(-1, 1)

    total_predicts = target_scaler.inverse_transform(total_predicts)
    total_targets = target_scaler.inverse_transform(total_targets)

    if task == "regression":
        eval_regression_task(total_targets, total_predicts, num_samples, num_features)
    else:
        eval_binary_classification_task(total_targets, total_predicts)


if __name__ == '__main__':
    app.run(main)
