import numpy as np

from sklearn.model_selection import train_test_split


def r_squared(targets, predicts):

    mean = np.mean(targets)
    sst = np.square(targets - mean).sum()
    ssr = np.square(targets - predicts).sum()

    score = 1 - (ssr / sst)

    return score


def r_absolute(targets, predicts):

    mean = np.mean(targets)
    sat = np.abs(targets - mean).sum()
    sar = np.abs(targets - predicts).sum()

    score = 1 - (sar / sat)

    return score


def adjusted_r_squared(targets, predicts, n_samples, n_features):

    score = r_squared(targets, predicts)

    adjusted_score = 1 - (1 - score) * (n_samples - 1) / (n_samples - n_features - 1)

    return adjusted_score


def adjusted_r_absolute(targets, predicts, n_samples, n_features):

    score = r_absolute(targets, predicts)

    adjusted_score = 1 - (1 - score) * (n_samples - 1) / (n_samples - n_features - 1)

    return adjusted_score


def imbalance_split(data):

    false_samples = data[data["target"] == 0]
    true_samples = data[data["target"] == 1]

    num_true_samples = true_samples.shape[0]
    train_size = int(num_true_samples * 0.8)

    train_false_data, test_false_data = train_test_split(false_samples, train_size=train_size)
    val_false_data, test_false_data = train_test_split(test_false_data, test_size=0.5)

    train_true_data, test_true_data = train_test_split(true_samples, train_size=train_size)
    val_true_data, test_true_data = train_test_split(test_true_data, test_size=0.5)

    train_data = train_false_data.append(train_true_data)
    val_data = val_false_data.append(val_true_data)
    test_data = test_false_data.append(test_true_data)

    return train_data, val_data, test_data
