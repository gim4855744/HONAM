import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

from utils import r_squared, r_absolute, adjusted_r_squared, adjusted_r_absolute


def eval_regression_task(total_targets, total_predicts, num_samples, num_features):

    r_squared_score = r_squared(total_targets, total_predicts)
    r_absolute_score = r_absolute(total_targets, total_predicts)

    adjusted_r_squared_score = adjusted_r_squared(total_targets, total_predicts, num_samples, num_features)
    adjusted_r_absolute_score = adjusted_r_absolute(total_targets, total_predicts, num_samples, num_features)

    print("R-squared: {:.7f}, R-absolute: {:.7f}, adj R-squared: {:.7f}, adj R-absolute: {:.7f}".format(
        r_squared_score, r_absolute_score,
        adjusted_r_squared_score, adjusted_r_absolute_score
    ))


def eval_binary_classification_task(total_targets, total_predicts):

    auroc = roc_auc_score(total_targets, total_predicts)
    auprc = average_precision_score(total_targets, total_predicts)
    acc = accuracy_score(total_targets, np.array(total_predicts) >= 0.5)

    print("AUROC: {:.7f}, AUPRC: {:.7f}, Acc: {:.7f}".format(auroc, auprc, acc))

    return auroc, auprc, acc
