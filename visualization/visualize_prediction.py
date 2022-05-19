import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import torch
import os

from torch.utils.data.dataloader import DataLoader
from torch.backends import cudnn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from matplotlib import colors
from matplotlib.ticker import StrMethodFormatter

from absl import app, flags

from data.load_data import california_housing_prices, insurance, house_prices, fico, credit, mimic, compas
from data.dataset import TensorDataset

from model.lr import LR
from model.crossnet import CrossNet
from model.mlp import MLP
from model.nam import NAM
from model.namcross import NAMWithCrossNet
from model.honam import HONAM

cudnn.deterministic = True
cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 1, "")
flags.DEFINE_string("dataset", "compas", "")
flags.DEFINE_string("model", "honam", "")
flags.DEFINE_integer("order", 2, "the order of feature interactions")

def plot_local_first_order(model, data_sample, feature_names):

    plt.tight_layout()
    plt.rcParams['axes.axisbelow'] = True
    plt.grid(True, axis="y", alpha=0.5, linestyle="--")

    feature, _ = data_sample

    interactions = model.local_first_order(feature.view(1, -1))

    plt.bar(feature_names, interactions, color="tab:blue")
    plt.xticks(rotation=90)
    plt.savefig("local_{}_first.pdf".format(FLAGS.dataset), bbox_inches='tight')
    plt.show()

def plot_local_second_order(model, data_sample, feature_names):

    feature, _ = data_sample

    interactions = model.local_second_order(feature.view(1, -1))

    cmap = plt.get_cmap("bwr")
    sns.heatmap(interactions, center=0, cmap=cmap)
    plt.xticks(np.arange(0.5, len(feature_names), 1), feature_names, rotation=90)
    plt.yticks(np.arange(0.5, len(feature_names), 1), feature_names, rotation=0)
    plt.savefig("local_{}_second.pdf".format(FLAGS.dataset), bbox_inches='tight')
    plt.show()

def plot_axvspan(dataset, idx, bins):

    loader = DataLoader(dataset, batch_size=len(dataset))

    features, _ = loader.__iter__().__next__()
    features = features.detach().cpu().numpy()[:, idx]

    freqs, ranges = np.histogram(features, bins=bins)
    freqs = freqs / max(freqs)

    for i in range(bins):
        plt.axvspan(ranges[i], ranges[i + 1], alpha=freqs[i], color="indianred")


def plot_global_first_order(model, feature_names, idx):

    plt.tight_layout()
    plt.grid(True, axis="y", alpha=0.5, linestyle="--")
    plt.rcParams['axes.axisbelow'] = True
    plt.rc('font', size=20)

    min_v = 0
    max_v = 1
    step = 1000

    features = torch.zeros((step, len(feature_names)), device=device)
    features[:, idx] = torch.linspace(min_v, max_v, step, device=device)

    interactions = model.global_first_order(features, idx)

    xs = features[:, idx].detach().cpu().numpy()
    ys = interactions

    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.plot(xs, ys, color="tab:blue", linewidth=2)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(feature_names[idx], fontsize=20)
    plt.savefig("global_{}_{}.pdf".format(FLAGS.dataset,
                                          feature_names[idx]), bbox_inches='tight')
    plt.show()

def plot_axvspan_for_categorical(dataset, idx_range):

    loader = DataLoader(dataset, batch_size=len(dataset))

    features, _ = loader.__iter__().__next__()
    features = features.detach().cpu().numpy()[:, idx_range]

    freqs = []
    for i in range(len(idx_range)):
        freqs.append(features[:, i].sum())
    freqs = np.array(freqs) / max(freqs)

    for i in range(len(idx_range)):
        plt.axvspan(i, i + 1, alpha=freqs[i], color="indianred")

def plot_global_first_order_for_categorical(model, feature_names, idx_range):

    plt.tight_layout()
    plt.grid(True, axis="y", alpha=0.5, linestyle="--")
    plt.rcParams['axes.axisbelow'] = True
    plt.rc('font', size=20)

    field_name = feature_names[idx_range[0]].split('_')[0]
    feature_names = feature_names.map(lambda x: x.split('_')[-1])

    step = len(idx_range)

    ys = []

    features = torch.zeros((step, len(feature_names)), device=device)
    for i, idx in enumerate(idx_range):
        features[i, idx] = 1
    for i, idx in enumerate(idx_range):
        interactions = model.global_first_order(features[i].view(1, -1), idx)
        ys.extend(interactions)

    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.stairs(ys, baseline=None, color="tab:blue", linewidth=2)
    plt.xticks(np.arange(0.5, len(idx_range), 1), labels=feature_names[idx_range], rotation=90, fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(field_name, fontsize=20)
    plt.savefig("global_{}_{}.pdf".format(FLAGS.dataset, field_name), bbox_inches='tight')
    plt.show()

def plot_global_second_order(model, feature_names, idx1, idx2):

    start_v = 0
    end_v = 1
    step = 100

    feature1 = torch.linspace(start_v, end_v, step, device=device)
    feature2 = torch.linspace(start_v, end_v, step, device=device)

    features = torch.zeros((step * step, len(feature_names)), device=device)
    for i1, i2 in enumerate(range(0, step * step, step)):
        features[i2: i2 + step, idx1] = feature1[i1]
        features[i2: i2 + step, idx2] = feature2

    interactions = model.global_second_order(features, idx1, idx2)

    xs = features.detach().cpu().numpy()[:, idx1]
    ys = features.detach().cpu().numpy()[:, idx2]
    zs = interactions

    divnorm = colors.TwoSlopeNorm(vcenter=0.)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    cmap = plt.get_cmap("bwr")

    ax.plot_trisurf(xs, ys, zs, cmap=cmap, norm=divnorm)
    ax.view_init(elev=30)
    ax.set_xlabel(feature_names[idx1])
    ax.set_ylabel(feature_names[idx2])

    fig.savefig("global_{}_{}_{}.pdf".format(FLAGS.dataset,
                                             feature_names[idx1],
                                             feature_names[idx2]), bbox_inches="tight")
    fig.show()

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

    # get numerical feature columns
    feature_names = data.drop("target", axis=1).columns
    numeric_feature_names = [name for name in feature_names if data[name].dtype != "object"]  # only numerical features

    # make dummy features
    data = pd.get_dummies(data)
    feature_names = data.columns
    feature_names = feature_names.drop("target")

    # set data info
    num_samples = data.shape[0]
    num_features = data.shape[1] - 1
    out_size = 1

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

    # torch data module
    train_dataset = TensorDataset(train_data, device)

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

    save_dir = "./save/{}/{}/".format(dataset, model.__class__.__name__)
    save_path = os.path.join(save_dir, "order{}_seed{}.ckpt".format(order, seed))

    model.load_state_dict(torch.load(save_path))

    model.eval()

    # plot_local_first_order(model, train_dataset.__getitem__(0), feature_names)

    # plot_local_second_order(model, train_dataset.__getitem__(0), feature_names)

    # idx_range = [4, 5, 6, 7, 8, 9]
    # plot_axvspan_for_categorical(train_dataset, idx_range)
    # plot_global_first_order_for_categorical(model, feature_names, idx_range)

    idx = 0
    plot_axvspan(train_dataset, idx, bins=25)
    plot_global_first_order(model, feature_names, idx)

    # plot_global_second_order(model, feature_names, 0, 20)

if __name__ == '__main__':
    app.run(main)
