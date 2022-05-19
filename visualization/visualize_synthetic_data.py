import matplotlib.pyplot as plt
import numpy as np
import random
import torch

from torch.nn import Module, ModuleList
from torch.nn import Linear
from torch.nn import functional
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
from torch.backends import cudnn

from absl import app, flags

cudnn.deterministic = True
cudnn.benchmark = False

FLAGS = flags.FLAGS
flags.DEFINE_string("task", "classification", "regression or classification")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# plt.style.use("seaborn-whitegrid")

class LinearCentered(Module):

    def __init__(self, in_features, out_features):

        super(LinearCentered, self).__init__()

        self._weight = torch.nn.Parameter(torch.empty(in_features, out_features))
        self._bias = torch.nn.Parameter(torch.empty(in_features))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self._weight, std=0.5)
        torch.nn.init.normal_(self._bias, std=0.5)

    def forward(self, features):
        return (features - self._bias) @ self._weight

class ExU(Module):

    def __init__(self, in_features, out_features):

        super(ExU, self).__init__()

        self._weight = torch.nn.Parameter(torch.empty(in_features, out_features))
        self._bias = torch.nn.Parameter(torch.empty(in_features))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self._weight, std=0.5)
        torch.nn.init.normal_(self._bias, std=0.5)

    def forward(self, features):
        return (features - self._bias) @ torch.exp(self._weight)

class ExpDive(Module):

    def __init__(self, in_features, out_features):

        super(ExpDive, self).__init__()

        self._weight = torch.nn.Parameter(torch.empty(in_features, out_features))
        self._bias = torch.nn.Parameter(torch.empty(in_features))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self._weight, std=0.5)
        torch.nn.init.normal_(self._bias, std=0.5)

    def forward(self, features):
        return (features - self._bias) @ (torch.exp(self._weight) - torch.exp(-self._weight))

class FeatureNet(Module):

    def __init__(self, in_unit):

        super(FeatureNet, self).__init__()

        self._units = [1, 32, 64, 32]

        if in_unit == "linear":
            self._in_layer = Linear(self._units[0], self._units[1])
        elif in_unit == "exu":
            self._in_layer = ExU(self._units[0], self._units[1])
        elif in_unit == "exp_dive":
            self._in_layer = ExpDive(self._units[0], self._units[1])
        else:
            raise Exception

        self._layers = ModuleList([Linear(self._units[i], self._units[i + 1]) for i in range(1, len(self._units) - 1)])

        self._out_layer = Linear(self._units[-1], 1, bias=True)

    def forward(self, feature):

        feature = self._in_layer(feature)
        feature = torch.clip(feature, 0, 1)

        for layer in self._layers:
            feature = layer(feature)
            feature = functional.leaky_relu(feature)

        feature = self._out_layer(feature)

        return feature

class MTLFeatureNet(Module):

    def __init__(self, in_unit, num_nets):

        super(MTLFeatureNet, self).__init__()

        self._nets = ModuleList([FeatureNet(in_unit) for _ in range(num_nets)])
        self._out_layer = Linear(1024 * 100, 1)
        self._bias = torch.nn.Parameter(torch.empty((1,)))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.zeros_(self._bias)

    def forward(self, feature):

        outputs = [net(feature) for net in self._nets]
        outputs = torch.cat(outputs, dim=1).sum(dim=1, keepdim=True)
        outputs = outputs + self._bias

        return outputs

def get_synthetic_regression():

    x = np.random.uniform(-1, 1, (100,))
    y = np.random.uniform(-1, 1, (100,))

    return x, y

def get_synthetic_classification():

    x, y = [], []

    for i in range(100):
        p = np.random.randint(1, 9) / 10.  # probability
        bernoulli = np.random.binomial(1, p, 100).astype(np.float32)

        x.extend(np.full(100, (np.random.random() - 0.5) * 2))
        y.extend(bernoulli)

    return x, y

def plot_synthetic_data(x, y, p, task, in_unit):

    plt.rc('font', size=15)

    plt.scatter(x, y, color="tab:blue", label="target")

    xp_zip = sorted(zip(x, p))
    x, p = zip(*xp_zip)
    plt.plot(x, p, color="tab:red", label="prediction")

    plt.xlabel("x")
    if task == "regression":
        plt.ylabel("y")
    elif task == "classification":
        plt.ylabel("log odd")

    plt.legend(facecolor="white", edgecolor="black", loc="upper right", framealpha=1)

    plt.savefig("stl-{}-{}.pdf".format(task, in_unit))
    plt.show()

def main(argv):

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    task = FLAGS.task

    in_unit = "exp_dive"

    if task == "regression":
        x, y =get_synthetic_regression()
        criterion = torch.nn.MSELoss()
    elif task == "classification":
        x, y = get_synthetic_classification()
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        raise Exception

    x = torch.tensor(x, dtype=torch.float32, device=device).view(-1, 1)
    y = torch.tensor(y, dtype=torch.float32, device=device).view(-1, 1)

    batch_size = len(x) // 10
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # model = MTLFeatureNet(in_unit, 100).to(device)
    model = FeatureNet(in_unit).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1000):

        train_loss, train_step = 0, 0

        for batch_x, batch_y in dataloader:

            predict = model(batch_x)
            loss = criterion(predict, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_step += 1

        print(epoch + 1, train_loss / train_step)

    p = torch.sigmoid(model(x)).detach().cpu().numpy().reshape(-1, 1)
    x = x.detach().cpu().numpy().reshape(-1, 1)
    y = y.detach().cpu().numpy().reshape(-1, 1)

    if task == "regression":
        plot_synthetic_data(x, y, p, task, in_unit)

    elif task == "classification":

        new_x, new_y, new_p = [], [], []

        for i in range(0, len(x), 100):

            target_prob = np.sum(y[i: i + 100]) / 100
            predict_prob = p[i]

            target_odd = target_prob / (1 - target_prob)
            target_log_odd = np.log(target_odd)

            predict_odd = predict_prob / (1 - predict_prob)
            predict_log_odd = np.log(predict_odd)

            new_x.append(x[i])
            new_y.append(target_log_odd)
            new_p.append(predict_log_odd)

        plot_synthetic_data(new_x, new_y, new_p, task, in_unit)

    else:
        raise Exception

if __name__ == '__main__':
    app.run(main)
