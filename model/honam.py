import numpy as np
import torch

from torch.nn import Module, ModuleList, Linear, MSELoss, BCELoss
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam

from model import FeatureNet

class HONAM(Module):

    def __init__(self, num_features: int, out_size: int, task: str, order: int=2):

        super(HONAM, self).__init__()

        self._task = task
        self._order = order

        num_units = [1, 32, 64, 32]
        self._feature_nets = ModuleList([FeatureNet(num_units) for _ in range(num_features)])

        self._output_layer = Linear(order * num_units[-1], out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # run feature networks
        transformed_x = []
        for i, feature_net in enumerate(self._feature_nets):
            transformed_xi = feature_net(x[:, i].view(-1, 1))
            transformed_x.append(transformed_xi)
        transformed_x = torch.stack(transformed_x, dim=1)

        # model interactions
        powers = [1, transformed_x.sum(dim=1)]
        interactions = [1, transformed_x.sum(dim=1)]
        for i in range(2, self._order + 1):
            powers.append(transformed_x.pow(i).sum(dim=1))
            curr_interaction = 0
            for j in range(1, i + 1):
                curr_interaction += pow(-1, j + 1) * powers[j] * interactions[i - j]
            interactions.append(curr_interaction / i)
        interactions = torch.concat(interactions[1:], dim=1)

        prediction = self._output_layer(interactions)
        if self._task == "binary_classification":
            prediction = torch.sigmoid(prediction)

        return prediction

    def fit(
            self,
            x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray=None, y_val:np.ndarray=None,
            batch_size: int=1024, lr: float=1e-3, epochs: int=1000, num_workers: int=0, verbose=True
    ):

        device = next(self.parameters()).device
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)

        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        optimizer = Adam(self.parameters(), lr=lr)
        if self._task == "regression":
            criterion = MSELoss()
        elif self._task == "binary_classification":
            criterion = BCELoss()

        total_losses, train_step = 0, 0

        self.train()

        for epoch in range(epochs):

            for x, y in train_loader:

                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                prediction = self(x)
                loss = criterion(prediction, y)
                loss.backward()
                optimizer.step()

                total_losses += loss.item()
                train_step += 1

            if verbose:
                print(epoch, total_losses / train_step)

    def predict(self, x: np.ndarray) -> np.ndarray:

        device = next(self.parameters()).device
        x = torch.tensor(x, dtype=torch.float32, device=device)

        self.eval()
        prediction = self(x).detach().cpu().numpy()

        return prediction
