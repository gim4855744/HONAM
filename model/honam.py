import numpy as np
import torch

from torch.nn import Module, ModuleList, Linear, MSELoss, BCELoss
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam

from model import FeatureNet

class HONAM(Module):

    def __init__(
        self,
        num_features: int,
        out_size: int,
        task: str,
        order: int=2,
        batch_size: int=1024,
        lr: float=1e-3,
        epochs: int=1000,
        num_workers: int=0,
        verbose=True
    ):

        super(HONAM, self).__init__()

        self._task = task
        self._order = order
        self._batch_size = batch_size
        self._lr = lr
        self._epochs = epochs
        self._num_workers = num_workers
        self._verbose = verbose

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

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray=None, y_val:np.ndarray=None) -> None:

        device = next(self.parameters()).device
        with_val = (x_val is not None and y_val is not None)

        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)

        if with_val:
            x_val = torch.tensor(x_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.float32)
            val_dataset = TensorDataset(x_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=self._batch_size, num_workers=self._num_workers)

        optimizer = Adam(self.parameters(), lr=self._lr)
        if self._task == "regression":
            criterion = MSELoss()
        elif self._task == "binary_classification":
            criterion = BCELoss()

        for epoch in range(self._epochs):

            self.train()

            train_losses, train_step = 0, 0
            if with_val:
                val_losses, val_step = 0, 0

            for x, y in train_loader:

                x, y = x.to(device), y.to(device)

                prediction = self(x)
                loss = criterion(prediction, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses += loss.item()
                train_step += 1

            if with_val:

                with torch.no_grad():

                    self.eval()

                    for x, y in val_loader:

                        x, y = x.to(device), y.to(device)

                        prediction = self(x)
                        loss = criterion(prediction, y)

                        val_losses += loss.item()
                        val_step += 1

            if self._verbose:
                if with_val:
                    print("Epoch {}, Train Loss: {:.7f}, Val Loss: {:.7f}".format(epoch + 1, train_losses / train_step, val_losses / val_step))
                else:
                    print("Epoch {}, Train Loss: {:.7f}".format(epoch + 1, train_losses / train_step))

    def predict(self, x: np.ndarray) -> np.ndarray:

        device = next(self.parameters()).device
        x = torch.tensor(x, dtype=torch.float32, device=device)

        self.eval()
        prediction = self(x).detach().cpu().numpy()

        return prediction
