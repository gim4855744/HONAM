import torch

from torch.utils.data.dataset import Dataset


class TensorDataset(Dataset):

    def __init__(self, data, device):

        features = data.drop("target", axis=1).values
        targets = data["target"].values.reshape(-1, 1)

        self._num_data = len(targets)
        self._features = torch.tensor(features, dtype=torch.float32, device=device)
        self._targets = torch.tensor(targets, dtype=torch.float32, device=device)

    def __getitem__(self, index):
        return self._features[index], self._targets[index]

    def __len__(self):
        return self._num_data
