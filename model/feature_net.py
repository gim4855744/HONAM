import torch

from torch.nn import Module, ModuleList, Linear
from torch.nn import functional
from typing import List

class FeatureNet(Module):

    def __init__(self, num_units: List[int], dropout: float=0.):

        super(FeatureNet, self).__init__()

        self._dropout = dropout
        self._num_layers = len(num_units) - 1
        self._layers = ModuleList([Linear(num_units[i], num_units[i + 1]) for i in range(self._num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self._layers):
            x = layer(x)
            if self._num_layers > i + 1:
                x = functional.relu(x)
                x = functional.dropout(x, self._dropout, self.training)
        return x
