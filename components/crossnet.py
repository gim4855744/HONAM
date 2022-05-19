import torch

from torch.nn import Module, ModuleList
from torch.nn import Linear


class CrossNetComponent(Module):

    def __init__(self, in_units, orders):

        super(CrossNetComponent, self).__init__()

        num_layers = orders - 1
        self._layers = ModuleList([Linear(in_units, in_units, bias=False) for _ in range(num_layers)])

    def forward(self, features):

        x0 = features
        x = features
        interactions = [features]

        for layer in self._layers:
            x = layer(x * x0)
            interactions.append(x)

        interactions = torch.cat(interactions, dim=1)

        return interactions
