import torch

from torch.nn import Module

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
