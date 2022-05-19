from torch.nn import Module, ModuleList
from torch.nn import Linear
from torch.nn import functional

from units.exu import ExU
from units.exp_dive import ExpDive

class FeatureNet(Module):

    def __init__(self):

        super(FeatureNet, self).__init__()

        self._units = [1, 32, 64, 32]
        num_layers = len(self._units) - 1

        self._layers = ModuleList([Linear(self._units[i], self._units[i + 1]) for i in range(num_layers)])

    def forward(self, feature):
        for layer in self._layers:
            feature = layer(feature)
            feature = functional.leaky_relu(feature)
        return feature

    def get_out_units(self):
        return self._units[-1]

class ExUNet(Module):

    def __init__(self):

        super(ExUNet, self).__init__()

        self._units = [1, 32, 64, 32]
        num_layers = len(self._units) - 1

        self._layers = ModuleList()
        for i in range(num_layers):
            if i == 0:
                self._layers.append(ExU(self._units[i], self._units[i + 1]))
            else:
                self._layers.append(Linear(self._units[i], self._units[i + 1]))

    def forward(self, feature):
        for layer in self._layers:
            feature = layer(feature)
            feature = functional.leaky_relu(feature)
        return feature

    def get_out_units(self):
        return self._units[-1]

class ExpDiveNet(Module):

    def __init__(self):

        super(ExpDiveNet, self).__init__()

        self._units = [1, 32, 64, 32]
        num_layers = len(self._units) - 1

        self._layers = ModuleList()
        for i in range(num_layers):
            if i == 0:
                self._layers.append(ExpDive(self._units[i], self._units[i + 1]))
            else:
                self._layers.append(Linear(self._units[i], self._units[i + 1]))

    def forward(self, feature):
        for layer in self._layers:
            feature = layer(feature)
            feature = functional.leaky_relu(feature)
        return feature

    def get_out_units(self):
        return self._units[-1]
