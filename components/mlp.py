from torch.nn import Module, ModuleList
from torch.nn import Linear
from torch.nn import functional


class MLPComponent(Module):

    def __init__(self, in_units):

        super(MLPComponent, self).__init__()

        self._units = [in_units, 32, 64, 32]
        num_layers = len(self._units) - 1

        self._layers = ModuleList([Linear(self._units[i], self._units[i + 1]) for i in range(num_layers)])

    def forward(self, feature):
        for layer in self._layers:
            feature = layer(feature)
            feature = functional.leaky_relu(feature)
        return feature

    def get_out_units(self):
        return self._units[-1]