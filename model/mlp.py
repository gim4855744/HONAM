from torch.nn import Module
from torch.nn import Linear

from components.mlp import MLPComponent


class MLP(Module):

    def __init__(self, num_features, out_size, **kwargs):

        super(MLP, self).__init__()

        self._mlp_component = MLPComponent(num_features)
        out_units = self._mlp_component.get_out_units()

        self._output_layer = Linear(out_units, out_size)

    def forward(self, features):
        features = self._mlp_component(features)
        outputs = self._output_layer(features)
        return outputs
