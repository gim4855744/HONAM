import torch

from torch.nn import Module
from torch.nn import Linear

from components.nam import NAMComponent


class NAM(Module):

    def __init__(self, num_features, out_size, **kwargs):

        super(NAM, self).__init__()

        self._nam_component = NAMComponent(num_features)
        out_units = self._nam_component.get_out_units()
        self._output_layer = Linear(num_features * out_units, out_size)  # see FeatureNet

    def forward(self, features):

        features = self._nam_component(features)
        features = torch.cat(features, dim=1)

        outputs = self._output_layer(features)

        return outputs
