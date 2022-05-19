import torch

from torch.nn import Module
from torch.nn import Linear

from components.nam import NAMComponent
from components.crossnet import CrossNetComponent


class NAMWithCrossNet(Module):

    def __init__(self, num_features, out_size, **kwargs):

        super(NAMWithCrossNet, self).__init__()

        self._nam_component = NAMComponent(num_features)
        out_units = self._nam_component.get_out_units()

        order = kwargs["order"]
        self._cross_net_component = CrossNetComponent(out_units, order)

        self._output_layer = Linear(order * out_units, out_size)  # see FeatureNet

    def forward(self, features):

        features = self._nam_component(features)
        features = torch.stack(features, dim=1).sum(dim=1)

        interactions = self._cross_net_component(features)

        outputs = self._output_layer(interactions)

        return outputs
