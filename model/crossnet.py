import torch

from torch.nn import Module
from torch.nn import Linear

from components.crossnet import CrossNetComponent


class CrossNet(Module):

    def __init__(self, num_features, out_size, **kwargs):

        super(CrossNet, self).__init__()

        emb_size = 32

        self._input_layer = Linear(num_features, emb_size)

        order = kwargs["order"]
        self._cross_net_component = CrossNetComponent(emb_size, order)

        self._output_layer = Linear(order * emb_size, out_size)

    def forward(self, features):

        features = self._input_layer(features)

        interactions = self._cross_net_component(features)

        outputs = self._output_layer(interactions)

        return outputs
