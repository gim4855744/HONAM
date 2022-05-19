from torch.nn import Module
from torch.nn import Linear


class LR(Module):

    def __init__(self, num_features, out_size, **kwargs):

        super(LR, self).__init__()

        self._output_layer = Linear(num_features, out_size)

    def forward(self, features):
        outputs = self._output_layer(features)
        return outputs
