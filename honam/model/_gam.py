import numpy as np

import torch

from torch.nn import Module, ModuleList, Linear, LeakyReLU
from torch.nn.init import uniform_
from ._base import PyTorchModel

__all__ = ['HONAM']


class FeatureNet(Module):

    def __init__(self):
        super().__init__()
        self._layers = ModuleList([
            Linear(1, 32),
            LeakyReLU(),
            Linear(32, 64),
            LeakyReLU(),
            Linear(64, 32),
            LeakyReLU()
        ])
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self._layers:
            if layer.__class__.__name__ == 'Linear':
                uniform_(layer.weight, a=-1e-4, b=1e-4)

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return x


class HONAM(PyTorchModel):

    def __init__(self, n_features, order, task, ckpt_path):
        super().__init__(task, ckpt_path)
        self._order = order
        self._emb_size = 32  # size of feature vector
        self._feature_nets = ModuleList([FeatureNet() for _ in range(n_features)])
        self._output_layer = Linear(order * self._emb_size, 1)

    def forward(self, x):
        
        x = x.T.unsqueeze(dim=2)
        x = [feature_net(xi) for feature_net, xi in zip(self._feature_nets, x)]
        x = torch.stack(x, dim=1)

        powers = [1, x.sum(dim=1)]
        interactions = [1, x.sum(dim=1)]

        for i in range(2, self._order + 1):

            curr_power = x.pow(i).sum(dim=1)
            powers.append(curr_power)

            curr_interaction = 0
            for j in range(1, i + 1):
                curr_interaction += pow(-1, j + 1) * powers[j] * interactions[i - j]

            interactions.append(curr_interaction / i)

        interactions = torch.concat(interactions[1:], dim=1)

        x = self._output_layer(interactions)

        return x
    
    def interpret(self, x, *args):

        """Get feature contributions from data samples.

        Input
        --------------------
        - x: Batch of data samples. Shape (N, n_features).
        - *args: feature ids.

        Return
        --------------------
        - contribution of the feature specified by *args.

        Example
        --------------------
        # get the contribution of 1st feature from x
        > model.interpret(x, 0)

        # get the contribution of 2nd-order feature from x
        > model.interpret(x, 1, 3)  # contribution of cross feature (x1 * x3).
        """

        assert self._order >= len(args)

        self.eval()

        device = next(self.parameters()).device

        x = torch.tensor(x, dtype=torch.float32, device=device)
        x = x.T.unsqueeze(dim=2)
        x = [feature_net(xi) for feature_net, xi in zip(self._feature_nets, x)]
        x = torch.stack(x, dim=1)

        x = x[:, args]
        x = x.prod(dim=1)
        weight = self._output_layer.weight.T[(len(args) - 1) * self._emb_size: len(args) * self._emb_size]

        contributions = x @ weight

        return contributions.detach().cpu().numpy()
    
    def get_interactions(self, x):

        """Get all feature contributions.

        Input
        --------------------
        - x: Batch of data samples. Shape (N, n_features).

        Return
        --------------------
        - contributions of features.
        - feature names
        """

        n_features = x.shape[1]
        contributions = []
        names = []

        def _get_interactions(start_idx, items=[]):

            if len(items) > self._order:
                return
            
            contributions.append(self.interpret(x, *items))
            names.append(items)

            for i in range(start_idx, n_features):
                _get_interactions(i + 1, items + [i])

        for i in range(n_features):
            _get_interactions(start_idx=i + 1, items=[i])

        contributions = np.concatenate(contributions, axis=1)
        
        return contributions, names
