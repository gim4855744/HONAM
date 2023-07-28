import torch
import torch.nn as nn
from torch import Tensor

from honam.models._base import LightningModel

__all__ = ['HONAM']


class HONAM(LightningModel):

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        hidden_dims: list,
        order: int = 2,
        lr: float = 1e-3,
        weight_decay: float = 0,
        batch_norm: bool = False,
        dropout: float = 0
    ):

        super().__init__(lr, weight_decay)

        self.save_hyperparameters()

        layers = []
        curr_dim = 1

        # construct feature nets
        for next_dim in hidden_dims:
            layers.append(
                nn.Conv1d(
                    in_channels=curr_dim * n_features,
                    out_channels=next_dim * n_features,
                    kernel_size=1,
                    groups=n_features
                )
            )
            if batch_norm:
                layers.append(nn.BatchNorm1d(num_features=next_dim * n_features))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.LeakyReLU())
            curr_dim = next_dim

        layers.append(
            nn.Conv1d(
                in_channels=curr_dim * n_features,
                out_channels=curr_dim * n_features,
                kernel_size=1,
                groups=n_features
            )  # to dense vector for interaction modeling
        )
        layers.append(nn.Flatten())

        self._featurenet = nn.Sequential(*layers)
        self._output_layer = nn.Linear(in_features=order * curr_dim, out_features=n_outputs)
        self._n_features = n_features
        self._order = order

    def forward(
        self,
        x: Tensor
    ) -> Tensor:
        
        x = x.unsqueeze(dim=-1)
        
        x = self._featurenet(x)

        feature_size = x.shape[-1] // self._n_features
        x = torch.split(x, feature_size, dim=1)
        x = torch.stack(x, dim=1)

        # interaction modeling
        powers = [1, x.sum(dim=1)]
        interactions = [1, x.sum(dim=1)]
        for i in range(2, self._order + 1):
            powers.append(x.pow(i).sum(dim=1))
            interaction = 0
            for j in range(1, i + 1):
                interaction += pow(-1, j + 1) * powers[j] * interactions[i - j]
            interactions.append(interaction / i)
        interactions = torch.concat(interactions[1:], dim=1)
        
        out = self._output_layer(interactions)
        
        return out
