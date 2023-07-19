from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import Tensor

__all__ = ['HONAM']


class HONAM(LightningModule):

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

        super().__init__()

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
        self._lr = lr
        self._weight_decay = weight_decay

    def forward(
        self,
        x: Tensor
    ) -> Tensor:
        
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
    
    def training_step(
        self,
        batch: List[Tensor],
        batch_idx: int
    ) -> Tensor:
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(
        self,
        batch: List[Tensor],
        batch_idx: int
    ) -> Tensor:
        with torch.no_grad():
            x, y = batch
            y_hat = self(x)
            loss = F.mse_loss(y_hat, y)
            self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
            return loss
    
    def test_step(self):
        pass
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self._lr, weight_decay=self._weight_decay)
