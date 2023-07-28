from typing import List

import torch
import torch.optim as optim
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import Tensor

__all__ = ['LightningModel']


class LightningModel(LightningModule):

    """Base for lightning models.

    Args:
        lr: learning rate.
        weight_decay: L2 penalty for trainable parameters.
    """

    def __init__(
        self,
        lr: float,
        weight_decay: float
    ) -> None:
        super().__init__()
        self._lr = lr
        self._weight_decay = weight_decay

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=self._lr, weight_decay=self._weight_decay)
    
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
