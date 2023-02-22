from typing import Optional

import torch
import torch.nn.functional as F

from engine_base import BaseEngine
from model_abmil import ABMIL


class EngineABMIL(BaseEngine):
    def __init__(
            self,
            in_channels: int,
            intermediate_dim: int,
            n_classes: int, stain_info: bool,
            dropout: bool,
    ) -> None:
        super().__init__(in_channels, intermediate_dim,
                         n_classes, stain_info, dropout)

        # Init model
        self.model = ABMIL(
            in_channels=in_channels,
            intermediate_dim=intermediate_dim,
            n_classes=n_classes,
            stain_info=stain_info,
            dropout=dropout
        )

    def training_step(self, batch, batch_idx):
        x = batch['img']
        x_fname = batch['filename']
        y = batch['label']

        wsi_logits, _, _ = self.model(x, x_fname)
        total_loss = self.abmil_loss(wsi_logits, y)

        self.log("train_loss",
                 total_loss,
                 on_step=False,
                 on_epoch=True,
                 #  prog_bar=True,
                 logger=True,
                 batch_size=1
                 )

        return {
            "loss": total_loss,
        }

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x = batch['img']
        x_fname = batch['filename']
        y = batch['label']

        wsi_logits, _, _ = self.model(x, x_fname)
        total_loss = self.abmil_loss(wsi_logits, y)

        self.log("val_loss",
                 total_loss,
                 on_step=False,
                 on_epoch=True,
                 #  prog_bar=True,
                 logger=True,
                 batch_size=1)

        return {
            'loss': total_loss,
        }

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        x = batch['img']
        x_fname = batch['filename']
        y = batch['label']

        wsi_logits, A_raw, top_ids = self.model(x, x_fname)
        total_loss = self.abmil_loss(wsi_logits, y)

        y_pred = torch.topk(wsi_logits, 1, dim=1)[1]
        y_pred = y_pred.squeeze(0)
        y_prob = F.softmax(wsi_logits, dim=1)

        return {
            'loss': total_loss,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'target': y,
            'top_ids': top_ids,
            'A_raw': A_raw,
            'filename': x_fname
        }

    def abmil_loss(self, logits, label):
        total_loss = self.bag_loss_fn(logits, label)
        total_loss = total_loss.unsqueeze_(0)
        return total_loss
