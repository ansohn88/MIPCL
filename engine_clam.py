from typing import Optional

import torch
import torch.nn.functional as F

from engine_base import BaseEngine
from model_clam import CLAM


class EngineCLAM(BaseEngine):

    def __init__(
            self,
            in_channels: int,
            intermediate_dim: int,
            n_classes: int,
            stain_info: bool,
            dropout: bool,
            k_sample: int,
            inst_loss: Optional[str] = 'svm',
            bag_weight: Optional[float] = None
    ) -> None:
        super().__init__(in_channels, intermediate_dim, n_classes, stain_info, dropout)

        # Init model
        self.model = CLAM(
            in_channels=in_channels,
            intermediate_dim=intermediate_dim,
            dropout=dropout,
            stain_info=stain_info,
            n_classes=n_classes,
            inst_loss_type=inst_loss,
            k_sample=k_sample
        )
        self.bag_weight = bag_weight

    def training_step(self, batch, batch_idx):
        x = batch['img']
        x_fname = batch['filename']
        y = batch['label']

        logits, _, _, inst_loss, _, _ = self.model(x,
                                                   fname=x_fname,
                                                   label=y,
                                                   #    instance_eval=True
                                                   )
        total_loss = self.clam_loss(
            logits=logits,
            y=y,
            inst_loss=inst_loss
        )

        self.log("train_loss",
                 total_loss,
                 on_step=False,
                 on_epoch=True,
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

        logits, _, _, inst_loss, _, _ = self.model(x,
                                                   fname=x_fname,
                                                   label=y,
                                                   #    instance_eval=True
                                                   )
        total_loss = self.clam_loss(
            logits=logits,
            y=y,
            inst_loss=inst_loss
        )

        self.log("val_loss",
                 total_loss,
                 on_step=True,
                 on_epoch=False,
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

        logits, y_prob, y_pred, inst_loss, A_raw, top_ids = self.model(x,
                                                                       fname=x_fname,
                                                                       label=y,
                                                                       #    instance_eval=True
                                                                       )
        total_loss = self.clam_loss(
            logits=logits,
            y=y,
            inst_loss=inst_loss
        )

        return {
            'loss': total_loss,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'target': y,
            'top_ids': top_ids,
            'A_raw': A_raw,
            'filename': x_fname
        }

    def clam_loss(self,
                  logits,
                  y,
                  inst_loss,
                  ):
        bag_loss = self.bag_loss_fn(logits, y)
        total_loss = (self.bag_weight * bag_loss) + \
            ((1 - self.bag_weight) * inst_loss)

        total_loss = total_loss.unsqueeze_(0)
        return total_loss
