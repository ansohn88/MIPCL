from typing import Optional

import torch
import torch.nn.functional as F

from engine_base import BaseEngine
from losses import InfoNCE, SimMaxLoss, SimMinLoss
from model_mipcl import MIPCL


class EngineMIPCL(BaseEngine):
    def __init__(
            self,
            in_channels: int,
            intermediate_dim: int,
            n_classes: int,
            stain_info: bool,
            dropout: bool,
            alpha: Optional[float] = None,
            thresh: Optional[float] = 0.85,
            temperature: Optional[float] = 0.07,
            bag_weight: Optional[float] = None
    ) -> None:
        super().__init__(in_channels, intermediate_dim, n_classes, stain_info, dropout)

        # Init model
        self.model = MIPCL(
            in_channels=in_channels,
            intermediate_dim=intermediate_dim,
            n_classes=n_classes,
            stain_info=stain_info,
            dropout=dropout,
            thresh=thresh
        )

        # Loss functions
        self.alpha = alpha
        self.bag_weight = bag_weight

        if alpha is not None:
            self.criterion = [
                SimMaxLoss(alpha=alpha),
                SimMinLoss(),
                SimMaxLoss(alpha=alpha)
            ]
        else:
            # Use InfoNCE
            self.criterion = [
                InfoNCE(temperature=temperature,
                        negative_mode='unpaired'),
            ]

    def training_step(self, batch, batch_idx):
        x = batch['img']
        x_fname = batch['filename']
        y = batch['label']

        wsi_logits, fg, bg, topk_idx, _, _ = self.model(x, x_fname)
        total_loss = self.ccam_loss(
            wsi_logits=wsi_logits,
            foreground=fg,
            background=bg,
            label=y
        )
        if self.global_step % 10000 == 0:
            top_pos_k = topk_idx['pos_idx']
            top_neg_k = topk_idx['neg_idx']
            print(
                f"""
                total N, top (+), top (-): {x.size(0)}, {len(top_pos_k)}, {len(top_neg_k)}
                """
            )

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

        wsi_logits, fg, bg, _, _, _ = self.model(x, x_fname)
        total_loss = self.ccam_loss(
            wsi_logits=wsi_logits,
            foreground=fg,
            background=bg,
            label=y
        )

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

        wsi_logits, fg, bg, ids, A_raw, _ = self.model(
            x, x_fname)
        total_loss = self.ccam_loss(
            wsi_logits=wsi_logits,
            foreground=fg,
            background=bg,
            label=y
        )
        y_pred = torch.topk(wsi_logits, 1, dim=1)[1]
        y_pred = y_pred.squeeze(0)
        y_prob = F.softmax(wsi_logits, dim=1)

        return {
            'loss': total_loss,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'target': y,
            'top_ids': ids,
            'A_raw': A_raw,
            'filename': x_fname
        }

    def ccam_loss(self,
                  wsi_logits,
                  foreground,
                  background,
                  label):
        if self.alpha is not None:
            bg_bg = self.criterion[0](background)
            bg_fg = self.criterion[1](background, foreground)
            fg_fg = self.criterion[2](foreground)
            instance_loss = bg_bg + bg_fg + fg_fg
        else:
            instance_loss = self.criterion[0](
                query=foreground,
                positive_key=foreground,
                negative_keys=background
            )

        wsi_pred_loss = self.bag_loss_fn(wsi_logits, label)
        if self.bag_weight is not None:
            total_loss = (self.bag_weight * wsi_pred_loss) + \
                ((1 - self.bag_weight) * instance_loss)
        else:
            total_loss = instance_loss + wsi_pred_loss

        total_loss = total_loss.unsqueeze_(0)
        return total_loss
