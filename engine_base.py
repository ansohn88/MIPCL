from abc import ABC, abstractmethod
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import (AUROC, AveragePrecision, ConfusionMatrix, F1Score,
                          Precision, Recall)


class BaseEngine(pl.LightningModule, ABC):

    @abstractmethod
    def __init__(
        self,
        in_channels: int,
        intermediate_dim: int,
        n_classes: int,
        stain_info: bool,
        dropout: bool,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.intermediate_dim = intermediate_dim
        self.n_classes = n_classes
        self.stain_info = stain_info
        self.dropout = dropout

        # Bag loss function
        self.bag_loss_fn = nn.CrossEntropyLoss()

        # Metrics
        self.recall = Recall(num_classes=self.n_classes,
                             average="weighted")
        self.prec = Precision(num_classes=self.n_classes,
                              average="weighted")
        self.auprc = AveragePrecision(average="weighted",
                                      num_classes=self.n_classes)
        self.auroc = AUROC(average="weighted",
                           num_classes=self.n_classes, pos_label=None)
        self.f1_score = F1Score(average="weighted",
                                num_classes=self.n_classes)
        self.confmat = ConfusionMatrix(num_classes=self.n_classes)

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    def training_epoch_end(self, outputs) -> None:
        avg_loss = torch.stack(
            [x['loss'] for x in outputs]
        )
        # print(avg_loss.shape)
        avg_loss = avg_loss[~torch.any(avg_loss.isnan(), dim=1)]
        avg_loss = avg_loss.mean()

        self.log("avg_train_loss",
                 avg_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 batch_size=1
                 )

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        pass

    @torch.no_grad()
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(
            [x['loss'] for x in outputs]
        )
        if avg_loss.dim() == 1:
            avg_loss = avg_loss.unsqueeze_(1)
        # print(avg_loss.shape, avg_loss.dim())
        avg_loss = avg_loss[~torch.any(avg_loss.isnan(), dim=1)]
        avg_loss = avg_loss.mean()

        self.log("avg_val_loss",
                 avg_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 batch_size=1
                 )

    @abstractmethod
    def test_step(self, batch, batch_idx):
        pass

    @torch.no_grad()
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack(
            [x['loss'] for x in outputs]
        )
        if avg_loss.dim() == 1:
            avg_loss = avg_loss.unsqueeze_(1)
        avg_loss = avg_loss[~torch.any(avg_loss.isnan(), dim=1)]
        avg_loss = avg_loss.mean()

        test_preds = []
        test_targets = []
        test_probs = []
        fnames_topk_ids = dict()

        for x in outputs:
            pred = x['y_pred']
            target = x['target']
            probs = x['y_prob']
            attn_raw = x['A_raw']
            top_ids = x['top_ids']
            fname = x['filename'][0]

            fnames_topk_ids[f'{fname}'] = {
                'A_raw': attn_raw,
                'top_ids': top_ids,
                'pred': pred,
                'target': target,
                'probs': probs,
            }

            test_preds.append(pred)
            test_targets.append(target)
            test_probs.append(probs)

        test_preds = torch.cat(test_preds, dim=0)
        test_targets = torch.cat(test_targets, dim=0)
        test_probs = torch.cat(test_probs, dim=0)

        # METRICS
        recall = self.recall(test_preds.long(), test_targets.long()).cpu()
        precision = self.prec(
            test_preds.long(), test_targets.long()).cpu()

        auprc = self.auprc(test_probs, test_targets.long()).cpu()
        f1 = self.f1_score(test_preds.long(), test_targets.long()).cpu()
        auroc = self.auroc(test_probs, test_targets.long()).cpu()
        confmat = self.confmat(test_preds.long(), test_targets.long()).cpu()

        self.log("avg_test_loss",
                 avg_loss,
                 batch_size=1)
        self.log("avg_test_recall",
                 recall,
                 batch_size=1)
        self.log("avg_test_precision",
                 precision,
                 batch_size=1)
        self.log("AUPRC",
                 auprc,
                 batch_size=1)
        self.log("avg_f1",
                 f1,
                 batch_size=1)
        self.log("AUROC",
                 auroc,
                 batch_size=1)

        self.test_results = {
            'test_loss': avg_loss,
            'predictions': test_preds,
            'probabilities': test_probs,
            'targets': test_targets,
            'test_recall': recall,
            'test_precision': precision,
            'auprc': auprc,
            'test_f1_score': f1,
            'confmat': confmat,
            'auroc': auroc,
            'fnames_topk_ids': fnames_topk_ids
        }

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(),
            lr=3e-4,
            betas=(0.9, 0.95),
            weight_decay=1e-4
        )
        schedule = torch.optim.lr_scheduler.OneCycleLR(
            optim,
            max_lr=3e-4,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.2,
            anneal_strategy='cos',
            cycle_momentum=True,
        )
        return {
            'optimizer': optim,
            'lr_scheduler': {
                'scheduler': schedule,
                'interval': 'step'
            }
        }
