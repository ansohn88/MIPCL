from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from model_base import BaseModel
from modules import FinalClassifier


class MIPCL(BaseModel):
    def __init__(
        self,
        in_channels: int,
        intermediate_dim: int,
        n_classes: int,
        stain_info: bool,
        dropout: bool = True,
        thresh: Optional[float] = 0.85
    ) -> None:
        super().__init__()

        # Mini-encoder for pretrained features
        if stain_info:
            in_channels += 2

        if dropout:
            drp_val = 0.2
        else:
            drp_val = None

        self.embedder = nn.Sequential(
            nn.Linear(in_channels, intermediate_dim),
            nn.GroupNorm(num_groups=int(intermediate_dim/16),
                         num_channels=intermediate_dim),
            nn.Dropout(drp_val),
            nn.Mish()
        )

        # Attention mechanism
        self.disentangler = nn.Sequential(
            nn.Linear(intermediate_dim, out_features=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=1),
            nn.Softmax(dim=0)
        )

        # For CAM maps probabilities
        # intermediate_dim = int(intermediate_dim * 2)
        # self.fg_1d_clf = FinalClassifier(
        #     in_channels=intermediate_dim,
        #     num_classes=num_classes,
        #     dropout_rate=dropout
        # )

        # Classifier for final bag logits
        self.final_classifier = FinalClassifier(
            in_channels=intermediate_dim,
            num_classes=n_classes,
            dropout_rate=dropout
        )

        # Parameters
        self.thresh_pos = thresh
        self.thresh_neg = thresh

        self.stain_info = stain_info

        self.apply(self._init_weights)

    def forward(self,
                x: Tensor,
                fname: str,
                ) -> Tuple[Tensor]:
        N = x.size(0)

        # Add a stain_info indicator vector

        if self.stain_info:
            is_img_diffquick = self.diffquick_or_papsmear(
                fname, keywords=['DQ']
            )
            if is_img_diffquick:
                stain = torch.tensor([1., 0.])
            else:
                stain = torch.tensor([0., 1.])
            stain = stain.repeat(N, 1)
            stain = stain.cuda()
            x_stain = torch.cat([x.clone(), stain], dim=1)

        # Encoder

        feats = self.embedder(x_stain)

        # Foreground and background

        ccam = self.disentangler(feats)
        ccam = ccam.reshape((N,))
        fg = torch.einsum(
            'ns, n -> ns',
            feats,
            ccam
        )
        bg = torch.einsum(
            'ns, n -> ns',
            feats,
            1 - ccam
        )

        # Select instances for bag

        # fgbg = torch.cat([fg, bg], dim=1)
        fgbg = fg.clone()

        fgbg_maps = self.get_cam_1d(
            # classifier=self.fg_1d_clf,
            classifier=self.final_classifier,
            features=fgbg
        )

        fgbg_softmax = torch.softmax(
            fgbg_maps.transpose(1, 0),
            dim=1
        )
        pos_softmax = fgbg_softmax[:, -1]
        neg_softmax = fgbg_softmax[:, 0]

        all_softmax = torch.cat([pos_softmax, neg_softmax], dim=0)

        threshold = (all_softmax.max()) * self.thresh_pos

        pos_top_k = len(pos_softmax[pos_softmax >= threshold])
        neg_top_k = len(neg_softmax[neg_softmax >= threshold])

        _, pos_topk_idx = torch.sort(pos_softmax, descending=True)
        pos_topk_idx = pos_topk_idx[:pos_top_k]

        _, neg_topk_idx = torch.sort(neg_softmax, descending=True)
        neg_topk_idx = neg_topk_idx[:neg_top_k]

        if pos_top_k > 0 and neg_top_k == 0:
            h = torch.index_select(fgbg, 0, pos_topk_idx)
            o = torch.index_select(pos_softmax, 0, pos_topk_idx)
        elif pos_top_k == 0 and neg_top_k > 0:
            h = torch.index_select(fgbg, 0, neg_topk_idx)
            o = torch.index_select(neg_softmax, 0, neg_topk_idx)
        elif pos_top_k > 0 and neg_top_k > 0:
            h = torch.cat(
                [
                    torch.index_select(fgbg, 0, pos_topk_idx),
                    torch.index_select(fgbg, 0, neg_topk_idx),
                ],
                dim=0
            )
            o = torch.cat(
                [
                    torch.index_select(pos_softmax, 0, pos_topk_idx),
                    torch.index_select(neg_softmax, 0, neg_topk_idx),
                ],
                dim=0
            )

        logits = self.final_classifier(
            torch.mm(
                o.unsqueeze(0),
                h
            )
        )

        topk_idx = {'pos_idx': pos_topk_idx, 'neg_idx': neg_topk_idx}
        softmaxes = {'pos_sm': pos_softmax, 'neg_sm': neg_softmax}
        thresholds = {'pos_thr': self.thresh_pos, 'neg_thr': self.thresh_neg}

        return logits, fg, bg, topk_idx, softmaxes, thresholds
