from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn.functional import softmax

from model_base import BaseModel
from modules import GatedAttention


class ABMIL(BaseModel):
    def __init__(
            self,
            in_channels: int,
            intermediate_dim: int,
            n_classes: int,
            stain_info: bool,
            dropout: bool
    ) -> None:
        super().__init__()

        # Mini-encoder for pretrained features
        if stain_info:
            in_channels += 2

        self.projector = nn.Sequential(
            nn.Linear(in_channels, intermediate_dim),
            nn.ReLU(),
        )
        # Attention mechanism
        self.attn = GatedAttention(
            in_dim=intermediate_dim,
            intermediate_dim=int(intermediate_dim//2),
            out_dim=1,
            dropout=dropout
        )

        # Classifier for final bag logits
        self.final_classifier = nn.Sequential(
            nn.Linear(intermediate_dim, n_classes),
            nn.Sigmoid()
        )

        # Parameters
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

        feats = self.projector(x_stain)

        # Attention

        A, _ = self.attn(feats)  # [N, C]
        A = torch.transpose(A, 1, 0)  # [C, N]
        A = softmax(A, dim=1)  # softmax over N

        _, top_ids = torch.sort(A, descending=True)
        topk_idx = {'pos_idx': top_ids, 'neg_idx': None}

        M = torch.mm(A, feats)

        logits = self.final_classifier(M)

        return logits, A, topk_idx
