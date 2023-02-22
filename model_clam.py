from typing import Tuple

import torch
from topk.svm import SmoothTop1SVM
from torch import Tensor, nn
from torch.nn.functional import one_hot, softmax

from model_base import BaseModel
from modules import GatedAttention


class CLAM(BaseModel):
    def __init__(
        self,
        in_channels: int,
        intermediate_dim: int,
        dropout: bool = False,
        k_sample: int = 8,
        n_classes: int = 2,
        inst_loss_type: str = "svm",
        stain_info: bool = True,
    ) -> None:
        super().__init__()

        if stain_info is True:
            in_channels += 2

        # Encoder with gated-attention
        fc = [
            nn.Linear(in_channels, intermediate_dim),
            nn.ReLU()
        ]
        if dropout:
            fc.append(nn.Dropout(0.25))

        attn = GatedAttention(
            in_dim=intermediate_dim,
            intermediate_dim=int(intermediate_dim // 2),
            out_dim=1,
            dropout=dropout
        )

        fc.append(attn)
        self.attn_net = nn.Sequential(*fc)

        # Instance clustering classifiers
        self.classifiers = nn.Linear(intermediate_dim, n_classes)
        instance_classifiers = [
            nn.Linear(intermediate_dim, 2) for _ in range(n_classes)
        ]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)

        # Instance loss
        if inst_loss_type == "svm":
            self.instance_loss_fn = SmoothTop1SVM(
                n_classes=n_classes)
        else:
            self.instance_loss_fn = nn.CrossEntropyLoss()

        self.inst_loss_type = inst_loss_type

        # Additional parameters
        self.k_sample = k_sample
        self.n_classes = n_classes
        self.stain_info = stain_info

        # Model init
        self.apply(self._init_weights)

    @staticmethod
    def create_positive_targets(length: int, device: str) -> Tensor:
        return torch.full(
            (length, ), 1, device=device
        ).long()

    @staticmethod
    def create_negative_targets(length: int, device: str) -> Tensor:
        return torch.full(
            (length, ), 0, device=device
        ).long()

    # instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, k_sample, classifier):
        device = h.device
        if self.inst_loss_type == "svm":
            self.instance_loss_fn.cuda(device=device)

        if len(A.shape) == 1:
            A = A.view(1, -1)

        if A.shape[1] <= k_sample:
            k_sample = A.shape[1]

        top_p_ids = torch.topk(A, k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)

        p_targets = self.create_positive_targets(k_sample, device)
        n_targets = self.create_negative_targets(k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        # all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        # return instance_loss, all_preds, all_targets
        return instance_loss, top_p_ids, top_n_ids

    def forward(
            self,
            x: Tensor,
            fname: str,
            label: Tensor = None,
            attention_only: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        num_instances = x.shape[0]

        if self.stain_info:
            is_img_diffquick = self.diffquick_or_papsmear(
                fname, keywords=['DQ']
            )
            if is_img_diffquick:
                stain = torch.tensor([1., 0.])
            else:
                stain = torch.tensor([0., 1.])
            stain = stain.repeat(num_instances, 1)
            stain = stain.cuda()
            x = torch.cat([x, stain], dim=1)

        A, x = self.attn_net(x)
        k_sample = self.k_sample

        A = torch.transpose(A, 1, 0)  # [class, N]
        if attention_only:
            return A
        A_raw = A
        A = softmax(A, dim=1)  # softmax over N

        total_inst_loss = 0.0
        inst_labels = one_hot(
            label, num_classes=self.n_classes).squeeze()
        for i in range(len(self.instance_classifiers)):
            inst_label = inst_labels[i].item()
            classifier = self.instance_classifiers[i]
            if inst_label == 1:  # in-the-class
                instance_loss, top_p_ids, top_n_ids = self.inst_eval(
                    A,
                    x,
                    k_sample,
                    classifier)
            else:  # out-of-the-class
                continue
            total_inst_loss += instance_loss

        M = torch.mm(A, x)
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = softmax(logits, dim=1)
        topk_idx = {'pos_idx': top_p_ids, 'neg_idx': top_n_ids}

        return logits, Y_prob, Y_hat, total_inst_loss, A_raw, topk_idx
