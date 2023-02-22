from abc import ABC, abstractmethod
from typing import List

from timm.models.layers import trunc_normal_
from torch import Tensor, einsum, nn


class BaseModel(nn.Module, ABC):

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, (nn.Linear)) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    @staticmethod
    def diffquick_or_papsmear(
        filename: str,
        keywords: List[str]
    ) -> bool:
        for keyword in keywords:
            if keyword in filename:
                return True
        return False

    @staticmethod
    def get_cam_1d(classifier: nn.Module,
                   features: Tensor
                   ) -> Tensor:
        t_weight = list(
            classifier.parameters()
        )
        final_weight = t_weight[-2]

        cam_maps = einsum(
            'gf, cf -> cg',
            features,
            final_weight
        )
        return cam_maps

    @abstractmethod
    def forward(self):
        pass
