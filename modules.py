from typing import Tuple

from torch import Tensor, nn


class GatedAttention(nn.Module):
    def __init__(
        self,
        in_dim: int,
        intermediate_dim: int,
        out_dim: int = 1,
        dropout: bool = False
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.intermediate_dim = intermediate_dim
        self.out_dim = out_dim

        self.attn_a = [
            nn.Linear(in_dim, intermediate_dim),
            nn.Tanh(),
        ]
        self.attn_b = [
            nn.Linear(in_dim, intermediate_dim),
            nn.Sigmoid()
        ]

        if dropout:
            self.attn_a.append(nn.Dropout(0.25))
            self.attn_b.append(nn.Dropout(0.25))

        self.attn_a = nn.Sequential(*self.attn_a)
        self.attn_b = nn.Sequential(*self.attn_b)

        self.final_attn = nn.Linear(intermediate_dim, out_dim)

    def forward(self,
                x: Tensor,
                ) -> Tuple[Tensor, Tensor]:
        attn_a = self.attn_a(x)
        attn_b = self.attn_b(x)

        A = attn_a.mul(attn_b)
        A = self.final_attn(A)

        return A, x


class FinalClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout_rate: float = 0.
    ) -> None:
        super().__init__()
        layers = [
            nn.GroupNorm(num_groups=int(in_channels/16),
                         num_channels=in_channels),
            nn.Linear(in_channels, num_classes),
        ]
        if dropout_rate != 0.0:
            layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.Mish())
        self.final_fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.final_fc(x)
