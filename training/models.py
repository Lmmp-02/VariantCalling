"""
Shared model architectures for training, evaluation and calling.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GroupwiseLinear3C(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 3):
        super().__init__()
        self.trunk = nn.Identity()
        self.heads = nn.ModuleDict({
            "1": nn.Linear(in_dim, n_classes),
            "10": nn.Linear(in_dim, n_classes),
            "11": nn.Linear(in_dim, n_classes),
        })

    def forward(self, x: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        out = torch.zeros((h.shape[0], 3), dtype=h.dtype, device=h.device)

        for g in (1, 10, 11):
            mask = (groups == g)
            if mask.any():
                out[mask] = self.heads[str(g)](h[mask])

        return out


class GroupwiseMLP3C(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 3, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict({
            "1": nn.Linear(hidden, n_classes),
            "10": nn.Linear(hidden, n_classes),
            "11": nn.Linear(hidden, n_classes),
        })

    def forward(self, x: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        out = torch.zeros((h.shape[0], 3), dtype=h.dtype, device=h.device)

        for g in (1, 10, 11):
            mask = (groups == g)
            if mask.any():
                out[mask] = self.heads[str(g)](h[mask])

        return out
