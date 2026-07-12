from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Two-layer MLP head for classification."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=float(dropout)) if dropout > 0 else nn.Identity()

        if hidden_dim is None or hidden_dim <= 0:
            self.net = nn.Linear(in_dim, num_classes)
        else:
            act = nn.GELU() if activation.lower() == "gelu" else nn.ReLU()
            ln = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                ln,
                act,
                self.dropout,
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"ClassificationHead expects [B,D], got {tuple(x.shape)}")
        return self.net(x)


__all__ = ["ClassificationHead"]
