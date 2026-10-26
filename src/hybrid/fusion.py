"""A small late-fusion head for paired image and tabular embeddings."""

import torch
from torch import nn


class LateFusionHead(nn.Module):
    def __init__(self, image_features: int, tabular_features: int, num_classes: int) -> None:
        super().__init__()
        if min(image_features, tabular_features) < 1 or num_classes < 2:
            raise ValueError("feature dimensions must be positive and classes at least two")
        self.head = nn.Sequential(
            nn.Linear(image_features + tabular_features, 64), nn.ReLU(),
            nn.Dropout(p=0.2), nn.Linear(64, num_classes),
        )

    def forward(self, image_embedding: torch.Tensor, tabular_embedding: torch.Tensor) -> torch.Tensor:
        if image_embedding.ndim != 2 or tabular_embedding.ndim != 2:
            raise ValueError("both embeddings must have shape (batch, features)")
        if image_embedding.shape[0] != tabular_embedding.shape[0]:
            raise ValueError("image and tabular batches must be aligned")
        return self.head(torch.cat([image_embedding, tabular_embedding], dim=1))
