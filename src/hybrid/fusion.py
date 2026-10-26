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


class JointHybridClassifier(nn.Module):
    """Encode raw RGB images and tabular rows, then optimize one joint loss."""

    def __init__(self, tabular_features: int, num_classes: int) -> None:
        super().__init__()
        if tabular_features < 1 or num_classes < 2:
            raise ValueError("tabular features must be positive and classes at least two")
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
        )
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_features, 32), nn.ReLU(), nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, images: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3 or tabular.ndim != 2:
            raise ValueError("expected RGB batches and a 2D tabular feature matrix")
        if images.shape[0] != tabular.shape[0]:
            raise ValueError("image and tabular rows must be aligned")
        image_features = self.image_encoder(images)
        tabular_features = self.tabular_encoder(tabular)
        return self.classifier(torch.cat([image_features, tabular_features], dim=1))
