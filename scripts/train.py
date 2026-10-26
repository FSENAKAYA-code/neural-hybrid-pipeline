"""Run a small aligned multimodal embedding demo without external data."""

import torch

from hybrid.experiment import train_fusion_head


def main() -> None:
    torch.manual_seed(42)
    labels = torch.arange(300) % 3
    image_embeddings = torch.randn(300, 32) + labels[:, None] * 0.15
    tabular_features = torch.randn(300, 8) + labels[:, None] * 0.15
    print(train_fusion_head(image_embeddings, tabular_features, labels, epochs=5))


if __name__ == "__main__":
    main()
