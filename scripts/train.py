"""Run a small aligned multimodal embedding demo without external data."""

import torch

from hybrid.experiment import train_fusion_head


def main() -> None:
    torch.manual_seed(42)
    labels = torch.arange(300) % 3
    images = torch.randn(300, 3, 24, 24) + labels[:, None, None, None] * 0.15
    tabular_features = torch.randn(300, 8) + labels[:, None] * 0.15
    print(train_fusion_head(images, tabular_features, labels, epochs=3))


if __name__ == "__main__":
    main()
