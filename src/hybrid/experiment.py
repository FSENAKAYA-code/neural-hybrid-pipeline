"""Train and evaluate a late-fusion head on aligned image/tabular embeddings."""

from copy import deepcopy

import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn

from hybrid.fusion import LateFusionHead


def train_fusion_head(
    image_embeddings: torch.Tensor,
    tabular_features: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int = 10,
    seed: int = 42,
) -> dict[str, float]:
    if image_embeddings.ndim != 2 or tabular_features.ndim != 2 or labels.ndim != 1:
        raise ValueError("expected 2D feature matrices and 1D labels")
    if not (len(image_embeddings) == len(tabular_features) == len(labels)):
        raise ValueError("paired image and tabular rows must be aligned")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    label_values = labels.detach().cpu().numpy()
    if min(torch.unique(labels, return_counts=True)[1]).item() < 5:
        raise ValueError("each class needs at least five examples for three-way evaluation")
    indices = list(range(len(labels)))
    train_val, test = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=label_values
    )
    train, validation = train_test_split(
        train_val,
        test_size=0.25,
        random_state=seed,
        stratify=label_values[train_val],
    )
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_embeddings = image_embeddings.float().to(device)
    tabular_features = tabular_features.float().to(device)
    labels = labels.long().to(device)
    classes = int(labels.max().item()) + 1
    model = LateFusionHead(image_embeddings.shape[1], tabular_features.shape[1], classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.CrossEntropyLoss()
    best_state = deepcopy(model.state_dict())
    best_validation_accuracy = -1.0

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(image_embeddings[train], tabular_features[train])
        loss = loss_function(logits, labels[train])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_predictions = model(
                image_embeddings[validation], tabular_features[validation]
            ).argmax(dim=1)
        score = accuracy_score(labels[validation].cpu(), validation_predictions.cpu())
        if score > best_validation_accuracy:
            best_validation_accuracy = float(score)
            best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_predictions = model(image_embeddings[test], tabular_features[test]).argmax(dim=1)
    return {
        "validation_accuracy": best_validation_accuracy,
        "test_accuracy": float(accuracy_score(labels[test].cpu(), test_predictions.cpu())),
    }
