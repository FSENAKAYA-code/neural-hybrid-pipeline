"""Train and evaluate a late-fusion head on aligned image/tabular embeddings."""

from copy import deepcopy

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn

from hybrid.fusion import JointHybridClassifier, LateFusionHead


def train_fusion_head(
    image_embeddings: torch.Tensor,
    tabular_features: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int = 10,
    seed: int = 42,
) -> dict[str, float]:
    if image_embeddings.ndim not in (2, 4) or tabular_features.ndim != 2 or labels.ndim != 1:
        raise ValueError("expected embeddings or RGB image batches, tabular matrix, and labels")
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
    class_values = torch.unique(labels).tolist()
    if class_values != list(range(len(class_values))):
        raise ValueError("labels must be contiguous integers starting at zero")
    classes = len(class_values)
    if image_embeddings.ndim == 4:
        model = JointHybridClassifier(tabular_features.shape[1], classes).to(device)
    else:
        model = LateFusionHead(image_embeddings.shape[1], tabular_features.shape[1], classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.CrossEntropyLoss()
    best_state = deepcopy(model.state_dict())
    best_validation_accuracy = -1.0
    best_train_loss = float("inf")
    best_validation_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(image_embeddings[train], tabular_features[train])
        loss = loss_function(logits, labels[train])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_logits = model(
                image_embeddings[validation], tabular_features[validation]
            )
            validation_loss = float(loss_function(
                validation_logits, labels[validation]
            ).item())
            validation_predictions = validation_logits.argmax(dim=1)
        score = accuracy_score(labels[validation].cpu(), validation_predictions.cpu())
        print(
            f"epoch={epoch + 1} train_loss={loss.item():.4f} "
            f"validation_loss={validation_loss:.4f}"
        )
        if score > best_validation_accuracy:
            best_validation_accuracy = float(score)
            best_train_loss = float(loss.item())
            best_validation_loss = validation_loss
            best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_predictions = model(image_embeddings[test], tabular_features[test]).argmax(dim=1)
    test_true = labels[test].cpu()
    test_pred = test_predictions.cpu()
    return {
        "train_loss": best_train_loss,
        "validation_loss": best_validation_loss,
        "validation_accuracy": best_validation_accuracy,
        "test_accuracy": float(accuracy_score(test_true, test_pred)),
        "test_precision_weighted": float(
            precision_score(test_true, test_pred, average="weighted", zero_division=0)
        ),
        "test_recall_weighted": float(
            recall_score(test_true, test_pred, average="weighted", zero_division=0)
        ),
        "test_f1_weighted": float(
            f1_score(test_true, test_pred, average="weighted", zero_division=0)
        ),
    }
