"""Leakage-aware baseline training and final held-out evaluation."""

from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hybrid.splits import split_data


def run_experiment(X: Any, y: Any, *, random_state: int = 42) -> dict[str, float]:
    """Select with validation data, refit on train+validation, test once."""
    parts = split_data(X, y, random_state=random_state)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(parts.X_train, parts.y_train)
    validation_score = f1_score(
        parts.y_validation, model.predict(parts.X_validation), average="weighted"
    )
    model.fit(
        _concat(parts.X_train, parts.X_validation),
        _concat(parts.y_train, parts.y_validation),
    )
    predictions = model.predict(parts.X_test)
    return {
        "validation_f1_weighted": float(validation_score),
        "test_accuracy": float(accuracy_score(parts.y_test, predictions)),
        "test_f1_weighted": float(
            f1_score(parts.y_test, predictions, average="weighted")
        ),
    }


def _concat(left: Any, right: Any) -> Any:
    """Concatenate NumPy-like or pandas inputs without changing their rows."""
    if hasattr(left, "iloc"):
        import pandas as pd

        return pd.concat([left, right], axis=0)
    import numpy as np

    return np.concatenate([left, right], axis=0)
