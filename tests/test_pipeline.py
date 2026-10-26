import numpy as np
import pytest
from sklearn.datasets import make_classification

from hybrid.experiment import run_experiment
from hybrid.splits import split_data


def test_split_has_three_disjoint_deterministic_partitions():
    X, y = make_classification(n_samples=200, n_features=8, random_state=5)
    first = split_data(X, y)
    second = split_data(X, y)
    assert len(first.X_train) == 120
    assert len(first.X_validation) == 40
    assert len(first.X_test) == 40
    np.testing.assert_array_equal(first.X_test, second.X_test)


def test_split_rejects_too_small_classes():
    with pytest.raises(ValueError, match="five or more"):
        split_data(np.arange(8).reshape(4, 2), np.array([0, 0, 1, 1]))


def test_experiment_reports_validation_and_test_metrics():
    X, y = make_classification(n_samples=250, n_features=8, random_state=7)
    scores = run_experiment(X, y)
    assert set(scores) == {
        "validation_f1_weighted", "test_accuracy", "test_f1_weighted"
    }
    assert all(0.0 <= value <= 1.0 for value in scores.values())
