import pytest

torch = pytest.importorskip("torch")
from hybrid.experiment import train_fusion_head
from hybrid.fusion import JointHybridClassifier, LateFusionHead


def test_fusion_head_checks_aligned_batches():
    model = LateFusionHead(8, 3, 2)
    with pytest.raises(ValueError, match="aligned"):
        model(torch.zeros(2, 8), torch.zeros(3, 3))


def test_joint_model_encodes_rgb_and_tabular_inputs():
    model = JointHybridClassifier(5, 3)
    assert model(torch.zeros(2, 3, 24, 24), torch.zeros(2, 5)).shape == (2, 3)


def test_fusion_training_reports_validation_and_test_scores():
    labels = torch.arange(60) % 2
    scores = train_fusion_head(torch.randn(60, 8), torch.randn(60, 3), labels, epochs=2)
    assert {
        "train_loss", "validation_loss", "validation_accuracy", "test_accuracy",
        "test_precision_weighted", "test_recall_weighted", "test_f1_weighted",
    } == set(scores)


def test_joint_fusion_training_accepts_raw_rgb_batches():
    labels = torch.arange(30) % 2
    scores = train_fusion_head(torch.rand(30, 3, 12, 12), torch.randn(30, 4), labels, epochs=1)
    assert "train_loss" in scores and "test_f1_weighted" in scores
