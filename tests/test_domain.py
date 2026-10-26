import pytest

torch = pytest.importorskip("torch")
from hybrid.experiment import train_fusion_head
from hybrid.fusion import LateFusionHead


def test_fusion_head_checks_aligned_batches():
    model = LateFusionHead(8, 3, 2)
    with pytest.raises(ValueError, match="aligned"):
        model(torch.zeros(2, 8), torch.zeros(3, 3))


def test_fusion_training_reports_validation_and_test_scores():
    labels = torch.arange(60) % 2
    scores = train_fusion_head(torch.randn(60, 8), torch.randn(60, 3), labels, epochs=2)
    assert set(scores) == {"validation_accuracy", "test_accuracy"}
