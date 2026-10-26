import pytest

torch = pytest.importorskip("torch")
from hybrid.fusion import LateFusionHead


def test_fusion_head_checks_aligned_batches():
    model = LateFusionHead(8, 3, 2)
    with pytest.raises(ValueError, match="aligned"):
        model(torch.zeros(2, 8), torch.zeros(3, 3))
