import pytest

torch = pytest.importorskip("torch")

from src.dl_model import LSTMForecaster  # noqa: E402


def test_forward_shape_and_one_train_step():
    m = LSTMForecaster(n_seq_features=15, n_known_features=14, n_items=3, n_stores=2, hidden=8)
    x = torch.randn(5, 28, 15)
    known = torch.randn(5, 14)
    item = torch.tensor([0, 1, 2, 0, 1])
    store = torch.tensor([0, 1, 0, 1, 0])
    y = torch.randn(5)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    before = torch.nn.functional.mse_loss(m(x, known, item, store), y)
    assert m(x, known, item, store).shape == (5,)
    before.backward()
    opt.step()
    after = torch.nn.functional.mse_loss(m(x, known, item, store), y)
    assert after < before
