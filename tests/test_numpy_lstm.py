"""The numpy serving model must match the PyTorch model it was exported from."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.serving.export import export_weights
from src.serving.numpy_lstm import NumpyLSTM
from src.training.config import TrainConfig
from src.training.train import WinProbLSTM


@pytest.fixture
def models(tmp_path):
    torch.manual_seed(0)
    cfg = TrainConfig()
    model = WinProbLSTM(cfg).eval()
    ckpt = tmp_path / "m.pt"
    torch.save({"model_state": model.state_dict(), "config": cfg}, ckpt)
    return model, NumpyLSTM.from_npz(export_weights(ckpt))


@pytest.mark.parametrize("T", [1, 5, 14])
def test_numpy_matches_torch(models, T):
    torch_model, np_model = models
    rng = np.random.default_rng(T)
    seq = rng.random((T, 15), dtype=np.float32)
    with torch.no_grad():
        expected = torch_model.forward_all(torch.tensor(seq[None]), torch.tensor([T]))[0].numpy()
    np.testing.assert_allclose(np_model.predict_all(seq), expected, atol=1e-5)


def test_committed_weights_match_committed_checkpoint():
    """Guards against retraining without re-exporting."""
    ckpt, npz = Path("artifacts/models/best.pt"), Path("artifacts/models/best_weights.npz")
    if not (ckpt.exists() and npz.exists()):
        pytest.skip("artifacts not present")
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    torch_model = WinProbLSTM(c["config"]).eval()
    torch_model.load_state_dict(c["model_state"])
    seq = np.random.default_rng(0).random((9, 15), dtype=np.float32)
    with torch.no_grad():
        expected = torch_model.forward_all(torch.tensor(seq[None]), torch.tensor([9]))[0].numpy()
    np.testing.assert_allclose(NumpyLSTM.from_npz(npz).predict_all(seq), expected, atol=1e-5)
