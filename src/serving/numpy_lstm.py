"""
Pure-numpy inference for the trained WinProbLSTM.

The model is trained in PyTorch (src/training/train.py). For serving we only
need the forward pass, which is a handful of matrix multiplies, so the
weights are exported to an .npz file (src/serving/export.py) and evaluated
here. That keeps PyTorch out of the production image.

Implements torch.nn.LSTM exactly (gate order i, f, g, o):
    gates = W_ih x_t + b_ih + W_hh h_{t-1} + b_hh
    i, f, o = sigmoid(...);  g = tanh(...)
    c_t = f * c_{t-1} + i * g
    h_t = o * tanh(c_t)
followed by the sigmoid head at every timestep. Dropout is inactive at
inference, so it's omitted. tests/test_numpy_lstm.py checks the outputs
match PyTorch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class NumpyLSTM:
    def __init__(self, layers: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                 head_w: np.ndarray, head_b: np.ndarray) -> None:
        # layers: (W_ih, W_hh, b_ih + b_hh) per layer; W_ih is transposed for x @ W
        self.layers = layers
        self.head_w = head_w
        self.head_b = head_b
        self.hidden_dim = layers[0][1].shape[0]

    @classmethod
    def from_npz(cls, path: str | Path) -> "NumpyLSTM":
        w = np.load(path)
        num_layers = int(w["num_layers"])
        layers = []
        for l in range(num_layers):
            layers.append((
                w[f"lstm.weight_ih_l{l}"].T.astype(np.float32),
                w[f"lstm.weight_hh_l{l}"].T.astype(np.float32),
                (w[f"lstm.bias_ih_l{l}"] + w[f"lstm.bias_hh_l{l}"]).astype(np.float32),
            ))
        return cls(layers, w["head.weight"].T.astype(np.float32), w["head.bias"].astype(np.float32))

    def predict_all(self, seq: np.ndarray) -> np.ndarray:
        """seq: (T, input_dim) -> attacker win probability after each timestep, (T,)."""
        x = np.asarray(seq, dtype=np.float32)
        H = self.hidden_dim
        for W_ih, W_hh, b in self.layers:
            # Input projection for all timesteps at once; only the recurrence is sequential
            x_proj = x @ W_ih + b
            h = np.zeros(H, dtype=np.float32)
            c = np.zeros(H, dtype=np.float32)
            out = np.empty((len(x), H), dtype=np.float32)
            for t in range(len(x)):
                gates = x_proj[t] + h @ W_hh
                i = _sigmoid(gates[:H])
                f = _sigmoid(gates[H : 2 * H])
                g = np.tanh(gates[2 * H : 3 * H])
                o = _sigmoid(gates[3 * H :])
                c = f * c + i * g
                h = o * np.tanh(c)
                out[t] = h
            x = out
        return _sigmoid(x @ self.head_w + self.head_b)[:, 0]
