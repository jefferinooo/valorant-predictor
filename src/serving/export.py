"""Export a trained PyTorch checkpoint to the .npz weights used for serving."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def export_weights(checkpoint_path: str | Path, out_path: str | Path | None = None) -> Path:
    import torch  # training-time dependency only

    checkpoint_path = Path(checkpoint_path)
    out_path = Path(out_path) if out_path else checkpoint_path.with_name(checkpoint_path.stem + "_weights.npz")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    arrays = {k: v.cpu().numpy() for k, v in ckpt["model_state"].items()}
    np.savez(out_path, num_layers=np.array(ckpt["config"].num_layers), **arrays)
    return out_path
