"""
CLI script to train the LSTM win probability model.

Run:
    python3 scripts/train_model.py        # v1 -> artifacts/models/best.pt (served by the API)
    python3 scripts/train_model.py --v2   # v2 -> artifacts/models/best_v2.pt (offline)
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.training.config import TrainConfig
from src.training.train import train

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

if __name__ == "__main__":
    v2 = "--v2" in sys.argv
    dataset_path = Path(settings.data_dir) / "processed" / ("dataset_v2.npz" if v2 else "dataset.npz")
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}. Run scripts/build_features.py first.")
        sys.exit(1)

    cfg = TrainConfig(checkpoint_path="artifacts/models/best_v2.pt") if v2 else TrainConfig()
    train(cfg, dataset_path)
