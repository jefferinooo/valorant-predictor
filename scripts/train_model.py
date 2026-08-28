"""
CLI script to train the LSTM win probability model.

Run:
    python3 scripts/train_model.py
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
    dataset_path = Path(settings.data_dir) / "processed" / "dataset.npz"
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}. Run scripts/build_features.py first.")
        sys.exit(1)

    cfg = TrainConfig()
    train(cfg, dataset_path)
