"""
CLI script to build the training dataset from raw match JSON files.

Run:
    python3 scripts/build_features.py        # v1: 15 features (served by the API)
    python3 scripts/build_features.py --v2   # v2: + loadout, trade, spike distance
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.features.sequences import save_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

if __name__ == "__main__":
    out = save_dataset(data_dir=settings.data_dir, version=2 if "--v2" in sys.argv else 1)
    print(f"Dataset saved to {out}")
