"""
Export the trained checkpoint to numpy weights for serving.

Run:
    python3 scripts/export_weights.py   # artifacts/models/best.pt -> best_weights.npz

train.py does this automatically after training; this script is for
re-exporting an existing checkpoint.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.serving.export import export_weights

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "artifacts/models/best.pt"
    print(f"Exported -> {export_weights(ckpt)}")
