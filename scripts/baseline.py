"""
Logistic regression baselines for in-round win probability.

Every model is scored on the same thing: all *live* snapshots (both teams
still have a player alive) from the held-out TEST matches.

  1. Round-start only — first snapshot of each round (t=0, before any kills),
     scored at every live state of that round. Pure pre-round prediction.

  2. Current snapshot — logistic regression on the current state alone.
     Same features the LSTM sees at that moment, but no memory of how the
     round got there.

  3. LSTM — loaded from artifacts/models/best.pt if present.

The gap between (2) and (3) is the honest measure of what temporal modeling adds.

Run:
    python3 scripts/baseline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.training.config import TrainConfig
from src.training.evaluate import (
    FEATURE_NAMES, live_mask, live_metrics, load_model, predict_all_steps,
)
from src.training.train import match_split


def fit_lr(X_train, y_train):
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=1000, random_state=42).fit(scaler.transform(X_train), y_train)
    return lambda X: clf.predict_proba(scaler.transform(X))[:, 1], clf


def report(name: str, m: dict, clf=None) -> None:
    print(f"\n{name}")
    print(f"  Accuracy : {m['accuracy']:.4f}  ({m['accuracy']*100:.1f}%)")
    print(f"  Log-loss : {m['log_loss']:.4f}")
    print(f"  Brier    : {m['brier']:.4f}")
    if clf is not None:
        print("  Top features by coefficient magnitude:")
        for i in np.argsort(np.abs(clf.coef_[0]))[::-1][:5]:
            print(f"    {FEATURE_NAMES[i]:>15s}  {clf.coef_[0][i]:+.3f}")


def main():
    dataset_path = Path("data/processed/dataset.npz")
    if not dataset_path.exists():
        print("Dataset not found. Run scripts/build_features.py first.")
        sys.exit(1)

    data      = np.load(dataset_path)
    sequences = data["sequences"]
    labels    = data["labels"]
    lengths   = data["lengths"]

    cfg = TrainConfig()
    # Baselines fit on train; everything is scored on the untouched test matches
    train_idx, _, test_idx = match_split(data["match_indices"], cfg.val_split, cfg.test_split, cfg.seed)

    live = live_mask(sequences, lengths)
    step_labels = np.broadcast_to(labels[:, None], live.shape)
    round_of_step = np.broadcast_to(np.arange(len(labels))[:, None], live.shape)

    tr_live = np.zeros_like(live); tr_live[train_idx] = live[train_idx]
    te_live = np.zeros_like(live); te_live[test_idx] = live[test_idx]
    y_test = step_labels[te_live]
    print(f"Test: {len(test_idx)} rounds, {te_live.sum()} live states")

    # --- Baseline 1: round-start snapshot, applied to every live state of the round ---
    predict, clf = fit_lr(sequences[train_idx, 0, :], labels[train_idx])
    p_start = predict(sequences[:, 0, :])[round_of_step[te_live]]
    report("Baseline 1: Round-start snapshot only (pre-round prediction)",
           live_metrics(p_start, y_test), clf)

    # --- Baseline 2: current snapshot only (no history) ---
    predict, clf = fit_lr(sequences[tr_live], step_labels[tr_live])
    report("Baseline 2: Current snapshot only (no temporal memory)",
           live_metrics(predict(sequences[te_live]), y_test), clf)

    # --- LSTM ---
    ckpt = Path(cfg.checkpoint_path)
    if ckpt.exists():
        device = torch.device("cpu")
        model, _ = load_model(ckpt, device)
        probs = predict_all_steps(model, sequences[test_idx], lengths[test_idx], device)
        report("LSTM (full kill-by-kill history)", live_metrics(probs[live[test_idx]], y_test))
    else:
        print(f"\nNo checkpoint at {ckpt}; run scripts/train_model.py to compare the LSTM.")


if __name__ == "__main__":
    main()
