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
    python3 scripts/baseline.py        # v1 models -> artifacts/situations.md
    python3 scripts/baseline.py --v2   # adds v2 models -> artifacts/situations_v2.md
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
from src.features.sequences import V2_FEATURE_NAMES
from src.training.situations import situation_table, to_markdown
from src.training.train import match_split


ALL_FEATURE_NAMES = FEATURE_NAMES + V2_FEATURE_NAMES


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
            print(f"    {ALL_FEATURE_NAMES[i]:>15s}  {clf.coef_[0][i]:+.3f}")


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
    p_snap = predict(sequences[te_live])
    report("Baseline 2: Current snapshot only (no temporal memory)",
           live_metrics(p_snap, y_test), clf)

    # --- LSTM ---
    ckpt = Path(cfg.checkpoint_path)
    if ckpt.exists():
        device = torch.device("cpu")
        model, _ = load_model(ckpt, device)
        probs = predict_all_steps(model, sequences[test_idx], lengths[test_idx], device)
        p_lstm = probs[live[test_idx]]
        report("LSTM (full event history)", live_metrics(p_lstm, y_test))

        preds = {"LR start": p_start, "LR snapshot": p_snap, "LSTM": p_lstm}
        primary, compare, out = "LSTM", "LR snapshot", Path("artifacts/situations.md")

        # --- v2: same rounds and live states, richer features ---
        if "--v2" in sys.argv:
            v2 = np.load("data/processed/dataset_v2.npz")
            assert (v2["lengths"] == lengths).all() and (v2["labels"] == labels).all(), \
                "v1 and v2 datasets must contain the same rounds and snapshots"
            seq2 = v2["sequences"]
            predict2, clf2 = fit_lr(seq2[tr_live], step_labels[tr_live])
            preds["LR snapshot v2"] = predict2(seq2[te_live])
            report("Baseline 2 v2: Current snapshot with v2 features",
                   live_metrics(preds["LR snapshot v2"], y_test), clf2)
            model2, _ = load_model("artifacts/models/best_v2.pt", device)
            probs2 = predict_all_steps(model2, seq2[test_idx], lengths[test_idx], device)
            preds["LSTM v2"] = probs2[live[test_idx]]
            report("LSTM v2 (event history + v2 features)", live_metrics(preds["LSTM v2"], y_test))
            primary, compare, out = "LSTM v2", "LSTM", Path("artifacts/situations_v2.md")

        # --- Results by situation ---
        match_ids = np.broadcast_to(data["match_indices"][:, None], live.shape)[te_live]
        rows = situation_table(sequences[te_live], y_test, preds, match_ids,
                               primary=primary, compare=compare)
        table = to_markdown(rows, primary=primary, compare=compare)
        out.write_text(table + "\n")
        print("\nResults by situation (test set, live states):\n")
        print(table)
        print(f"\nSaved to {out}")
    else:
        print(f"\nNo checkpoint at {ckpt}; run scripts/train_model.py to compare the LSTM.")


if __name__ == "__main__":
    main()
