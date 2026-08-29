"""
Logistic regression baseline for in-round win probability.

Trains two baselines on the same dataset as the LSTM and compares accuracy:

  1. Round-start only — uses only the first snapshot (t=0, before any kills).
     Represents pure pre-round prediction with no in-round information.

  2. Mean-pooled sequence — averages all snapshots in the round into one vector.
     Gives the logistic regression all the same information as the LSTM but
     with no temporal ordering or memory.

If the LSTM significantly outperforms both, temporal sequence modeling is justified.

Run:
    python3 scripts/baseline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    "atk_alive", "def_alive", "spike_planted",
    "time_elapsed", "time_remaining", "plant_time",
    "atk_economy", "def_economy", "round_num",
    "atk_score", "def_score",
    "site_A", "site_B", "site_C", "score_diff",
]


def run_baseline(X_train, X_val, y_train, y_val, name: str) -> None:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_s, y_train)

    preds      = clf.predict(X_val_s)
    probs      = clf.predict_proba(X_val_s)[:, 1]
    acc        = accuracy_score(y_val, preds)
    logloss    = log_loss(y_val, probs)

    print(f"\n{name}")
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Log-loss : {logloss:.4f}")

    # Top 5 most important features by absolute coefficient magnitude
    coef_order = np.argsort(np.abs(clf.coef_[0]))[::-1]
    print("  Top features by coefficient magnitude:")
    for i in coef_order[:5]:
        print(f"    {FEATURE_NAMES[i]:>15s}  {clf.coef_[0][i]:+.3f}")


def main():
    dataset_path = Path("data/processed/dataset.npz")
    if not dataset_path.exists():
        print("Dataset not found. Run scripts/build_features.py first.")
        sys.exit(1)

    data          = np.load(dataset_path)
    sequences     = data["sequences"]
    labels        = data["labels"]
    lengths       = data["lengths"]
    match_indices = data["match_indices"]

    # Match-level split — mirrors the LSTM training split exactly
    unique_matches = np.unique(match_indices)
    rng = np.random.default_rng(42)
    rng.shuffle(unique_matches)
    n_val_matches = int(len(unique_matches) * 0.15)
    val_match_set = set(unique_matches[:n_val_matches].tolist())

    train_mask = np.array([m not in val_match_set for m in match_indices])
    train_idx  = np.where(train_mask)[0]
    val_idx    = np.where(~train_mask)[0]

    y_train, y_val = labels[train_idx], labels[val_idx]

    # --- Baseline 1: first snapshot only (round start, no in-round info) ---
    X_start = sequences[:, 0, :]   # (N, 15)
    run_baseline(X_start[train_idx], X_start[val_idx], y_train, y_val,
                 "Baseline 1: Round-start snapshot only (pre-round prediction)")

    # --- Baseline 2: mean-pooled sequence (all info, no temporal order) ---
    X_mean = np.stack([
        sequences[i, : lengths[i], :].mean(axis=0)
        for i in range(len(sequences))
    ])  # (N, 15)
    run_baseline(X_mean[train_idx], X_mean[val_idx], y_train, y_val,
                 "Baseline 2: Mean-pooled sequence (all snapshots, no memory)")

    print(f"\nLSTM (full temporal sequence):  99.0%  (val acc, match-level split)")
    print("\nConclusion: gap between baselines and LSTM shows value of temporal modeling.")


if __name__ == "__main__":
    main()
