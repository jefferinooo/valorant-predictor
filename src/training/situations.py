"""
Break model performance down by in-round situation.

Every function works on *live* states: a flat (n, 15) array of normalized
feature vectors (see src/features/sequences.py), one per state, with labels,
per-model predictions and the match each state came from.

Confidence intervals use a match-level (cluster) bootstrap. States from the
same round or match are strongly correlated, so resampling individual states
would make the intervals far too narrow.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# Decode normalized features (see _snapshot_to_vector)
def _atk(f: np.ndarray) -> np.ndarray:
    return np.rint(f[:, 0] * 5).astype(int)


def _def(f: np.ndarray) -> np.ndarray:
    return np.rint(f[:, 1] * 5).astype(int)


def _planted(f: np.ndarray) -> np.ndarray:
    return f[:, 2] > 0.5


def _elapsed_s(f: np.ndarray) -> np.ndarray:
    return f[:, 3] * 100.0


SITUATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "All live states":        lambda f: np.ones(len(f), dtype=bool),
    "Round start":            lambda f: _elapsed_s(f) == 0,
    "Pre-plant":              lambda f: ~_planted(f),
    "Post-plant":             lambda f: _planted(f),
    "Even numbers":           lambda f: _atk(f) == _def(f),
    "Attackers up":           lambda f: _atk(f) > _def(f),
    "Defenders up":           lambda f: _atk(f) < _def(f),
    "1vX clutch":             lambda f: ((_atk(f) == 1) & (_def(f) >= 2)) | ((_def(f) == 1) & (_atk(f) >= 2)),
    "1v1":                    lambda f: (_atk(f) == 1) & (_def(f) == 1),
    "Pre-plant, 0-30s":       lambda f: ~_planted(f) & (_elapsed_s(f) < 30),
    "Pre-plant, 30-60s":      lambda f: ~_planted(f) & (_elapsed_s(f) >= 30) & (_elapsed_s(f) < 60),
    "Pre-plant, 60s+":        lambda f: ~_planted(f) & (_elapsed_s(f) >= 60),
}


def _per_state_logloss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def situation_table(
    state_feats: np.ndarray,
    labels: np.ndarray,
    preds: dict[str, np.ndarray],
    match_ids: np.ndarray,
    primary: str,
    compare: str,
    n_boot: int = 1000,
    seed: int = 0,
) -> list[dict]:
    """
    One row per situation with the state count, every model's accuracy, the
    primary model's accuracy 95% CI, and the primary − compare log-loss
    difference with its 95% CI (negative = primary is better).
    """
    _, match_idx = np.unique(match_ids, return_inverse=True)
    n_matches = match_idx.max() + 1
    rng = np.random.default_rng(seed)
    # Each bootstrap draw = how many times each match was resampled
    draws = rng.multinomial(n_matches, np.full(n_matches, 1 / n_matches), size=n_boot)

    correct = {name: ((p >= 0.5) == labels.astype(bool)) for name, p in preds.items()}
    ll_diff = _per_state_logloss(preds[primary], labels) - _per_state_logloss(preds[compare], labels)

    def per_match(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return np.bincount(match_idx[mask], weights=values[mask], minlength=n_matches)

    rows = []
    for name, situation in SITUATIONS.items():
        mask = situation(state_feats)
        n = int(mask.sum())
        if n == 0:
            continue
        row: dict = {"situation": name, "n": n}
        for model, c in correct.items():
            row[f"acc:{model}"] = float(c[mask].mean())

        counts = per_match(np.ones(len(labels)), mask)
        boot_n = draws @ counts
        ok = boot_n > 0
        boot_acc = (draws @ per_match(correct[primary].astype(float), mask))[ok] / boot_n[ok]
        boot_dll = (draws @ per_match(ll_diff, mask))[ok] / boot_n[ok]

        row["acc_ci"] = tuple(np.percentile(boot_acc, [2.5, 97.5]))
        row["dll"] = float(ll_diff[mask].mean())
        row["dll_ci"] = tuple(np.percentile(boot_dll, [2.5, 97.5]))
        row["atk_win_rate"] = float(labels[mask].mean())
        rows.append(row)
    return rows


def to_markdown(rows: list[dict], primary: str, compare: str) -> str:
    models = [k.split(":", 1)[1] for k in rows[0] if k.startswith("acc:")]
    header = (
        ["Situation", "States", "Atk win rate"]
        + [f"{m} acc" for m in models]
        + [f"{primary} 95% CI", f"Δ log-loss vs {compare} (95% CI)"]
    )
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        lo, hi = r["acc_ci"]
        dlo, dhi = r["dll_ci"]
        cells = (
            [r["situation"], f"{r['n']:,}", f"{r['atk_win_rate']:.0%}"]
            + [f"{r['acc:' + m]:.1%}" for m in models]
            + [f"{lo:.1%} – {hi:.1%}", f"{r['dll']:+.3f} ({dlo:+.3f}, {dhi:+.3f})"]
        )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
