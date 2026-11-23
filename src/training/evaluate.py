"""
Evaluation for the LSTM win probability model.

All metrics are computed on the held-out TEST matches, which play no part
in training or checkpoint selection, and on *live* states only: every in-round snapshot where
both teams still have at least one player alive. Final snapshots where one
side has been wiped are excluded because they give away the outcome.

Produces four outputs saved to artifacts/:
  1. calibration_curve.png   — predicted probability vs actual win rate
  2. timestep_accuracy.png   — accuracy at each kill-event index
  3. trajectories.png        — win probability over time for sample rounds
  4. feature_importance.png  — permutation importance on live states
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.training.config import TrainConfig
from src.training.train import WinProbLSTM, match_split

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "atk_alive", "def_alive", "spike_planted",
    "time_elapsed", "time_remaining", "plant_time",
    "atk_economy", "def_economy", "round_num",
    "atk_score", "def_score",
    "site_A", "site_B", "site_C", "score_diff",
]


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[WinProbLSTM, TrainConfig]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = WinProbLSTM(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


def live_mask(sequences: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """(N, T) bool: real timesteps where both teams still have someone alive."""
    T = sequences.shape[1]
    valid = np.arange(T)[None, :] < lengths[:, None]
    return valid & (sequences[:, :, 0] > 0) & (sequences[:, :, 1] > 0)


def predict_all_steps(
    model: WinProbLSTM,
    sequences: np.ndarray,
    lengths: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """(N, T) win probability after every timestep (padding positions are meaningless)."""
    out = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            x = torch.tensor(sequences[i : i + batch_size], dtype=torch.float32, device=device)
            l = torch.tensor(lengths[i : i + batch_size], dtype=torch.long, device=device)
            out.append(model.forward_all(x, l).cpu().numpy())
    return np.concatenate(out)


def live_metrics(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    p = np.clip(probs, 1e-7, 1 - 1e-7)
    return {
        "accuracy": float(((p >= 0.5) == labels.astype(bool)).mean()),
        "log_loss": float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))),
        "brier": float(np.mean((p - labels) ** 2)),
        "n_states": int(len(labels)),
    }


def plot_calibration(preds: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers, actual_rates, counts = [], [], []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (preds >= lo) & (preds < hi)
        if mask.sum() == 0:
            continue
        bin_centers.append(preds[mask].mean())
        actual_rates.append(labels[mask].mean())
        counts.append(mask.sum())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    size_scale = 300 / max(counts)
    ax.scatter(bin_centers, actual_rates, s=[c * size_scale for c in counts], zorder=3, label="Model")
    ax.plot(bin_centers, actual_rates, alpha=0.6)
    ax.set_xlabel("Predicted attacker win probability")
    ax.set_ylabel("Actual attacker win rate")
    ax.set_title("Calibration (test matches, live states)")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved calibration curve → %s", out_path)


def plot_timestep_accuracy(
    probs: np.ndarray, labels: np.ndarray, live: np.ndarray, out_path: Path
) -> None:
    """Accuracy at each kill-event index, over live states only."""
    ts, accs = [], []
    for t in range(live.shape[1]):
        m = live[:, t]
        if m.sum() < 30:  # too few rounds to say anything
            continue
        ts.append(t)
        accs.append(((probs[m, t] >= 0.5) == labels[m].astype(bool)).mean())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ts, accs, marker="o")
    ax.axhline(0.5, color="gray", linestyle="--", label="Chance")
    ax.set_xlabel("Kill events so far (0 = round start)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs. Kills Elapsed (live states)")
    ax.set_ylim(0.4, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved timestep accuracy → %s", out_path)


def plot_trajectories(
    probs: np.ndarray,
    labels: np.ndarray,
    lengths: np.ndarray,
    out_path: Path,
    n_samples: int = 6,
) -> None:
    """Win probability trajectory for a handful of sample rounds."""
    interesting = np.where(lengths >= 4)[0]
    rng = np.random.default_rng(0)
    chosen = rng.choice(interesting, size=min(n_samples, len(interesting)), replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, idx in zip(axes.flatten(), chosen):
        length = int(lengths[idx])
        label = int(labels[idx])
        color = "#2ecc71" if label == 1 else "#e74c3c"
        outcome = "Attackers won" if label == 1 else "Defenders won"
        ax.plot(range(length), probs[idx, :length], marker="o", color=color)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Kills elapsed")
        ax.set_ylabel("Attacker win prob")
        ax.set_title(outcome, color=color)

    fig.suptitle("Win Probability Trajectories (test matches)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved trajectories → %s", out_path)


def plot_feature_importance(
    sequences: np.ndarray,
    labels: np.ndarray,
    lengths: np.ndarray,
    live: np.ndarray,
    model: WinProbLSTM,
    device: torch.device,
    out_path: Path,
) -> None:
    """Permutation importance: shuffle each feature across rounds, measure live-state log-loss increase."""
    step_labels = np.broadcast_to(labels[:, None], live.shape)[live]
    base = live_metrics(predict_all_steps(model, sequences, lengths, device)[live], step_labels)["log_loss"]

    rng = np.random.default_rng(42)
    importances = []
    for feat_idx in range(sequences.shape[2]):
        permuted = sequences.copy()
        permuted[:, :, feat_idx] = permuted[rng.permutation(len(permuted)), :, feat_idx]
        probs = predict_all_steps(model, permuted, lengths, device)[live]
        importances.append(live_metrics(probs, step_labels)["log_loss"] - base)

    order = np.argsort(importances)
    sorted_vals = [importances[i] for i in order]
    colors = ["#e74c3c" if v > 0 else "#95a5a6" for v in sorted_vals]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([FEATURE_NAMES[i] for i in order], sorted_vals, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Log-loss increase when feature is shuffled")
    ax.set_title("Permutation Feature Importance (live states)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved feature importance → %s", out_path)


def evaluate(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    out_dir: str | Path = "artifacts",
) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(checkpoint_path, device)
    logger.info("Loaded model from %s", checkpoint_path)

    data = np.load(dataset_path)
    _, _, test_idx = match_split(data["match_indices"], cfg.val_split, cfg.test_split, cfg.seed)
    sequences = data["sequences"][test_idx]
    labels = data["labels"][test_idx]
    lengths = data["lengths"][test_idx]

    probs = predict_all_steps(model, sequences, lengths, device)
    live = live_mask(sequences, lengths)
    step_labels = np.broadcast_to(labels[:, None], live.shape)[live]

    metrics = live_metrics(probs[live], step_labels)
    start = live_metrics(probs[:, 0], labels)
    logger.info(
        "Live states (n=%d): accuracy=%.4f | log-loss=%.4f | brier=%.4f",
        metrics["n_states"], metrics["accuracy"], metrics["log_loss"], metrics["brier"],
    )
    logger.info("Round start only: accuracy=%.4f | log-loss=%.4f", start["accuracy"], start["log_loss"])

    plot_calibration(probs[live], step_labels, out_dir / "calibration_curve.png")
    plot_timestep_accuracy(probs, labels, live, out_dir / "timestep_accuracy.png")
    plot_trajectories(probs, labels, lengths, out_dir / "trajectories.png")
    plot_feature_importance(sequences, labels, lengths, live, model, device, out_dir / "feature_importance.png")

    logger.info("Evaluation complete. Plots saved to %s/", out_dir)
    return metrics
