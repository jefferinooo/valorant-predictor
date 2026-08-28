"""
Evaluation for the LSTM win probability model.

Produces three outputs saved to artifacts/:
  1. calibration_curve.png  — predicted probability vs actual win rate
  2. timestep_accuracy.png  — how accuracy improves as kills accumulate
  3. trajectories.png       — win probability over time for sample rounds
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
from src.training.train import WinProbLSTM, RoundDataset, _collate
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[WinProbLSTM, TrainConfig]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = WinProbLSTM(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


def run_inference(
    model: WinProbLSTM,
    sequences: np.ndarray,
    labels: np.ndarray,
    lengths: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (predictions, labels) as numpy arrays."""
    ds = RoundDataset(sequences, labels, lengths)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for seqs, lbls, lens in loader:
            preds = model(seqs.to(device), lens.to(device))
            all_preds.append(preds.cpu().numpy())
            all_labels.append(lbls.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def plot_calibration(preds: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers, actual_rates, counts = [], [], []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (preds >= lo) & (preds < hi)
        if mask.sum() == 0:
            continue
        bin_centers.append((lo + hi) / 2)
        actual_rates.append(labels[mask].mean())
        counts.append(mask.sum())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.scatter(bin_centers, actual_rates, s=[c / 5 for c in counts], zorder=3, label="Model")
    ax.plot(bin_centers, actual_rates, alpha=0.6)
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Actual win rate")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved calibration curve → %s", out_path)


def plot_timestep_accuracy(
    sequences: np.ndarray,
    labels: np.ndarray,
    lengths: np.ndarray,
    model: WinProbLSTM,
    device: torch.device,
    out_path: Path,
) -> None:
    """Accuracy at each position in the sequence (kill index)."""
    max_t = sequences.shape[1]
    accs = []

    model.eval()
    with torch.no_grad():
        for t in range(1, max_t + 1):
            mask = lengths >= t
            if mask.sum() == 0:
                accs.append(None)
                continue
            sub_seq = sequences[mask, :t, :]
            sub_len = np.minimum(lengths[mask], t)
            sub_lbl = labels[mask]

            # Pad back to t timesteps (already correct size here)
            padded = np.zeros((mask.sum(), t, sequences.shape[2]), dtype=np.float32)
            for i, (s, l) in enumerate(zip(sub_seq, sub_len)):
                padded[i, :l] = s[:l]

            preds = model(
                torch.tensor(padded).to(device),
                torch.tensor(sub_len).to(device),
            ).cpu().numpy()
            acc = ((preds >= 0.5) == sub_lbl.astype(bool)).mean()
            accs.append((t, acc, mask.sum()))

    ts = [x[0] for x in accs if x]
    acc_vals = [x[1] for x in accs if x]
    ns = [x[2] for x in accs if x]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ts, acc_vals, marker="o")
    ax.axhline(0.5, color="gray", linestyle="--", label="Chance")
    ax.set_xlabel("Kill event index (timestep)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs. Kill Event Index")
    ax.set_ylim(0.4, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved timestep accuracy → %s", out_path)


def plot_trajectories(
    sequences: np.ndarray,
    labels: np.ndarray,
    lengths: np.ndarray,
    model: WinProbLSTM,
    device: torch.device,
    out_path: Path,
    n_samples: int = 6,
) -> None:
    """Win probability trajectory for a handful of sample rounds."""
    # Pick rounds with at least 4 timesteps for an interesting trajectory
    interesting = np.where(lengths >= 4)[0]
    rng = np.random.default_rng(0)
    chosen = rng.choice(interesting, size=min(n_samples, len(interesting)), replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    model.eval()
    with torch.no_grad():
        for ax, idx in zip(axes, chosen):
            seq = sequences[idx]
            length = int(lengths[idx])
            label = int(labels[idx])
            probs = []

            for t in range(1, length + 1):
                sub = torch.tensor(seq[:t][None], dtype=torch.float32).to(device)
                l = torch.tensor([t], dtype=torch.long).to(device)
                p = model(sub, l).item()
                probs.append(p)

            color = "#2ecc71" if label == 1 else "#e74c3c"
            outcome = "Attackers won" if label == 1 else "Defenders won"
            ax.plot(range(1, length + 1), probs, marker="o", color=color)
            ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Kill #")
            ax.set_ylabel("Attacker win prob")
            ax.set_title(outcome, color=color)

    fig.suptitle("Win Probability Trajectories", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved trajectories → %s", out_path)


def evaluate(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    out_dir: str | Path = "artifacts",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(checkpoint_path, device)
    logger.info("Loaded model from %s", checkpoint_path)

    data = np.load(dataset_path)
    sequences = data["sequences"]
    labels = data["labels"]
    lengths = data["lengths"]

    preds, lbls = run_inference(model, sequences, labels, lengths, device)

    acc = ((preds >= 0.5) == lbls.astype(bool)).mean()
    bce = -np.mean(lbls * np.log(preds + 1e-8) + (1 - lbls) * np.log(1 - preds + 1e-8))
    logger.info("Overall accuracy: %.4f | log-loss: %.4f", acc, bce)

    plot_calibration(preds, lbls, out_dir / "calibration_curve.png")
    plot_timestep_accuracy(sequences, labels, lengths, model, device, out_dir / "timestep_accuracy.png")
    plot_trajectories(sequences, labels, lengths, model, device, out_dir / "trajectories.png")

    logger.info("Evaluation complete. Plots saved to %s/", out_dir)
