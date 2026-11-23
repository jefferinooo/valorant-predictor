"""
LSTM model and training loop for in-round win probability prediction.

Architecture
------------
Input:  (batch, T_max, 15) padded sequences
        → nn.LSTM (packed, so padding is ignored)
        → nn.Linear(hidden_dim, 1) applied at EVERY timestep
        → sigmoid → win probability after each kill event

Training
--------
- 70/15/15 train/val/test split by match; val picks the checkpoint,
  test is used only for final reporting (src/training/evaluate.py)
- Supervise every prefix of the round, not just the final state.  The last
  snapshot of a round usually shows one side fully eliminated, which gives
  the label away; a model scored only on that state looks ~99% accurate
  while learning almost nothing about live prediction.
- Validation metrics are computed on *live* states only: snapshots where
  both teams still have at least one player alive.
- BCE loss, Adam optimiser, best checkpoint by live-state validation loss
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from src.training.config import TrainConfig

logger = logging.getLogger(__name__)


class RoundDataset(Dataset):
    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        lengths: np.ndarray,
    ) -> None:
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.lengths = torch.tensor(lengths, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.labels[idx], self.lengths[idx]


class WinProbLSTM(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(cfg.hidden_dim, 1)

    def forward_all(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Win probability after every timestep: (batch, T). Padding positions are garbage."""
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        return torch.sigmoid(self.head(out)).squeeze(-1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Win probability at the last real timestep of each sequence: (batch,)."""
        probs = self.forward_all(x, lengths)
        return probs[torch.arange(len(lengths), device=probs.device), lengths - 1]


def match_split(
    match_indices: np.ndarray, val_split: float, test_split: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (train_idx, val_idx, test_idx) round indices, with every match
    entirely in one set. Validation picks the checkpoint; test is touched
    only for final reporting.
    """
    unique_matches = np.unique(match_indices)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_matches)
    n_val = int(len(unique_matches) * val_split)
    n_test = int(len(unique_matches) * test_split)
    val_mask = np.isin(match_indices, unique_matches[:n_val])
    test_mask = np.isin(match_indices, unique_matches[n_val : n_val + n_test])
    train_mask = ~(val_mask | test_mask)
    return np.where(train_mask)[0], np.where(val_mask)[0], np.where(test_mask)[0]


def step_masks(seqs: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    valid: (B, T) real (non-padding) timesteps.
    live:  valid timesteps where both teams still have someone alive, i.e. the
           round is genuinely undecided by player count. Headline metrics use these.
    """
    T = seqs.shape[1]
    valid = torch.arange(T, device=seqs.device)[None, :] < lengths[:, None]
    live = valid & (seqs[:, :, 0] > 0) & (seqs[:, :, 1] > 0)
    return valid, live


def _collate(batch):
    seqs, labels, lengths = zip(*batch)
    # Sort descending by length (required by pack_padded_sequence)
    order = torch.argsort(torch.stack(lengths), descending=True)
    return (
        torch.stack(seqs)[order],
        torch.stack(labels)[order],
        torch.stack(lengths)[order],
    )


def train(cfg: TrainConfig, dataset_path: str | Path) -> None:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    data = np.load(dataset_path)
    sequences     = data["sequences"]      # (N, T_max, 15)
    labels        = data["labels"]         # (N,)
    lengths       = data["lengths"]        # (N,)
    match_indices = data["match_indices"]  # (N,)

    train_idx, val_idx, test_idx = match_split(
        match_indices, cfg.val_split, cfg.test_split, cfg.seed
    )
    logger.info(
        "Split: %d train / %d val / %d test rounds (%d / %d / %d matches); test is held out",
        len(train_idx), len(val_idx), len(test_idx),
        *(len(np.unique(match_indices[i])) for i in (train_idx, val_idx, test_idx)),
    )

    train_ds = RoundDataset(sequences[train_idx], labels[train_idx], lengths[train_idx])
    val_ds   = RoundDataset(sequences[val_idx],   labels[val_idx],   lengths[val_idx])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  collate_fn=_collate)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, collate_fn=_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s", device)

    model = WinProbLSTM(cfg).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    bce = nn.BCELoss(reduction="none")

    Path(cfg.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        # --- train: loss on every real timestep of every round ---
        model.train()
        train_loss, train_steps = 0.0, 0
        for seqs, lbls, lens in train_loader:
            seqs, lbls, lens = seqs.to(device), lbls.to(device), lens.to(device)
            valid, _ = step_masks(seqs, lens)
            optimiser.zero_grad()
            probs = model.forward_all(seqs, lens)
            targets = lbls[:, None].expand_as(probs)
            loss = bce(probs[valid], targets[valid]).mean()
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * valid.sum().item()
            train_steps += valid.sum().item()
        train_loss /= train_steps

        # --- validate on live (undecided) states only ---
        model.eval()
        val_loss, correct, n_live = 0.0, 0, 0
        with torch.no_grad():
            for seqs, lbls, lens in val_loader:
                seqs, lbls, lens = seqs.to(device), lbls.to(device), lens.to(device)
                _, live = step_masks(seqs, lens)
                probs = model.forward_all(seqs, lens)[live]
                targets = lbls[:, None].expand(-1, seqs.shape[1])[live]
                val_loss += bce(probs, targets).sum().item()
                correct += ((probs >= 0.5) == targets.bool()).sum().item()
                n_live += live.sum().item()
        val_loss /= n_live
        val_acc = correct / n_live

        logger.info(
            "Epoch %2d/%d | train_loss=%.4f | live val_loss=%.4f | live val_acc=%.3f",
            epoch, cfg.epochs, train_loss, val_loss, val_acc,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": model.state_dict(), "config": cfg}, cfg.checkpoint_path)
            logger.info("  Saved new best checkpoint (val_loss=%.4f)", best_val_loss)

    logger.info("Training complete. Best val_loss=%.4f", best_val_loss)
