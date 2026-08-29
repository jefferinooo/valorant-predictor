"""
LSTM model and training loop for in-round win probability prediction.

Architecture
------------
Input:  (batch, T_max, 15) padded sequences
        → nn.LSTM (packed, so padding is ignored)
        → take hidden state at last real timestep
        → nn.Linear(hidden_dim, 1)
        → sigmoid → win probability

Training
--------
- Split by match index (not round) to prevent data leakage
- BCELoss, Adam optimiser
- Save best checkpoint by validation loss
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
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

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Pack so LSTM ignores padding
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        # hidden: (num_layers, batch, hidden_dim) — take last layer
        last_hidden = hidden[-1]
        return torch.sigmoid(self.head(last_hidden)).squeeze(-1)


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

    # Split by match to prevent data leakage.
    # All rounds from a given match go entirely into train OR val — never both.
    unique_matches = np.unique(match_indices)
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(unique_matches)
    n_val_matches = int(len(unique_matches) * cfg.val_split)
    val_match_set = set(unique_matches[:n_val_matches].tolist())

    train_mask = np.array([m not in val_match_set for m in match_indices])
    val_mask   = ~train_mask
    train_idx  = np.where(train_mask)[0]
    val_idx    = np.where(val_mask)[0]

    logger.info(
        "Split: %d train rounds (%d matches) | %d val rounds (%d matches)",
        train_mask.sum(), len(unique_matches) - n_val_matches,
        val_mask.sum(), n_val_matches,
    )

    train_ds = RoundDataset(sequences[train_idx], labels[train_idx], lengths[train_idx])
    val_ds   = RoundDataset(sequences[val_idx],   labels[val_idx],   lengths[val_idx])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  collate_fn=_collate)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, collate_fn=_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s", device)

    model = WinProbLSTM(cfg).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.BCELoss()

    Path(cfg.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for seqs, lbls, lens in train_loader:
            seqs, lbls, lens = seqs.to(device), lbls.to(device), lens.to(device)
            optimiser.zero_grad()
            preds = model(seqs, lens)
            loss = criterion(preds, lbls)
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * len(lbls)
        train_loss /= len(train_ds)

        # --- validate ---
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for seqs, lbls, lens in val_loader:
                seqs, lbls, lens = seqs.to(device), lbls.to(device), lens.to(device)
                preds = model(seqs, lens)
                val_loss += criterion(preds, lbls).item() * len(lbls)
                correct += ((preds >= 0.5) == lbls.bool()).sum().item()
        val_loss /= len(val_ds)
        val_acc = correct / len(val_ds)

        logger.info(
            "Epoch %2d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.3f",
            epoch, cfg.epochs, train_loss, val_loss, val_acc,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": model.state_dict(), "config": cfg}, cfg.checkpoint_path)
            logger.info("  Saved new best checkpoint (val_loss=%.4f)", best_val_loss)

    logger.info("Training complete. Best val_loss=%.4f", best_val_loss)
