"""
Feature engineering: convert raw match JSON files into padded numpy arrays.

Each round becomes one training example — a sequence of timestep vectors.
One vector per kill event (plus the initial round-start snapshot), each
normalized to [0, 1] so the model trains stably.

Feature vector (15 dimensions per timestep)
--------------------------------------------
0   attackers_alive   / 5
1   defenders_alive   / 5
2   spike_planted       (0 or 1)
3   time_elapsed_ms   / 100_000
4   time_remaining_ms / 100_000
5   plant_time_ms     / 100_000
6   attacker_economy  / 40_000   (5 players x ~8k max loadout)
7   defender_economy  / 40_000
8   round_num         / 24
9   attacker_score    / 13
10  defender_score    / 13
11  plant_site_A        (one-hot)
12  plant_site_B        (one-hot)
13  plant_site_C        (one-hot)
14  score_diff        / 13       (attacker_score - defender_score, signed)

Output arrays (saved to data/processed/dataset.npz)
----------------------------------------------------
sequences  float32  (N, T_max, 15)  padded feature matrices
labels     float32  (N,)            1.0 if attackers won, 0.0 otherwise
lengths    int32    (N,)            actual number of timesteps per round
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.ingestion.parser import parse_match_file

logger = logging.getLogger(__name__)

FEATURE_DIM = 15


def _snapshot_to_vector(snap: dict) -> list[float]:
    site = snap.get("plant_site", "")
    return [
        snap["attackers_alive"] / 5.0,
        snap["defenders_alive"] / 5.0,
        float(snap["spike_planted"]),
        snap["time_elapsed_ms"] / 100_000.0,
        snap["time_remaining_ms"] / 100_000.0,
        snap["plant_time_ms"] / 100_000.0,
        snap["attacker_economy"] / 40_000.0,
        snap["defender_economy"] / 40_000.0,
        snap["round_num"] / 24.0,
        snap["attacker_score"] / 13.0,
        snap["defender_score"] / 13.0,
        1.0 if site == "A" else 0.0,
        1.0 if site == "B" else 0.0,
        1.0 if site == "C" else 0.0,
        (snap["attacker_score"] - snap["defender_score"]) / 13.0,
    ]


def build_dataset(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all match files, parse them, and return padded arrays.

    Returns
    -------
    sequences : float32 (N, T_max, 15)
    labels    : float32 (N,)
    lengths   : int32   (N,)
    """
    match_dir = Path(data_dir) / "raw" / "matches"
    files = sorted(match_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No match files found in {match_dir}")

    all_sequences: list[np.ndarray] = []
    all_labels: list[float] = []
    all_lengths: list[int] = []

    skipped = 0
    for f in files:
        try:
            rounds = parse_match_file(f)
        except Exception as exc:
            logger.warning("Skipping %s: %s", f.name, exc)
            skipped += 1
            continue

        for round_snaps in rounds:
            if not round_snaps:
                continue
            vectors = [_snapshot_to_vector(s) for s in round_snaps]
            seq = np.array(vectors, dtype=np.float32)
            label = float(round_snaps[-1]["attacker_won"])
            all_sequences.append(seq)
            all_labels.append(label)
            all_lengths.append(len(seq))

    if not all_sequences:
        raise ValueError("No valid rounds found across all match files.")

    logger.info(
        "Loaded %d rounds from %d matches (%d skipped)",
        len(all_sequences), len(files) - skipped, skipped,
    )

    t_max = max(all_lengths)
    n = len(all_sequences)

    padded = np.zeros((n, t_max, FEATURE_DIM), dtype=np.float32)
    for i, seq in enumerate(all_sequences):
        padded[i, : len(seq)] = seq

    labels = np.array(all_labels, dtype=np.float32)
    lengths = np.array(all_lengths, dtype=np.int32)

    return padded, labels, lengths


def save_dataset(data_dir: str | Path, out_path: str | Path | None = None) -> Path:
    """Build the dataset and save to .npz. Returns the output path."""
    data_dir = Path(data_dir)
    if out_path is None:
        out_path = data_dir / "processed" / "dataset.npz"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sequences, labels, lengths = build_dataset(data_dir)
    np.savez_compressed(out_path, sequences=sequences, labels=labels, lengths=lengths)

    logger.info(
        "Saved dataset: %d rounds, T_max=%d, features=%d -> %s",
        len(labels), sequences.shape[1], sequences.shape[2], out_path,
    )
    return out_path


def load_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a previously saved dataset.npz."""
    data = np.load(path)
    return data["sequences"], data["labels"], data["lengths"]
