"""Tests for the situation breakdown."""

import numpy as np

from src.features.sequences import _snapshot_to_vector
from src.training.situations import SITUATIONS, situation_table


def _state(atk=5, dfn=5, planted=0, elapsed=0):
    return _snapshot_to_vector({
        "attackers_alive": atk, "defenders_alive": dfn, "spike_planted": planted,
        "time_elapsed_ms": elapsed, "time_remaining_ms": 0,
        "plant_time_ms": 30_000 if planted else 0,
        "attacker_economy": 20_000, "defender_economy": 20_000, "round_num": 0,
        "attacker_score": 0, "defender_score": 0, "plant_site": "A" if planted else "",
    })


def _masks(states):
    f = np.array(states, dtype=np.float32)
    return {name: fn(f) for name, fn in SITUATIONS.items()}


def test_masks_decode_features():
    m = _masks([
        _state(),                                        # round start, even
        _state(atk=1, dfn=1, planted=1, elapsed=70_000), # 1v1 post-plant
        _state(atk=3, dfn=1, elapsed=40_000),            # attackers up, 1vX, pre-plant mid
        _state(atk=2, dfn=4, elapsed=65_000),            # defenders up, pre-plant late
    ])
    assert m["Round start"].tolist() == [True, False, False, False]
    assert m["Post-plant"].tolist() == [False, True, False, False]
    assert m["1v1"].tolist() == [False, True, False, False]
    assert m["1vX clutch"].tolist() == [False, False, True, False]
    assert m["Attackers up"].tolist() == [False, False, True, False]
    assert m["Defenders up"].tolist() == [False, False, False, True]
    assert m["Pre-plant, 30-60s"].tolist() == [False, False, True, False]
    assert m["Pre-plant, 60s+"].tolist() == [False, False, False, True]


def test_man_advantage_buckets_partition_all_states():
    rng = np.random.default_rng(0)
    states = [_state(atk=int(a), dfn=int(d)) for a, d in rng.integers(1, 6, size=(200, 2))]
    m = _masks(states)
    total = m["Even numbers"].astype(int) + m["Attackers up"] + m["Defenders up"]
    assert (total == 1).all()


def test_table_point_estimates_and_ci():
    rng = np.random.default_rng(1)
    n = 2000
    feats = np.array([_state(atk=int(rng.integers(1, 6)), dfn=int(rng.integers(1, 6))) for _ in range(n)], dtype=np.float32)
    labels = rng.integers(0, 2, size=n).astype(float)
    good = np.clip(labels * 0.7 + 0.15 + rng.normal(0, 0.05, n), 0.01, 0.99)
    bad = np.full(n, 0.5)
    matches = rng.integers(0, 100, size=n)

    rows = situation_table(feats, labels, {"good": good, "bad": bad}, matches,
                           primary="good", compare="bad", n_boot=200)
    all_row = rows[0]
    assert all_row["situation"] == "All live states"
    assert all_row["n"] == n
    assert all_row["acc:good"] == ((good >= 0.5) == labels.astype(bool)).mean()
    lo, hi = all_row["acc_ci"]
    assert lo <= all_row["acc:good"] <= hi
    assert all_row["dll"] < 0 and all_row["dll_ci"][1] < 0  # good model clearly better
