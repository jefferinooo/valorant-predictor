"""Tests for feature engineering."""

import numpy as np
import pytest
from src.features.sequences import _snapshot_to_vector, FEATURE_DIM


def _make_snapshot(**kwargs) -> dict:
    defaults = {
        "attackers_alive": 5,
        "defenders_alive": 5,
        "spike_planted": 0,
        "time_elapsed_ms": 0,
        "time_remaining_ms": 100_000,
        "plant_time_ms": 0,
        "attacker_economy": 20_000,
        "defender_economy": 20_000,
        "round_num": 0,
        "attacker_score": 0,
        "defender_score": 0,
        "plant_site": "",
        "attacker_won": 1,
    }
    defaults.update(kwargs)
    return defaults


def test_feature_vector_length():
    snap = _make_snapshot()
    vec = _snapshot_to_vector(snap)
    assert len(vec) == FEATURE_DIM


def test_feature_values_in_range():
    """All features should be in [-1, 1]."""
    snap = _make_snapshot()
    vec = _snapshot_to_vector(snap)
    for v in vec:
        assert -1.0 <= v <= 1.0, f"Feature out of range: {v}"


def test_full_team_alive_normalizes_to_one():
    snap = _make_snapshot(attackers_alive=5, defenders_alive=5)
    vec = _snapshot_to_vector(snap)
    assert vec[0] == pytest.approx(1.0)
    assert vec[1] == pytest.approx(1.0)


def test_empty_team_normalizes_to_zero():
    snap = _make_snapshot(attackers_alive=0, defenders_alive=0)
    vec = _snapshot_to_vector(snap)
    assert vec[0] == pytest.approx(0.0)
    assert vec[1] == pytest.approx(0.0)


def test_spike_planted_flag():
    snap_no = _make_snapshot(spike_planted=0)
    snap_yes = _make_snapshot(spike_planted=1)
    assert _snapshot_to_vector(snap_no)[2] == pytest.approx(0.0)
    assert _snapshot_to_vector(snap_yes)[2] == pytest.approx(1.0)


def test_plant_site_one_hot():
    for site, expected_idx in [("A", 11), ("B", 12), ("C", 13)]:
        snap = _make_snapshot(plant_site=site)
        vec = _snapshot_to_vector(snap)
        assert vec[expected_idx] == pytest.approx(1.0)
        # Other site flags should be 0
        for other_idx in {11, 12, 13} - {expected_idx}:
            assert vec[other_idx] == pytest.approx(0.0)


def test_no_plant_site_all_zeros():
    snap = _make_snapshot(plant_site="")
    vec = _snapshot_to_vector(snap)
    assert vec[11] == pytest.approx(0.0)
    assert vec[12] == pytest.approx(0.0)
    assert vec[13] == pytest.approx(0.0)


def test_score_diff_sign():
    snap_atk_lead = _make_snapshot(attacker_score=5, defender_score=2)
    snap_def_lead = _make_snapshot(attacker_score=2, defender_score=5)
    vec_atk = _snapshot_to_vector(snap_atk_lead)
    vec_def = _snapshot_to_vector(snap_def_lead)
    assert vec_atk[14] > 0   # attackers leading → positive score_diff
    assert vec_def[14] < 0   # defenders leading → negative score_diff
