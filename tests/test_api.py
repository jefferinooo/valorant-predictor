"""Tests for the FastAPI serving endpoint."""

import pytest
from fastapi.testclient import TestClient

# Import the app — this will load the model at import time.
# Tests are skipped if the checkpoint doesn't exist.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

CHECKPOINT = Path("artifacts/models/best.pt")

if not CHECKPOINT.exists():
    pytest.skip("Model checkpoint not found — run scripts/train_model.py first",
                allow_module_level=True)

from scripts.serve import app

client = TestClient(app)


def _snapshot(**kwargs) -> dict:
    defaults = {
        "attackers_alive": 5,
        "defenders_alive": 5,
        "spike_planted": 0,
        "time_elapsed_ms": 0,
        "time_remaining_ms": 100000,
        "plant_time_ms": 0,
        "attacker_economy": 15000,
        "defender_economy": 20000,
        "round_num": 5,
        "attacker_score": 3,
        "defender_score": 2,
        "plant_site": "",
    }
    defaults.update(kwargs)
    return defaults


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_single_snapshot():
    resp = client.post("/predict", json={"snapshots": [_snapshot()]})
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["win_probability"] <= 1.0
    assert len(body["per_timestep"]) == 1


def test_predict_multi_snapshot():
    snapshots = [
        _snapshot(attackers_alive=5, defenders_alive=5, time_elapsed_ms=0),
        _snapshot(attackers_alive=4, defenders_alive=5, time_elapsed_ms=12000),
        _snapshot(attackers_alive=4, defenders_alive=4, spike_planted=1,
                  time_elapsed_ms=35000, time_remaining_ms=45000, plant_site="B"),
    ]
    resp = client.post("/predict", json={"snapshots": snapshots})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["per_timestep"]) == 3
    assert all(0.0 <= p <= 1.0 for p in body["per_timestep"])


def test_predict_win_prob_matches_last_timestep():
    snapshots = [_snapshot(), _snapshot(attackers_alive=3)]
    resp = client.post("/predict", json={"snapshots": snapshots})
    body = resp.json()
    assert body["win_probability"] == body["per_timestep"][-1]


def test_predict_rejects_empty_snapshots():
    resp = client.post("/predict", json={"snapshots": []})
    assert resp.status_code == 422


def test_predict_rejects_invalid_alive_count():
    resp = client.post("/predict", json={"snapshots": [_snapshot(attackers_alive=6)]})
    assert resp.status_code == 422


def test_predict_rejects_invalid_plant_site():
    resp = client.post("/predict", json={"snapshots": [_snapshot(plant_site="D")]})
    assert resp.status_code == 422


def test_predict_attacker_advantage_higher_prob():
    """5v1 should give higher win prob than 1v5."""
    resp_5v1 = client.post("/predict", json={"snapshots": [
        _snapshot(attackers_alive=5, defenders_alive=1)
    ]})
    resp_1v5 = client.post("/predict", json={"snapshots": [
        _snapshot(attackers_alive=1, defenders_alive=5)
    ]})
    prob_5v1 = resp_5v1.json()["win_probability"]
    prob_1v5 = resp_1v5.json()["win_probability"]
    assert prob_5v1 > prob_1v5
