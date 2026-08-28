"""Tests for the Henrik match parser."""

import pytest
from src.ingestion.parser import parse_henrik_match, _henrik_attacker_team

# ---------------------------------------------------------------------------
# Minimal fake match fixture
# ---------------------------------------------------------------------------

def _make_player(puuid: str, team: str, loadout: int = 3900) -> dict:
    return {
        "puuid": puuid,
        "name": puuid,
        "tag": "TEST",
        "team": team,
    }


def _make_kill(killer_team: str, victim_team: str, round_time: int) -> dict:
    return {
        "kill_time_in_round": round_time,
        "killer_puuid": "killer",
        "killer_team": killer_team,
        "victim_puuid": "victim",
        "victim_team": victim_team,
    }


def _make_round(
    winning_team: str,
    kills: list[dict],
    bomb_planted: bool = False,
    plant_time: int = 0,
    plant_site: str | None = None,
    player_teams: list[str] | None = None,
) -> dict:
    if player_teams is None:
        player_teams = ["Blue"] * 5 + ["Red"] * 5

    player_stats = []
    for i, team in enumerate(player_teams):
        player_stats.append({
            "player_puuid": f"player_{i}",
            "player_team": team,
            "kill_events": [],
            "economy": {"loadout_value": 3900},
        })

    # Assign kills to player_stats of the killer
    for kill in kills:
        player_stats[0]["kill_events"].append(kill)

    plant_events = {
        "plant_time_in_round": plant_time if bomb_planted else None,
        "plant_site": plant_site,
    }

    return {
        "winning_team": winning_team,
        "end_type": "Eliminated",
        "bomb_planted": bomb_planted,
        "bomb_defused": False,
        "plant_events": plant_events,
        "defuse_events": {},
        "player_stats": player_stats,
    }


def _make_match(rounds: list[dict]) -> dict:
    return {
        "metadata": {"matchid": "test-match-1", "map": "Ascent", "mode": "Competitive"},
        "players": {
            "all_players": [
                _make_player(f"player_{i}", "Blue" if i < 5 else "Red")
                for i in range(10)
            ]
        },
        "teams": {
            "blue": {"has_won": True, "rounds_won": 1},
            "red": {"has_won": False, "rounds_won": 0},
        },
        "rounds": rounds,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_attacker_team_assignment():
    assert _henrik_attacker_team(0) == "Blue"
    assert _henrik_attacker_team(11) == "Blue"
    assert _henrik_attacker_team(12) == "Red"
    assert _henrik_attacker_team(23) == "Red"
    assert _henrik_attacker_team(24) == "Blue"
    assert _henrik_attacker_team(25) == "Blue"
    assert _henrik_attacker_team(26) == "Red"


def test_parse_simple_round_no_kills():
    """A round with no kills still produces one snapshot (round start)."""
    match = _make_match([_make_round("Blue", kills=[])])
    rounds = parse_henrik_match(match)
    assert len(rounds) == 1
    assert len(rounds[0]) == 1  # only the t=0 snapshot
    snap = rounds[0][0]
    assert snap.attackers_alive == 5
    assert snap.defenders_alive == 5
    assert snap.spike_planted == 0
    assert snap.attacker_won == 1  # Blue wins, Blue attacks round 0


def test_parse_round_with_kills():
    """Kills reduce alive counts correctly."""
    kills = [
        _make_kill("Blue", "Red", round_time=5000),
        _make_kill("Red", "Blue", round_time=15000),
    ]
    match = _make_match([_make_round("Blue", kills=kills)])
    rounds = parse_henrik_match(match)
    snapshots = rounds[0]

    # t=0, t=5000, t=15000
    assert len(snapshots) == 3

    assert snapshots[0].attackers_alive == 5
    assert snapshots[0].defenders_alive == 5

    assert snapshots[1].defenders_alive == 4  # Red player killed
    assert snapshots[1].attackers_alive == 5

    assert snapshots[2].attackers_alive == 4  # Blue player killed
    assert snapshots[2].defenders_alive == 4


def test_parse_round_spike_planted():
    """Spike plant flag and site are captured correctly."""
    match = _make_match([
        _make_round("Blue", kills=[], bomb_planted=True, plant_time=30000, plant_site="B")
    ])
    rounds = parse_henrik_match(match)
    snap = rounds[0][0]
    assert snap.spike_planted == 0  # t=0 is before plant
    assert snap.plant_time_ms == 30000
    assert snap.plant_site == ""  # site only shown after plant


def test_scores_accumulate_across_rounds():
    """attacker_score and defender_score increment correctly round to round."""
    rounds_data = [
        _make_round("Blue", kills=[]),   # round 0: attackers win
        _make_round("Red", kills=[]),    # round 1: defenders win
        _make_round("Blue", kills=[]),   # round 2: attackers win
    ]
    match = _make_match(rounds_data)
    rounds = parse_henrik_match(match)

    assert rounds[0][0].attacker_score == 0
    assert rounds[0][0].defender_score == 0

    assert rounds[1][0].attacker_score == 1
    assert rounds[1][0].defender_score == 0

    assert rounds[2][0].attacker_score == 1
    assert rounds[2][0].defender_score == 1


def test_parse_match_returns_all_rounds():
    match = _make_match([
        _make_round("Blue", kills=[]),
        _make_round("Red", kills=[]),
        _make_round("Blue", kills=[]),
    ])
    rounds = parse_henrik_match(match)
    assert len(rounds) == 3


def test_attacker_won_label():
    """attacker_won is 1 when attacker team wins, 0 otherwise."""
    match = _make_match([
        _make_round("Blue", kills=[]),  # round 0: Blue attacks, Blue wins → 1
        _make_round("Red", kills=[]),   # round 1: Blue attacks, Red wins → 0
    ])
    rounds = parse_henrik_match(match)
    assert rounds[0][-1].attacker_won == 1
    assert rounds[1][-1].attacker_won == 0
