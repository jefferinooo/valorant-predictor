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


_victim_counter = 0


def _make_kill(killer_team: str, victim_team: str, round_time: int, victim_puuid: str | None = None) -> dict:
    global _victim_counter
    if victim_puuid is None:  # distinct victim per kill unless specified
        _victim_counter += 1
        victim_puuid = f"victim_{_victim_counter}"
    return {
        "kill_time_in_round": round_time,
        "killer_puuid": "killer",
        "killer_team": killer_team,
        "victim_puuid": victim_puuid,
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
    """Red attacks first half, Blue second half, OT swaps every round."""
    assert _henrik_attacker_team(0) == "Red"
    assert _henrik_attacker_team(11) == "Red"
    assert _henrik_attacker_team(12) == "Blue"
    assert _henrik_attacker_team(23) == "Blue"
    assert _henrik_attacker_team(24) == "Red"
    assert _henrik_attacker_team(25) == "Blue"
    assert _henrik_attacker_team(26) == "Red"


def test_parse_simple_round_no_kills():
    """A round with no kills still produces one snapshot (round start)."""
    match = _make_match([_make_round("Red", kills=[])])
    rounds = parse_henrik_match(match)
    assert len(rounds) == 1
    assert len(rounds[0]) == 1  # only the t=0 snapshot
    snap = rounds[0][0]
    assert snap.attackers_alive == 5
    assert snap.defenders_alive == 5
    assert snap.spike_planted == 0
    assert snap.attacker_won == 1  # Red wins, Red attacks round 0


def test_parse_round_with_kills():
    """Kills reduce alive counts correctly."""
    kills = [
        _make_kill("Red", "Blue", round_time=5000),
        _make_kill("Blue", "Red", round_time=15000),
    ]
    match = _make_match([_make_round("Red", kills=kills)])
    rounds = parse_henrik_match(match)
    snapshots = rounds[0]

    # t=0, t=5000, t=15000
    assert len(snapshots) == 3

    assert snapshots[0].attackers_alive == 5
    assert snapshots[0].defenders_alive == 5

    assert snapshots[1].defenders_alive == 4  # Blue (defender) killed
    assert snapshots[1].attackers_alive == 5

    assert snapshots[2].attackers_alive == 4  # Red (attacker) killed
    assert snapshots[2].defenders_alive == 4


def test_parse_round_spike_planted():
    """Plant flag, time and site only appear once the plant has happened."""
    kills = [_make_kill("Red", "Blue", round_time=10000), _make_kill("Red", "Blue", round_time=40000)]
    match = _make_match([
        _make_round("Red", kills=kills, bomb_planted=True, plant_time=30000, plant_site="B")
    ])
    snaps = parse_henrik_match(match)[0]
    before, pre_plant_kill, after = snaps
    for snap in (before, pre_plant_kill):
        assert snap.spike_planted == 0
        assert snap.plant_time_ms == 0  # must not leak the future plant
        assert snap.plant_site == ""
    assert after.spike_planted == 1
    assert after.plant_time_ms == 30000
    assert after.plant_site == "B"


def test_scores_accumulate_across_rounds():
    """attacker_score and defender_score increment correctly round to round."""
    rounds_data = [
        _make_round("Red", kills=[]),    # round 0: attackers win
        _make_round("Blue", kills=[]),   # round 1: defenders win
        _make_round("Red", kills=[]),    # round 2: attackers win
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
        _make_round("Red", kills=[]),   # round 0: Red attacks, Red wins → 1
        _make_round("Blue", kills=[]),  # round 1: Red attacks, Blue wins → 0
    ])
    rounds = parse_henrik_match(match)
    assert rounds[0][-1].attacker_won == 1
    assert rounds[1][-1].attacker_won == 0


def test_surrendered_rounds_are_skipped():
    surrendered = _make_round("Blue", kills=[])
    surrendered["end_type"] = "Surrendered"
    match = _make_match([_make_round("Red", kills=[]), surrendered])
    rounds = parse_henrik_match(match)
    assert len(rounds) == 1


def test_revived_player_not_double_counted():
    """A player who dies twice was revived; alive count drops only once."""
    kills = [
        _make_kill("Blue", "Red", 5000, victim_puuid="sage_target"),
        _make_kill("Blue", "Red", 20000, victim_puuid="sage_target"),
    ]
    snaps = parse_henrik_match(_make_match([_make_round("Blue", kills=kills)]))[0]
    assert [s.attackers_alive for s in snaps] == [5, 4, 4]


def test_post_detonation_kills_dropped():
    """Kills after the spike detonates happen post-round and must not create snapshots."""
    rnd = _make_round(
        "Red",
        kills=[_make_kill("Red", "Blue", 10000), _make_kill("Red", "Blue", 30000 + 45000 + 2000)],
        bomb_planted=True, plant_time=30000, plant_site="A",
    )
    rnd["end_type"] = "Bomb detonated"
    snaps = parse_henrik_match(_make_match([rnd]))[0]
    assert [s.time_elapsed_ms for s in snaps] == [0, 10000]


def test_post_defuse_kills_dropped():
    rnd = _make_round(
        "Blue",
        kills=[_make_kill("Blue", "Red", 40000), _make_kill("Red", "Blue", 61000)],
        bomb_planted=True, plant_time=30000, plant_site="A",
    )
    rnd["end_type"] = "Bomb defused"
    rnd["defuse_events"] = {"defuse_time_in_round": 60000}
    snaps = parse_henrik_match(_make_match([rnd]))[0]
    assert [s.time_elapsed_ms for s in snaps] == [0, 40000]


def test_scores_follow_teams_across_halftime():
    """After the side swap, attacker_score is the new attacking team's total."""
    # Red attacks rounds 0-11; Red wins 8, Blue wins 4
    rounds = [_make_round("Red" if i < 8 else "Blue", kills=[]) for i in range(12)]
    rounds.append(_make_round("Blue", kills=[]))  # round 12: Blue attacks
    parsed = parse_henrik_match(_make_match(rounds))
    r12 = parsed[12][0]
    assert r12.attacker_score == 4  # Blue's wins
    assert r12.defender_score == 8  # Red's wins


def test_spike_blast_kills_dropped_even_if_labelled_eliminated():
    """Spike blast kills come at/after plant+45s; the round is decided at detonation."""
    rnd = _make_round(
        "Red",
        kills=[_make_kill("Red", "Blue", 10000), _make_kill("Red", "Blue", 30000 + 45000 + 300)],
        bomb_planted=True, plant_time=30000, plant_site="A",
    )
    rnd["end_type"] = "Eliminated"
    snaps = parse_henrik_match(_make_match([rnd]))[0]
    assert [s.time_elapsed_ms for s in snaps] == [0, 10000]
