"""
Parse raw Riot API match JSON into in-round temporal sequences.

Each round is converted into a list of state snapshots — one per kill event
plus an initial snapshot at round start.  Each snapshot is the feature vector
fed as one timestep to the LSTM/Transformer, and carries the ground-truth label
(did the attacking team win this round?).

Snapshot feature schema
-----------------------
attackers_alive     int   0-5
defenders_alive     int   0-5
spike_planted       int   0 or 1
time_elapsed_ms     int   milliseconds elapsed since round start
time_remaining_ms   int   ms remaining (uses defuse clock once planted)
plant_time_ms       int   ms into round when spike was planted (0 until planted)
attacker_economy    int   total starting loadout value for attacking team
defender_economy    int   total starting loadout value for defending team
round_num           int   0-indexed round number
attacker_score      int   rounds the currently attacking team has won so far
defender_score      int   rounds the currently defending team has won so far
plant_site          str   "A", "B", "C", or "" (empty if not planted)
attacker_won        int   ground-truth label: 1 if attackers won, 0 otherwise

Timing constants (milliseconds)
--------------------------------
ROUND_LENGTH_MS    = 100_000   standard round timer (buy phase excluded)
DEFUSE_TIMER_MS    =  45_000   defuse window after spike plant

Attack-side assignment
-----------------------
Red attacks rounds 0-11, Blue attacks rounds 12-23.  In overtime (round 24+)
sides swap every round: Red attacks even rounds, Blue attacks odd rounds.
Verified against `plant_events.planted_by.team` in the raw Henrik data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

ROUND_LENGTH_MS = 100_000
DEFUSE_TIMER_MS = 45_000


@dataclass
class RoundSnapshot:
    attackers_alive: int
    defenders_alive: int
    spike_planted: int
    time_elapsed_ms: int
    time_remaining_ms: int
    plant_time_ms: int
    attacker_economy: int
    defender_economy: int
    round_num: int
    attacker_score: int
    defender_score: int
    plant_site: str
    attacker_won: int

    def to_dict(self) -> dict:
        return asdict(self)


def _attacker_team_id(teams: list[dict] | None, round_num: int) -> str:
    """
    Determine which teamId ('Red' or 'Blue') is attacking in the given round.

    Red attacks the first half, Blue the second; in overtime sides swap
    every round, starting with Red on round 24.
    """
    if round_num < 12:
        return "Red"
    if round_num < 24:
        return "Blue"
    return "Red" if (round_num - 24) % 2 == 0 else "Blue"


def _other_team(team: str) -> str:
    return "Blue" if team == "Red" else "Red"


def _round_end_ms(plant_time_ms: int | None, defuse_time_ms: int | None) -> int | None:
    """
    Latest moment the round can still be undecided: the defuse, or else the
    spike detonating 45s after plant. Kills at or after this point are
    spike-blast or post-round kills; the outcome is already known, so their
    snapshots must be dropped. (The API labels rounds where the blast kills
    the last players as "Eliminated", so end_type can't be trusted here.)
    """
    if defuse_time_ms:
        return defuse_time_ms
    if plant_time_ms:
        return plant_time_ms + DEFUSE_TIMER_MS
    return None


class _AliveTracker:
    """
    Alive counts that survive revives (Sage, Clove, ...). A player who dies a
    second time must have been revived in between, so their second death
    doesn't reduce the count again. The API doesn't log revive times, so a
    revived player is still counted dead until they die again.
    """

    def __init__(self, attacker_team: str) -> None:
        self.attacker_team = attacker_team
        self.dead: dict[str, set[str]] = {"atk": set(), "def": set()}

    def record_death(self, victim_puuid: str, victim_team: str) -> None:
        if not victim_team:
            return
        side = "atk" if victim_team == self.attacker_team else "def"
        self.dead[side].add(victim_puuid)

    @property
    def atk_alive(self) -> int:
        return max(0, 5 - len(self.dead["atk"]))

    @property
    def def_alive(self) -> int:
        return max(0, 5 - len(self.dead["def"]))


def _economy_by_team(
    player_economies: list[dict],
    puuid_to_team: dict[str, str],
    attacker_team: str,
) -> tuple[int, int]:
    """Sum loadout values for attackers and defenders from playerEconomies."""
    atk_val = 0
    def_val = 0
    for eco in player_economies:
        team = puuid_to_team.get(eco.get("puuid", ""), "")
        val = eco.get("loadoutValue", 0)
        if team == attacker_team:
            atk_val += val
        else:
            def_val += val
    return atk_val, def_val


def _kill_timeline(player_stats: list[dict]) -> list[dict]:
    """Collect all kills across all players, sorted by round time."""
    kills: list[dict] = []
    for ps in player_stats:
        for kill in ps.get("kills", []):
            kills.append(kill)
    return sorted(kills, key=lambda k: k.get("roundTime", 0))


def parse_round(
    round_data: dict,
    round_num: int,
    attacker_team: str,
    puuid_to_team: dict[str, str],
    attacker_score: int,
    defender_score: int,
) -> list[RoundSnapshot]:
    """
    Convert one round's data into a list of RoundSnapshot objects.
    Returns an empty list if the round lacks usable data.
    """
    winning_team = round_data.get("winningTeam", "")
    # Surrendered rounds have no in-round outcome to learn from.
    if not winning_team or round_data.get("roundResult") == "Surrendered":
        return []

    attacker_won = 1 if winning_team == attacker_team else 0

    # Economy from round start
    economies = round_data.get("playerEconomies") or []
    atk_eco, def_eco = _economy_by_team(economies, puuid_to_team, attacker_team)

    # Spike plant info
    plant_time_ms: int = round_data.get("plantRoundTime") or 0
    spike_planted_at: int | None = plant_time_ms if plant_time_ms > 0 else None
    plant_site: str = round_data.get("plantSite") or ""

    # Kill timeline, cut off when the spike decided the round
    end_ms = _round_end_ms(
        spike_planted_at,
        round_data.get("defuseRoundTime"),
    )
    all_kills = [
        k for k in _kill_timeline(round_data.get("playerStats", []))
        if end_ms is None or k.get("roundTime", 0) < end_ms
    ]

    alive = _AliveTracker(attacker_team)
    snapshots: list[RoundSnapshot] = []

    def time_remaining(elapsed_ms: int) -> int:
        if spike_planted_at is not None and elapsed_ms >= spike_planted_at:
            elapsed_since_plant = elapsed_ms - spike_planted_at
            return max(0, DEFUSE_TIMER_MS - elapsed_since_plant)
        return max(0, ROUND_LENGTH_MS - elapsed_ms)

    def make_snapshot(elapsed_ms: int) -> RoundSnapshot:
        planted = 1 if (spike_planted_at is not None and elapsed_ms >= spike_planted_at) else 0
        # Only expose plant time once the plant has happened — otherwise
        # pre-plant snapshots would leak that a plant is coming.
        pt = spike_planted_at if planted else 0
        return RoundSnapshot(
            attackers_alive=alive.atk_alive,
            defenders_alive=alive.def_alive,
            spike_planted=planted,
            time_elapsed_ms=elapsed_ms,
            time_remaining_ms=time_remaining(elapsed_ms),
            plant_time_ms=pt,
            attacker_economy=atk_eco,
            defender_economy=def_eco,
            round_num=round_num,
            attacker_score=attacker_score,
            defender_score=defender_score,
            plant_site=plant_site if planted else "",
            attacker_won=attacker_won,
        )

    # Snapshot at t=0 (round start)
    snapshots.append(make_snapshot(0))

    for kill in all_kills:
        round_time = kill.get("roundTime", 0)
        victim_puuid = kill.get("victim", "")
        alive.record_death(victim_puuid, puuid_to_team.get(victim_puuid, ""))
        snapshots.append(make_snapshot(round_time))

    return snapshots


def parse_match(match_data: dict) -> list[list[RoundSnapshot]]:
    """
    Parse a full match JSON into a list of rounds, each a list of snapshots.
    Rounds with fewer than 2 snapshots (no kills) are kept — the model sees
    the initial state even in quick eliminations.
    """
    players: list[dict] = match_data.get("players", [])
    puuid_to_team: dict[str, str] = {p["puuid"]: p["teamId"] for p in players}

    teams: list[dict] = match_data.get("teams", [])
    round_results: list[dict] = match_data.get("roundResults", [])

    rounds: list[list[RoundSnapshot]] = []
    # Scores are tracked per team, then mapped to whoever is attacking this
    # round — sides swap at halftime, teams don't.
    team_wins = {"Red": 0, "Blue": 0}

    for round_data in round_results:
        round_num: int = round_data.get("roundNum", len(rounds))
        attacker_team = _attacker_team_id(teams, round_num)

        snapshots = parse_round(
            round_data=round_data,
            round_num=round_num,
            attacker_team=attacker_team,
            puuid_to_team=puuid_to_team,
            attacker_score=team_wins[attacker_team],
            defender_score=team_wins[_other_team(attacker_team)],
        )

        if snapshots:
            rounds.append(snapshots)
            winner = attacker_team if snapshots[-1].attacker_won else _other_team(attacker_team)
            team_wins[winner] += 1

    return rounds


def parse_match_file(path: Path) -> list[list[dict]]:
    """
    Load a saved match JSON file and return parsed snapshots as plain dicts.
    Auto-detects Henrik vs official Riot format based on JSON structure.
    """
    match_data = json.loads(path.read_text(encoding="utf-8"))
    # Henrik format has 'metadata' + 'rounds'; official has 'matchInfo' + 'roundResults'
    if "metadata" in match_data and "rounds" in match_data:
        rounds = parse_henrik_match(match_data)
    else:
        rounds = parse_match(match_data)
    return [[snap.to_dict() for snap in rnd] for rnd in rounds]


# ---------------------------------------------------------------------------
# Henrik API v3 format parser
# ---------------------------------------------------------------------------

def _henrik_attacker_team(round_num: int) -> str:
    """Same side-swap logic as _attacker_team_id, for the Henrik format."""
    return _attacker_team_id(None, round_num)


def _parse_henrik_round(
    round_data: dict,
    round_num: int,
    attacker_team: str,
    attacker_score: int,
    defender_score: int,
) -> list[RoundSnapshot]:
    winning_team = round_data.get("winning_team", "")
    # Surrendered rounds have no in-round outcome to learn from.
    if not winning_team or round_data.get("end_type") == "Surrendered":
        return []

    attacker_won = 1 if winning_team == attacker_team else 0

    # Economy — sum loadout values per team for this round
    atk_eco = 0
    def_eco = 0
    for ps in round_data.get("player_stats", []):
        val = (ps.get("economy") or {}).get("loadout_value", 0)
        team = ps.get("player_team", "")
        if team == attacker_team:
            atk_eco += val
        else:
            def_eco += val

    # Spike plant info
    plant_events = round_data.get("plant_events") or {}
    plant_time_ms: int = plant_events.get("plant_time_in_round") or 0
    spike_planted_at: int | None = plant_time_ms if plant_time_ms > 0 else None
    plant_site: str = plant_events.get("plant_site") or ""

    # Kill timeline — collect from all players, sort by round time, and
    # drop post-round kills after a detonation or defuse
    end_ms = _round_end_ms(
        spike_planted_at,
        (round_data.get("defuse_events") or {}).get("defuse_time_in_round"),
    )
    all_kills: list[dict] = []
    for ps in round_data.get("player_stats", []):
        for kill in ps.get("kill_events", []):
            t = kill.get("kill_time_in_round", 0)
            if end_ms is not None and t >= end_ms:
                continue
            all_kills.append({
                "round_time": t,
                "victim_puuid": kill.get("victim_puuid", ""),
                "victim_team": kill.get("victim_team", ""),
            })
    all_kills.sort(key=lambda k: k["round_time"])

    alive = _AliveTracker(attacker_team)
    snapshots: list[RoundSnapshot] = []

    def time_remaining(elapsed_ms: int) -> int:
        if spike_planted_at is not None and elapsed_ms >= spike_planted_at:
            return max(0, DEFUSE_TIMER_MS - (elapsed_ms - spike_planted_at))
        return max(0, ROUND_LENGTH_MS - elapsed_ms)

    def make_snapshot(elapsed_ms: int) -> RoundSnapshot:
        planted = 1 if (spike_planted_at is not None and elapsed_ms >= spike_planted_at) else 0
        # Only expose plant time once the plant has happened — otherwise
        # pre-plant snapshots would leak that a plant is coming.
        pt = spike_planted_at if planted else 0
        return RoundSnapshot(
            attackers_alive=alive.atk_alive,
            defenders_alive=alive.def_alive,
            spike_planted=planted,
            time_elapsed_ms=elapsed_ms,
            time_remaining_ms=time_remaining(elapsed_ms),
            plant_time_ms=pt,
            attacker_economy=atk_eco,
            defender_economy=def_eco,
            round_num=round_num,
            attacker_score=attacker_score,
            defender_score=defender_score,
            plant_site=plant_site if planted else "",
            attacker_won=attacker_won,
        )

    snapshots.append(make_snapshot(0))

    for kill in all_kills:
        alive.record_death(kill["victim_puuid"], kill["victim_team"])
        snapshots.append(make_snapshot(kill["round_time"]))

    return snapshots


def parse_henrik_match(match_data: dict) -> list[list[RoundSnapshot]]:
    """Parse a Henrik API v3 match dict into round snapshot sequences."""
    rounds_raw = match_data.get("rounds", [])
    rounds: list[list[RoundSnapshot]] = []
    # Scores are tracked per team, then mapped to whoever is attacking this
    # round — sides swap at halftime, teams don't.
    team_wins = {"Red": 0, "Blue": 0}

    for round_num, round_data in enumerate(rounds_raw):
        attacker_team = _henrik_attacker_team(round_num)
        snapshots = _parse_henrik_round(
            round_data=round_data,
            round_num=round_num,
            attacker_team=attacker_team,
            attacker_score=team_wins[attacker_team],
            defender_score=team_wins[_other_team(attacker_team)],
        )
        if snapshots:
            rounds.append(snapshots)
            winner = attacker_team if snapshots[-1].attacker_won else _other_team(attacker_team)
            team_wins[winner] += 1

    return rounds
