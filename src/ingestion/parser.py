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
plant_time_ms       int   ms into round when spike was planted (0 if not planted)
attacker_economy    int   total starting loadout value for attacking team
defender_economy    int   total starting loadout value for defending team
round_num           int   0-indexed round number
attacker_score      int   rounds won by attacking team so far (before this round)
defender_score      int   rounds won by defending team so far (before this round)
plant_site          str   "A", "B", "C", or "" (empty if not planted)
attacker_won        int   ground-truth label: 1 if attackers won, 0 otherwise

Timing constants (milliseconds)
--------------------------------
ROUND_LENGTH_MS    = 100_000   standard round timer (buy phase excluded)
DEFUSE_TIMER_MS    =  45_000   defuse window after spike plant

Attack-side assignment
-----------------------
Standard map: Blue is attacker for rounds 0-11, Red for rounds 12+.
Overtime alternates every 2 rounds starting at round 24.
We derive this from the `teams` array in the match JSON where possible.
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


def _attacker_team_id(teams: list[dict], round_num: int) -> str:
    """
    Determine which teamId ('Blue' or 'Red') is attacking in the given round.
    Sides swap at round 12 and every 2 rounds in overtime (round 24+).
    """
    # Find which team started as Blue (attacker in round 0)
    # In standard Valorant: Blue attacks first.
    # After round 11, sides swap.
    swaps = round_num // 12  # 0 = first half, 1 = second half, 2+ = overtime pairs
    if swaps == 0:
        return "Blue"
    elif swaps == 1:
        return "Red"
    else:
        # Overtime: each pair of rounds alternates starting from round 24
        ot_round = round_num - 24
        ot_pair = ot_round // 2
        return "Blue" if ot_pair % 2 == 0 else "Red"


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
    if not winning_team:
        return []

    attacker_won = 1 if winning_team == attacker_team else 0

    # Economy from round start
    economies = round_data.get("playerEconomies") or []
    atk_eco, def_eco = _economy_by_team(economies, puuid_to_team, attacker_team)

    # Spike plant info
    plant_time_ms: int = round_data.get("plantRoundTime") or 0
    spike_planted_at: int | None = plant_time_ms if plant_time_ms > 0 else None
    plant_site: str = round_data.get("plantSite") or ""

    # Kill timeline
    all_kills = _kill_timeline(round_data.get("playerStats", []))

    # Track alive counts — start with 5v5
    atk_alive = 5
    def_alive = 5
    snapshots: list[RoundSnapshot] = []

    def time_remaining(elapsed_ms: int) -> int:
        if spike_planted_at is not None and elapsed_ms >= spike_planted_at:
            elapsed_since_plant = elapsed_ms - spike_planted_at
            return max(0, DEFUSE_TIMER_MS - elapsed_since_plant)
        return max(0, ROUND_LENGTH_MS - elapsed_ms)

    def make_snapshot(elapsed_ms: int) -> RoundSnapshot:
        planted = 1 if (spike_planted_at is not None and elapsed_ms >= spike_planted_at) else 0
        pt = spike_planted_at if spike_planted_at is not None else 0
        return RoundSnapshot(
            attackers_alive=atk_alive,
            defenders_alive=def_alive,
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
        victim_team = puuid_to_team.get(victim_puuid, "")

        if victim_team == attacker_team:
            atk_alive = max(0, atk_alive - 1)
        elif victim_team:
            def_alive = max(0, def_alive - 1)

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
    attacker_score = 0
    defender_score = 0

    for round_data in round_results:
        round_num: int = round_data.get("roundNum", len(rounds))
        attacker_team = _attacker_team_id(teams, round_num)

        snapshots = parse_round(
            round_data=round_data,
            round_num=round_num,
            attacker_team=attacker_team,
            puuid_to_team=puuid_to_team,
            attacker_score=attacker_score,
            defender_score=defender_score,
        )

        if snapshots:
            rounds.append(snapshots)
            # Update scores for next round
            if snapshots[-1].attacker_won:
                attacker_score += 1
            else:
                defender_score += 1

    return rounds


def parse_match_file(path: Path) -> list[list[dict]]:
    """Load a saved match JSON file and return parsed snapshots as plain dicts."""
    match_data = json.loads(path.read_text(encoding="utf-8"))
    rounds = parse_match(match_data)
    return [[snap.to_dict() for snap in rnd] for rnd in rounds]
