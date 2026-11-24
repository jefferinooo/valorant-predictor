"""
Turn the round so far plus the model's probabilities into the small set of
facts the commentary is allowed to use.

Everything the LLM may say comes from this dict, and the grounding checker
(grounding.py) verifies the output against the same dict.
"""

from __future__ import annotations


def _event(prev: dict | None, cur: dict) -> str:
    if prev is None:
        return "round_start"
    if not prev["spike_planted"] and cur["spike_planted"]:
        return "spike_planted"
    if cur["attackers_alive"] < prev["attackers_alive"]:
        return "attacker_killed"
    if cur["defenders_alive"] < prev["defenders_alive"]:
        return "defender_killed"
    return "update"


def build_facts(snapshots: list[dict], probs: list[float]) -> dict:
    """
    snapshots: the round so far (API Snapshot dicts), oldest first.
    probs: attacker win probability after each snapshot (same length).
    """
    cur = snapshots[-1]
    prev = snapshots[-2] if len(snapshots) > 1 else None
    atk_pct = int(round(float(probs[-1]) * 100))
    facts = {
        "event": _event(prev, cur),
        "attackers_alive": int(cur["attackers_alive"]),
        "defenders_alive": int(cur["defenders_alive"]),
        "spike_planted": bool(cur["spike_planted"]),
        "plant_site": (cur.get("plant_site") or None) if cur["spike_planted"] else None,
        "seconds_left": int(round(cur["time_remaining_ms"] / 1000)),
        "round_number": int(cur["round_num"]) + 1,
        "attacker_score": int(cur["attacker_score"]),
        "defender_score": int(cur["defender_score"]),
        "attacker_win_pct": atk_pct,
        "defender_win_pct": 100 - atk_pct,
        "attacker_win_pct_change": (
            atk_pct - int(round(float(probs[-2]) * 100)) if len(probs) > 1 else None
        ),
    }
    return facts


def template_commentary(facts: dict) -> str:
    """Deterministic fallback line: used when the LLM is off, fails, or isn't grounded."""
    a, d = facts["attackers_alive"], facts["defenders_alive"]
    atk = facts["attacker_win_pct"]
    lead = {
        "round_start": "Round underway",
        "attacker_killed": "Attacker down",
        "defender_killed": "Defender down",
        "spike_planted": f"Spike planted on {facts['plant_site']}" if facts["plant_site"] else "Spike planted",
    }.get(facts["event"], "Update")
    return f"{lead}: {a}v{d}, attackers at {atk}% to win the round."
