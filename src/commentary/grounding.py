"""
Deterministic grounding checks for a commentary line.

Each check looks for a specific kind of claim in the text (a percentage, a
player count, a site, a plant) and verifies it against the facts the line
was generated from. Returns a list of human-readable issues; an empty list
means the line passed.

These checks are deliberately conservative: they only verify claims they can
parse. Anything else (tone, invented player names) is left to the LLM judge
in scripts/eval_commentary.py.
"""

from __future__ import annotations

import re

MAX_WORDS = 40

_PCT = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*(?:%|percent\b)", re.IGNORECASE)
_VS = re.compile(r"\b([0-5])\s*(?:v|vs\.?|versus|-on-|on)\s*([0-5])\b", re.IGNORECASE)
# Upper-case site letters only, so the article "a" isn't read as site A.
_SITE = re.compile(r"\b([ABC])[\s-]+site\b|\bsite\s+([ABC])\b|\b(?:on|at)\s+([ABC])\b")
# Scores like "11-2" or "4–4". Must be the current score (either order).
_SCORE = re.compile(r"\b(\d{1,2})\s*[-–]\s*(\d{1,2})\b")
# A kill placed at a site ("down on B", "drops at A"). The facts never say where kills happen.
_KILL_AT_SITE = re.compile(
    r"\b(?:down|drops?|falls?|killed|dies|picked off|eliminated)\s+(?:on|at|in)\s+(?:site\s+)?[ABC]\b",
    re.IGNORECASE,
)
# Claims that the spike is already down (or being defused).
_PLANT_CLAIM = re.compile(
    r"\b(planted|spike(?:'s| is)? down|post-plant|defus\w*|retake)\b", re.IGNORECASE
)


_FIRST_BLOOD = re.compile(r"\bfirst blood\b", re.IGNORECASE)
_NEGATION = re.compile(r"\b(no|not|without|yet to|before|never)\b", re.IGNORECASE)


def _clause_before(text: str, pos: int) -> str:
    """Text from the start of the current clause up to pos."""
    head = text[:pos]
    cut = max(head.rfind(c) for c in ",.;!?—")
    return head[cut + 1 :]


def check_grounding(text: str | None, facts: dict) -> list[str]:
    issues: list[str] = []
    if not text or not text.strip():
        return ["empty commentary"]
    if "\n" in text.strip():
        issues.append("more than one line")
    if len(text.split()) > MAX_WORDS:
        issues.append(f"longer than {MAX_WORDS} words")

    # Percentages must be the attacker %, defender %, or the size of the change.
    allowed = {facts["attacker_win_pct"], facts["defender_win_pct"]}
    if facts.get("attacker_win_pct_change") is not None:
        allowed.add(abs(facts["attacker_win_pct_change"]))
    for m in _PCT.finditer(text):
        val = float(m.group(1))
        if not any(abs(val - a) <= 1 for a in allowed):
            issues.append(f"percentage {m.group(0)!r} not in facts {sorted(allowed)}")

    # Player counts like "4v3" must match the alive counts (either order).
    a, d = facts["attackers_alive"], facts["defenders_alive"]
    for m in _VS.finditer(text):
        pair = (int(m.group(1)), int(m.group(2)))
        if pair not in {(a, d), (d, a)}:
            issues.append(f"player count {m.group(0)!r} doesn't match {a}v{d}")

    # Sites only exist once the spike is planted, and must be the right one.
    for m in _SITE.finditer(text):
        letter = next(g for g in m.groups() if g)
        if not facts["spike_planted"]:
            issues.append(f"mentions site {letter} before any plant")
        elif facts.get("plant_site") and letter != facts["plant_site"]:
            issues.append(f"mentions site {letter}, but the plant is on {facts['plant_site']}")

    # Scores must be the current one: no projected or invented scores.
    score = (facts["attacker_score"], facts["defender_score"])
    for m in _SCORE.finditer(text):
        pair = (int(m.group(1)), int(m.group(2)))
        if pair not in {score, score[::-1]}:
            issues.append(f"score {m.group(0)!r} isn't the current score {score[0]}-{score[1]}")

    for m in _KILL_AT_SITE.finditer(text):
        # "Spike down on B" / "Spike goes down on A" is about the spike, not a kill
        if re.search(r"\b(spike|bomb)\b", _clause_before(text, m.start()), re.IGNORECASE):
            continue
        issues.append(f"says where a kill happened ({m.group(0)!r}), which the facts don't include")

    # "First blood" is the round's first kill: exactly 9 players left alive.
    if _FIRST_BLOOD.search(text) and a + d != 9:
        issues.append(f"says 'first blood' but {a}v{d} isn't the round's first kill")

    if not facts["spike_planted"]:
        for m in _PLANT_CLAIM.finditer(text):
            # "no spike down", "without a plant", "yet to get the spike down" are fine pre-plant
            if _NEGATION.search(_clause_before(text, m.start())):
                continue
            issues.append(f"claims {m.group(0)!r} but the spike isn't planted")
            break

    return issues
