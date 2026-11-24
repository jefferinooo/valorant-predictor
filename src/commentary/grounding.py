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
# Claims that the spike is already down (or being defused).
_PLANT_CLAIM = re.compile(
    r"\b(planted|spike(?:'s| is)? down|post-plant|defus\w*|retake)\b", re.IGNORECASE
)


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

    if not facts["spike_planted"]:
        m = _PLANT_CLAIM.search(text)
        if m:
            issues.append(f"claims {m.group(0)!r} but the spike isn't planted")

    return issues
