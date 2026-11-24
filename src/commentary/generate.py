"""
Generate one line of caster-style commentary with Claude, grounded in the
win-probability model's output.

The model only sees the facts dict from facts.build_facts. The line is then
checked by grounding.check_grounding; if it fails (or the API call fails),
callers fall back to facts.template_commentary, so the user never sees an
ungrounded line.
"""

from __future__ import annotations

import json
import os

import anthropic

from src.commentary.facts import template_commentary
from src.commentary.grounding import check_grounding

MODEL = os.environ.get("COMMENTARY_MODEL", "claude-opus-5-5")

SYSTEM_PROMPT = """\
You write live commentary for a Valorant win-probability overlay, in the voice of an esports caster.

You receive the current round state as JSON. Write one sentence of at most 25 words reacting to the latest event and what it does to the attackers' chances.

Use only the facts in the JSON:
- Don't invent player names, agents, weapons, abilities, map locations or anything else not given.
- Any number you use must appear in the facts: a win percentage, its change, a player count, seconds left or the score.
- Write player counts as XvY with attackers first, e.g. 4v3.
- Mention the spike, a plant, a site or a defuse only if spike_planted is true. A site must be plant_site. plant_site is where the spike is; the facts don't say where kills happen, so don't place a kill at a site.
- If you mention the score, use the current attacker_score-defender_score. Don't predict future scores or say how many rounds a team is from winning.
- attacker_win_pct_change is the change in percentage points since the previous event (null at round start).

Reply with one final sentence only: no quotes, preamble, drafts, corrections or notes."""


def generate_commentary(facts: dict, client: anthropic.Anthropic, model: str = MODEL) -> tuple[str | None, dict]:
    """
    Returns (text or None, usage dict). None means Claude declined; callers
    should use the template line.
    """
    response = client.beta.messages.create(
        model=model,
        max_tokens=1024,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # re-run on a fallback model if this one declines
        output_config={"effort": "low"},  # a one-line reaction doesn't need deep reasoning
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(facts, sort_keys=True)}],
    )
    usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
    if response.stop_reason == "refusal":
        return None, usage
    text = "".join(b.text for b in response.content if b.type == "text").strip().strip('"')
    return text or None, usage


def grounded_commentary(facts: dict, client: anthropic.Anthropic | None) -> dict:
    """
    What the API serves: an LLM line if it passes the grounding checks,
    otherwise the deterministic template.
    """
    if client is None:
        return {"commentary": template_commentary(facts), "source": "template", "rejected": []}
    try:
        text, _ = generate_commentary(facts, client)
    except anthropic.APIError as e:
        return {"commentary": template_commentary(facts), "source": "template",
                "rejected": [f"api error: {type(e).__name__}"]}
    issues = check_grounding(text, facts)
    if issues:
        return {"commentary": template_commentary(facts), "source": "template", "rejected": issues}
    return {"commentary": text, "source": "llm", "rejected": []}
