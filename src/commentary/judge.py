"""LLM judge for commentary quality, used by scripts/eval_commentary.py."""

from __future__ import annotations

import json
import os

import anthropic

JUDGE_MODEL = os.environ.get("COMMENTARY_JUDGE_MODEL", "claude-opus-5-5")

JUDGE_SYSTEM_PROMPT = """\
You grade one line of live Valorant commentary written for a win-probability overlay.

You get the facts the writer was given (JSON) and the line it wrote. Judge only against those facts.

- faithful: true if every claim in the line is supported by the facts. Invented player names, agents, weapons, abilities, locations, numbers, or events not in the facts make it unfaithful. Reasonable caster framing ("big pick", "momentum") is fine.
- unsupported_claims: each claim that isn't supported, quoted briefly. Empty if faithful.
- reacts_to_event: true if the line reacts to the latest event in the facts.
- tone: 1-5, how much it sounds like a natural, energetic esports caster (5 = broadcast-ready, 1 = robotic or wrong register).
- notes: one short sentence explaining the grade."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "faithful": {"type": "boolean"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "reacts_to_event": {"type": "boolean"},
        "tone": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "notes": {"type": "string"},
    },
    "required": ["faithful", "unsupported_claims", "reacts_to_event", "tone", "notes"],
    "additionalProperties": False,
}


def judge_line(facts: dict, line: str, client: anthropic.Anthropic) -> tuple[dict | None, dict]:
    """Returns (grade dict or None if the judge declined, usage dict)."""
    response = client.beta.messages.create(
        model=JUDGE_MODEL,
        max_tokens=4096,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Facts:\n{json.dumps(facts, sort_keys=True, indent=2)}\n\nLine:\n{line}",
        }],
    )
    usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
    if response.stop_reason == "refusal":
        return None, usage
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text), usage
