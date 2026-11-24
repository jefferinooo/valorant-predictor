"""
Evaluate LLM commentary on real moments from the held-out test matches.

For each sampled moment (a live state partway through a test round):
  1. run the win-probability model on the round so far
  2. build the facts and generate a commentary line with Claude
  3. score it with the deterministic grounding checks (src/commentary/grounding.py)
  4. grade it with an LLM judge for faithfulness, relevance and tone (src/commentary/judge.py)

Writes artifacts/commentary_eval.md.

Run (needs an Anthropic API key; costs roughly $1-2 for 200 samples with the defaults):
    python3 scripts/eval_commentary.py --n 200
    python3 scripts/eval_commentary.py --dry-run     # no API calls: exercises the pipeline with the template line
"""

from __future__ import annotations

import argparse
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.commentary.facts import build_facts, template_commentary
from src.commentary.grounding import check_grounding
from src.features.sequences import _snapshot_to_vector
from src.ingestion.parser import parse_match_file
from src.serving.numpy_lstm import NumpyLSTM
from src.training.config import TrainConfig
from src.training.train import match_split

ROOT = Path(__file__).parent.parent
# $ per million tokens (input, output); update if you change models
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}


def sample_moments(n: int, seed: int) -> list[list[dict]]:
    """Random live prefixes (both teams alive) of rounds from the test matches."""
    data = np.load(ROOT / "data/processed/dataset.npz")
    cfg = TrainConfig()
    _, _, test_idx = match_split(data["match_indices"], cfg.val_split, cfg.test_split, cfg.seed)
    test_matches = set(data["match_indices"][test_idx].tolist())

    # Same file order and match numbering as src/features/sequences.build_dataset
    rounds: list[list[dict]] = []
    match_idx = 0
    for f in sorted((ROOT / "data/raw/matches").glob("*.json")):
        try:
            parsed = parse_match_file(f)
        except Exception:
            continue
        if match_idx in test_matches:
            rounds.extend(r for r in parsed if r)
        match_idx += 1

    rng = random.Random(seed)
    moments = []
    while len(moments) < n:
        rnd = rng.choice(rounds)
        live = [t for t, s in enumerate(rnd) if s["attackers_alive"] > 0 and s["defenders_alive"] > 0]
        t = rng.choice(live)
        moments.append(rnd[: t + 1])
    return moments


def evaluate_one(prefix: list[dict], model: NumpyLSTM, client, judge: bool) -> dict:
    probs = model.predict_all([_snapshot_to_vector(s) for s in prefix])
    facts = build_facts(prefix, list(probs))
    usage = []
    if client is None:
        line = template_commentary(facts)
    else:
        from src.commentary.generate import generate_commentary
        line, u = generate_commentary(facts, client)
        usage.append(("gen", u))
    issues = check_grounding(line, facts)
    grade = None
    if judge and client is not None and line:
        from src.commentary.judge import judge_line
        grade, u = judge_line(facts, line, client)
        usage.append(("judge", u))
    return {"facts": facts, "line": line, "issues": issues, "grade": grade, "usage": usage}


def cost(results: list[dict], kind: str, model_name: str) -> float:
    p_in, p_out = PRICES.get(model_name, (0.0, 0.0))
    total = 0.0
    for r in results:
        for k, u in r["usage"]:
            if k == kind:
                total += u["input_tokens"] * p_in / 1e6 + u["output_tokens"] * p_out / 1e6
    return total


def write_report(results: list[dict], args, gen_model: str, judge_model: str) -> str:
    n = len(results)
    grounded = [r for r in results if not r["issues"]]
    graded = [r for r in results if r["grade"]]
    lines = [
        "# Commentary eval",
        "",
        f"{n} moments sampled from the held-out test matches (seed {args.seed}). "
        + ("**Dry run**: template lines, no API calls." if args.dry_run
           else f"Generator: `{gen_model}` (low effort). Judge: `{judge_model}`."),
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Passes deterministic grounding checks | {len(grounded)}/{n} ({len(grounded)/n:.0%}) |",
    ]
    if graded:
        faithful = sum(r["grade"]["faithful"] for r in graded)
        reacts = sum(r["grade"]["reacts_to_event"] for r in graded)
        tone = np.mean([r["grade"]["tone"] for r in graded])
        both = sum(1 for r in graded if r["grade"]["faithful"] and not r["issues"])
        lines += [
            f"| Judge: faithful to the facts | {faithful}/{len(graded)} ({faithful/len(graded):.0%}) |",
            f"| Judge: reacts to the latest event | {reacts}/{len(graded)} ({reacts/len(graded):.0%}) |",
            f"| Judge: caster tone (1-5) | {tone:.2f} |",
            f"| Grounded **and** judged faithful | {both}/{len(graded)} ({both/len(graded):.0%}) |",
        ]
    if not args.dry_run:
        lines += [
            f"| Cost: generation | ${cost(results, 'gen', gen_model):.3f} "
            f"(${cost(results, 'gen', gen_model) / n * 1000:.2f} per 1,000 lines) |",
            f"| Cost: judging | ${cost(results, 'judge', judge_model):.3f} |",
        ]
    lines += [
        "",
        "In production, lines that fail the grounding checks are replaced by a template line, "
        "so users only see lines that pass.",
    ]

    failures = [r for r in results if r["issues"] or (r["grade"] and not r["grade"]["faithful"])]
    if failures:
        lines += ["", "## Failures (up to 15)", "", "| Line | Grounding issues | Judge |", "|---|---|---|"]
        for r in failures[:15]:
            judge_note = "; ".join(r["grade"]["unsupported_claims"]) if r["grade"] else ""
            lines.append(f"| {r['line']} | {'; '.join(r['issues'])} | {judge_note} |".replace("\n", " "))

    lines += ["", "## Samples", "", "| Event | State | Line |", "|---|---|---|"]
    for r in results[:10]:
        f = r["facts"]
        state = (f"{f['attackers_alive']}v{f['defenders_alive']}, atk {f['attacker_win_pct']}%"
                 + (f", planted {f['plant_site']}" if f["spike_planted"] else ""))
        lines.append(f"| {f['event']} | {state} | {r['line']} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="no API calls; use the template line")
    args = ap.parse_args()

    client = None
    if not args.dry_run:
        import anthropic
        client = anthropic.Anthropic()
    from src.commentary.generate import MODEL as gen_model
    from src.commentary.judge import JUDGE_MODEL as judge_model

    model = NumpyLSTM.from_npz(ROOT / "artifacts/models/best_weights.npz")
    moments = sample_moments(args.n, args.seed)
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda m: evaluate_one(m, model, client, not args.no_judge), moments))

    report = write_report(results, args, gen_model, judge_model)
    out = ROOT / ("artifacts/commentary_eval_dryrun.md" if args.dry_run else "artifacts/commentary_eval.md")
    out.write_text(report)
    print(report)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
