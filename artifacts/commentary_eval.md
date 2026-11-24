# Commentary eval

200 moments sampled from the held-out test matches (seed 0). Generator: `claude-opus-5-5` (low effort). Judge: `claude-opus-5-5`.

| Metric | Result |
|---|---|
| Passes deterministic grounding checks | 200/200 (100%) |
| Judge: faithful to the facts | 197/200 (98%) |
| Judge: reacts to the latest event | 200/200 (100%) |
| Judge: caster tone (1-5) | 4.03 |
| Grounded **and** judged faithful | 197/200 (98%) |
| Cost: generation | $0.524 ($2.62 per 1,000 lines) |
| Cost: judging | $1.259 |

In production, lines that fail the grounding checks are replaced by a template line, so users only see lines that pass.

## Failures (up to 15)

| Line | Grounding issues | Judge |
|---|---|---|
| Another defender drops, it's 4v2 with 85 seconds left, and attackers surge 9 points to 96%, one round from closing this out at 11-2! |  | one round from closing this out at 11-2 |
| Defender down on B and the attackers surge 13 points to 78% — it's 3v3 with 32 seconds left, and the defenders are running out of time! |  | Defender down on B (the facts give no kill location, only that the spike was planted on B) |
| Spike down on A with 37 seconds left, and attackers holding a 3v2 now sit at 87 percent, up 3 to level this at 4-4! |  | "level this at 4-4" (the score is 3-4 and the round is still in progress) |

## Samples

| Event | State | Line |
|---|---|---|
| attacker_killed | 1v3, atk 6% | Another attacker falls, it's 1v3 with 75 seconds left, and attacker chances crash 14 points to just 6 percent! |
| attacker_killed | 3v3, atk 45% | An attacker goes down and it's 3v3 now, attackers sliding 18 points to 45% with 79 seconds left to find something. |
| defender_killed | 2v1, atk 81% | Defender down, and it's a 2v1 with 57 seconds left as the attackers surge 32 points to 81% to win this round! |
| attacker_killed | 3v5, atk 19% | Another attacker falls and it's 3v5 now, win chances plunging 17 points to just 19 percent with 72 seconds left! |
| spike_planted | 3v3, atk 50%, planted A | Spike down on A at 6-6, and the attackers climb 4 points to a dead-even 50% in this 3v3 with 45 seconds left! |
| attacker_killed | 3v4, atk 32% | Attackers lose one and the odds crash by 17 to just 32 percent, now down 3v4 with 68 seconds left to claw back. |
| defender_killed | 4v4, atk 53% | Defender down and it's 4v4 now, attackers surge 16 points to 53% with 79 seconds left as they push for an 11th round! |
| defender_killed | 4v4, atk 53% | A defender goes down and it's 4v4 with 72 seconds left, attackers surging 16 points to take a 53 percent edge! |
| defender_killed | 5v4, atk 65% | First blood to the attackers, a 5v4 edge with 79 seconds left vaults their chances up 20 points to 65 percent! |
| defender_killed | 2v4, atk 7% | A defender drops and the attackers claw back 3 points, but at 2v4 they still sit at just 7 percent with 90 seconds left! |
