# Commentary eval

200 moments sampled from the held-out test matches (seed 0). **Dry run**: template lines, no API calls.

| Metric | Result |
|---|---|
| Passes deterministic grounding checks | 200/200 (100%) |

In production, lines that fail the grounding checks are replaced by a template line, so users only see lines that pass.

## Samples

| Event | State | Line |
|---|---|---|
| attacker_killed | 1v3, atk 6% | Attacker down: 1v3, attackers at 6% to win the round. |
| attacker_killed | 3v3, atk 45% | Attacker down: 3v3, attackers at 45% to win the round. |
| defender_killed | 2v1, atk 81% | Defender down: 2v1, attackers at 81% to win the round. |
| attacker_killed | 3v5, atk 19% | Attacker down: 3v5, attackers at 19% to win the round. |
| spike_planted | 3v3, atk 50%, planted A | Spike planted on A: 3v3, attackers at 50% to win the round. |
| attacker_killed | 3v4, atk 32% | Attacker down: 3v4, attackers at 32% to win the round. |
| defender_killed | 4v4, atk 53% | Defender down: 4v4, attackers at 53% to win the round. |
| defender_killed | 4v4, atk 53% | Defender down: 4v4, attackers at 53% to win the round. |
| defender_killed | 5v4, atk 65% | Defender down: 5v4, attackers at 65% to win the round. |
| defender_killed | 2v4, atk 7% | Defender down: 2v4, attackers at 7% to win the round. |
