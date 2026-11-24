# Commentary eval

200 moments sampled from the held-out test matches (seed 0). Generator: `claude-opus-5-5` (low effort). Judge: `claude-opus-5-5`.

| Metric | Result |
|---|---|
| Passes deterministic grounding checks | 198/200 (99%) |
| **Faithful lines shown to users** (after template fallback) | **199/200 (100%)** |
| Faithful lines replaced by the template (false alarms) | 2/200 |
| Judge: faithful to the facts | 199/200 (100%) |
| Judge: reacts to the latest event | 200/200 (100%) |
| Judge: caster tone (1-5) | 3.84 |
| Grounded **and** judged faithful | 197/200 (98%) |
| Cost: generation | $0.595 ($2.97 per 1,000 lines) |
| Cost: judging | $1.223 |

In production, lines that fail the grounding checks are replaced by a template line, so users only see lines that pass.

## Failures (up to 15)

| Line | Grounding issues | Judge |
|---|---|---|
| Attackers draw first blood to make it 4v4, and their win chances leap 16 points to 57% with 72 seconds still on the clock! |  | "draw first blood": 4v4 means an attacker had already died, so this kill was not first blood |
| A defender drops, it's 4v3 with 60 seconds left, and the attackers surge 12 points to 88% as they chase a 6-1 lead.  Wait—"chase a 6-1 lead" predicts future score. Must avoid.  Defender down, now 4v3 with 60 seconds left, and the attackers leading 5-1 jump 12 points to an 88% win chance! | more than one line; longer than 40 words; score '6-1' isn't the current score 5-1; score '6-1' isn't the current score 5-1 |  |
| Another defender drops, it's 5v3 with 71 seconds left, and the attackers surge to 97% as they push toward an 11-8 lead! | score '11-8' isn't the current score 10-8 |  |

## Samples

| Event | State | Line |
|---|---|---|
| attacker_killed | 1v3, atk 6% | Another attacker drops and it's 1v3 with 75 seconds left, attacker chances crashing 14 points to just 6 percent! |
| attacker_killed | 3v3, atk 45% | Attacker down and it's 3v3 now, win chances sliding 18 points to 45% with 79 seconds left, and defenders hold the edge at 7-9. |
| defender_killed | 2v1, atk 81% | Defender goes down and it's 2v1 now, attackers surge 32 points to 81% with 57 seconds left at 10-8! |
| attacker_killed | 3v5, atk 19% | Another attacker drops, and it's 3v5 with 72 seconds left as attacker hopes crash 17 points to just 19 percent! |
| spike_planted | 3v3, atk 50%, planted A | Spike down on A at 3v3, and the attackers climb 4 points to a dead-even 50 percent with 45 seconds left at 6-6! |
| attacker_killed | 3v4, atk 32% | An attacker goes down and it's 3v4 now, chances crashing 17 points to just 32% with 68 seconds left on the clock! |
| defender_killed | 4v4, atk 53% | Attackers strike back to even it at 4v4, surging 16 points to 53 percent with 79 seconds left on the clock! |
| defender_killed | 4v4, atk 53% | A defender goes down and it's 4v4 now, attackers surge 16 points to 53% with 72 seconds left in this 4-4 thriller! |
| defender_killed | 5v4, atk 65% | First blood to the attackers, 5v4 with 79 seconds left, and their win chance jumps 20 points to 65 percent! |
| defender_killed | 2v4, atk 7% | Attackers finally strike back, 2v4 now and win odds tick up 3 points to 7 percent, with 90 seconds left to mount a miracle. |
