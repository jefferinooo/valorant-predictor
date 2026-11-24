# Commentary eval

200 moments sampled from the held-out test matches (seed 0). Generator: `claude-opus-5-5` (low effort). Judge: `claude-opus-5-5`.

| Metric | Result |
|---|---|
| Passes deterministic grounding checks | 192/200 (96%) |
| Judge: faithful to the facts | 199/200 (100%) |
| Judge: reacts to the latest event | 200/200 (100%) |
| Judge: caster tone (1-5) | 3.90 |
| Grounded **and** judged faithful | 191/200 (96%) |
| Cost: generation | $0.599 ($3.00 per 1,000 lines) |
| Cost: judging | $1.237 |

In production, lines that fail the grounding checks are replaced by a template line, so users only see lines that pass.

## Failures (up to 15)

| Line | Grounding issues | Judge |
|---|---|---|
| Attackers lose one and slide 17 points to just 32 percent, now down 3v4 with 68 seconds left and no spike down. | claims 'spike down' but the spike isn't planted |  |
| Another attacker drops, it's 2v5 with 76 seconds left, and the attack slides 4 points to just 1% as defenders smell a 7th round.  Wait—"7th round" predicts a future score. Revise:  Another attacker falls, it's 2v5 with 76 seconds left, and the attack slumps 4 points to a near-hopeless 1%. | more than one line; longer than 40 words |  |
| Spike goes down on A with 45 seconds left, a tense 2v2, and the attackers still hold 57 percent despite slipping 2 points! | says where a kill happened ('down on A'), which the facts don't include |  |
| Defender down, it's a 2v1 with 60 seconds left, and the attackers surge 28 points to 83% as they eye a 10-3 lead! | score '10-3' isn't the current score 9-3 |  |
| Attackers draw first blood back to level it 4v4, and that pick swings them up 16 points to a 57 percent favorite with 72 seconds left! |  | "draw first blood": at 4v4 an attacker had already died, so this kill is not the round's first blood |
| Another defender drops and it's 4v1 with 67 seconds left, attackers surge 8 points to 99% as they chase 2-2! | score '2-2' isn't the current score 1-2 |  |
| Another defender drops and it's 5v3 with 90 seconds left, attackers surging to 96% as they push for an 11-5 lead!  Wait—"push for an 11-5 lead" predicts a future score. Fix:  Another defender goes down, it's 5v3 with 90 seconds left, and the attackers leading 10-5 rocket up 8 points to 96%! | more than one line; longer than 40 words; score '11-5' isn't the current score 10-5; score '11-5' isn't the current score 10-5 |  |
| Spike goes down on A with 45 seconds left, and the attackers climb 7 points to 34% despite trailing 2v3 at 9-12! | says where a kill happened ('down on A'), which the facts don't include |  |
| Attackers strike back to even it at 3v3, win chances jumping 17 points to 35%, but only 25 seconds remain with no spike down! | claims 'spike down' but the spike isn't planted |  |

## Samples

| Event | State | Line |
|---|---|---|
| attacker_killed | 1v3, atk 6% | Another attacker drops and it's 1v3 now, win chances crash 14 points to just 6 percent with 75 seconds left! |
| attacker_killed | 3v3, atk 45% | Attackers lose a body and it's level at 3v3, their chances sliding 18 points to 45% with 79 seconds left at 7-9! |
| defender_killed | 2v1, atk 81% | Huge pick for the attackers, a 2v1 now with 57 seconds left, and their win chance rockets up 32 points to 81%! |
| attacker_killed | 3v5, atk 19% | Another attacker drops, it's 3v5 now and the attack's chances crater by 17 points to just 19 percent with 72 seconds left! |
| spike_planted | 3v3, atk 50%, planted A | Spike down on A at 6-6, and the 3v3 retake is a true coin flip, attackers up 4 points to 50 with 45 seconds left! |
| attacker_killed | 3v4, atk 32% | Attackers lose one and slide 17 points to just 32 percent, now down 3v4 with 68 seconds left and no spike down. |
| defender_killed | 4v4, atk 53% | Defender down, it's 4v4 with 79 seconds left, and the attackers surge 16 points to a 53 percent edge at 10-7! |
| defender_killed | 4v4, atk 53% | Defender down, it's 4v4 with 72 seconds left, and the attackers surge 16 points to 53% in this 4-4 deadlock! |
| defender_killed | 5v4, atk 65% | First blood to the attackers, 5v4 with 79 seconds left, and their win chances jump 20 points to 65 percent! |
| defender_killed | 2v4, atk 7% | Attackers find a pick to make it 2v4, but at just 7 percent with 90 seconds left, they've still got a mountain to climb. |
