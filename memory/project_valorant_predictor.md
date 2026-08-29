---
name: project-valorant-predictor
description: Valorant in-round live win probability predictor — project goals, approach, and API setup status
metadata:
  type: project
---

Goal: Predict win probability as a Valorant round progresses — spike planted, players alive, time remaining — updating after each kill event. Model justification: true temporal sequence within a round (LSTM/Transformer). This is Option 3 (hardest, most impressive).

Data source: Riot Games API (`val/match/v1/matches/{matchId}`). Post-game match JSON contains per-kill timestamps within each round, letting us reconstruct the in-round state at each kill event.

**Why:** More impressive and technically justified than pre-round prediction. Real broadcast overlays use this type of live win probability.

**How to apply:** All data decisions should serve in-round sequence modeling. Feature vectors encode: players alive (both sides), spike planted, time elapsed/remaining, economy, round/score context.

Key files built:
- `config/settings.py` — pydantic-settings, reads RIOT_API_KEY/RIOT_REGION/RIOT_CLUSTER from .env
- `src/ingestion/riot_client.py` — rate-limited sync HTTP client (sliding window, 18 req/s / 90 req/2min)
- `src/ingestion/fetch_matches.py` — batch fetcher seeded from player Riot IDs; saves raw JSON to data/raw/matches/
- `src/ingestion/parser.py` — converts match JSON into RoundSnapshot sequences (one snapshot per kill event per round)

To get an API key: developer.riotgames.com — dev key is free, rate-limited. Production key needs application approval.

Usage to start collecting data:
```bash
cp .env.example .env  # then fill in RIOT_API_KEY
pip install -r requirements.txt
python3 -m src.ingestion.fetch_matches --seeds "PlayerName#TAG"
```
