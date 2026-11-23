# Valorant In-Round Win Probability Predictor

**[Live Demo →](https://valorant-predictor-production.up.railway.app)**

A live win probability model for Valorant that updates dynamically after each kill — the same concept used in professional esports broadcast overlays. Given the current round state (players alive, spike planted, time remaining, economy), the model outputs the probability that the attacking team wins.

---

## Results

All numbers are on **147 held-out matches** (3,064 rounds), scored at every **live state**: each snapshot where both teams still have a player alive (22,439 states). Final snapshots where one side has been wiped are excluded because they give the answer away.

| Model | Accuracy | Log-loss | Brier |
|---|---|---|---|
| Logistic regression, round-start snapshot only | 59.8% | 0.665 | 0.236 |
| Logistic regression, current snapshot only | 74.5% | 0.502 | 0.168 |
| **2-layer LSTM, full kill-by-kill history** | **74.9%** | **0.490** | **0.164** |

In-round information is worth about 15 points of accuracy over pre-round prediction. The LSTM's history adds only a small further gain (log-loss 0.502 → 0.490). Most of the signal is in the *current* state: who's alive, economy, and spike status.

### Calibration
The model's probabilities track observed win rates closely across the full range. When it says 75%, attackers win about 75% of the time. For a live win-probability overlay that matters more than raw accuracy.

> **Earlier versions of this README reported 99% accuracy.** That number came from evaluation leakage: the model was scored on the round's final state, which usually shows one team fully eliminated, and a one-line "who got wiped?" rule gets 98.5% on those rounds. The parser also inverted attack/defense sides and exposed the spike plant time before the plant happened. All three are fixed; see `src/training/train.py` and `src/ingestion/parser.py`.

<p align="center">
  <img src="assets/calibration_curve.png" width="420"/>
  <img src="assets/feature_importance.png" width="420"/>
</p>

<p align="center">
  <img src="assets/timestep_accuracy.png" width="420"/>
  <img src="assets/trajectories.png" width="420"/>
</p>

### Feature Importance
Permutation importance on live states: **players alive dominates**, followed by **team economy**, which still matters mid-round because it decides what weapons and utility the survivors have. Time, spike status and score add comparatively little once alive counts and economy are known.

---

## Architecture

```
Henrik Unofficial Valorant API
         │
         ▼
  Batch Fetcher           981 matches, snowball sampled
  (fetch_matches.py)      20,802 labeled rounds
         │
         ▼
  Parser                  Henrik JSON → RoundSnapshot sequences
  (parser.py)             One snapshot per kill event per round
         │
         ▼
  Feature Engineering     15-dim normalized vector per timestep
  (sequences.py)          Padded to (N=20802, T_max=13, F=15)
         │
         ▼
  2-layer LSTM            PackedSequence → hidden state → sigmoid
  (train.py)              BCELoss, Adam, best checkpoint saved
         │
         ▼
  FastAPI endpoint        POST /predict → win probability
  (serve.py)              Per-timestep probabilities included
```

---

## Data

Match data was collected from Henrik's Unofficial Valorant API using a **snowball sampling** strategy:

1. Started from a set of seed players (known accounts)
2. Downloaded their 20 most recent competitive matches
3. Extracted all 10 players from each match
4. Fetched those players' match histories in turn

This produced **981 unique competitive matches** across NA servers with no duplicates (deduplication by match ID). Each match was saved as raw JSON and parsed into **RoundSnapshot** sequences — one snapshot per kill event per round, capturing:

- Players alive on each team
- Whether the spike is planted and on which site
- Time elapsed / time remaining (switches to 45s defuse clock after plant)
- Team economy at round start
- Round number and current score

This yielded **20,696 labeled rounds** (surrendered rounds excluded) (~7 snapshots each) as the training dataset.

---

## Training Details

- **Split**: 85% train / 15% validation, split by **match**: all rounds from a given match go entirely into train or entirely into val (834 train matches / 147 val matches).
- **Supervision**: the model emits a win probability after every snapshot, and the loss is applied at every timestep. This matches how it is used live, where it only ever sees the round so far.
- **Validation metric**: log-loss on live states only (both teams have a player alive), so the decided final state can't inflate the score.
- **Loss**: Binary Cross-Entropy
- **Optimizer**: Adam, learning rate 1e-3
- **Checkpointing**: best checkpoint saved by live-state validation loss
- **Architecture**: 2-layer LSTM, hidden dim 128, dropout 0.3, sigmoid output head
- **Sequences**: Padded to max length (T=13), PackedSequence used so the LSTM ignores padding

---

## Quick Start

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Set up API keys**
```bash
cp .env.example .env
# Fill in RIOT_API_KEY (developer.riotgames.com)
# Fill in HENRIK_API_KEY (docs.henrikdev.xyz)
```

**3. Collect match data**
```bash
python3 -m src.ingestion.fetch_matches --seeds "PlayerName#TAG"
```

**4. Build features**
```bash
python3 scripts/build_features.py
```

**5. Train the model**
```bash
python3 scripts/train_model.py
```

**6. Evaluate**
```bash
python3 -c "
from src.training.evaluate import evaluate
evaluate('artifacts/models/best.pt', 'data/processed/dataset.npz')
"
```

**7. Serve**
```bash
python3 scripts/serve.py
```

---

## Docker

Train the model first (`scripts/train_model.py`), then run the API in a container:

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`.

> **Note:** The model checkpoint (`artifacts/models/best.pt`) is not tracked in git due to file size. You must train the model locally before building the Docker image.

---

## API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "snapshots": [
      {
        "attackers_alive": 5, "defenders_alive": 5,
        "spike_planted": 0, "time_elapsed_ms": 0,
        "time_remaining_ms": 100000, "plant_time_ms": 0,
        "attacker_economy": 15000, "defender_economy": 20000,
        "round_num": 5, "attacker_score": 3, "defender_score": 2,
        "plant_site": ""
      },
      {
        "attackers_alive": 4, "defenders_alive": 4,
        "spike_planted": 1, "time_elapsed_ms": 35000,
        "time_remaining_ms": 45000, "plant_time_ms": 35000,
        "attacker_economy": 15000, "defender_economy": 20000,
        "round_num": 5, "attacker_score": 3, "defender_score": 2,
        "plant_site": "B"
      }
    ]
  }'
```

```json
{
  "win_probability": 0.5506,
  "per_timestep": [0.5595, 0.5506]
}
```

---

## Project Structure

```
├── src/
│   ├── ingestion/
│   │   ├── henrik_client.py     Rate-limited Henrik API client
│   │   ├── fetch_matches.py     Batch match downloader (snowball sampling)
│   │   └── parser.py            Match JSON → RoundSnapshot sequences
│   ├── features/
│   │   └── sequences.py         Feature engineering, normalization, padding
│   └── training/
│       ├── config.py            Hyperparameter dataclass
│       ├── train.py             LSTM model + training loop
│       └── evaluate.py          Calibration, timestep accuracy, feature importance
├── scripts/
│   ├── build_features.py        CLI: build dataset.npz
│   ├── train_model.py           CLI: train LSTM
│   ├── baseline.py              Logistic regression baseline comparison
│   └── serve.py                 FastAPI prediction server
├── tests/                       23 pytest tests (parser, features, API)
└── assets/                      Evaluation plots
```

---

## Tech Stack

Python · PyTorch · FastAPI · scikit-learn · NumPy · httpx · Pydantic
