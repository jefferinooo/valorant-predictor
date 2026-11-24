"""
FastAPI serving endpoint for the LSTM win probability model.

Run:
    python3 scripts/serve.py

Then call it:
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
          }
        ]
      }'
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.features.sequences import _snapshot_to_vector
from src.serving.numpy_lstm import NumpyLSTM
from src.commentary.facts import build_facts
from src.commentary.ratelimit import RateLimiter

# ---------------------------------------------------------------------------
# Load model at startup
# ---------------------------------------------------------------------------

# The model is trained in PyTorch; serving runs the exported weights in numpy
# (src/serving/numpy_lstm.py) so the production image doesn't need torch.
WEIGHTS = Path(__file__).parent.parent / "artifacts" / "models" / "best_weights.npz"

if not WEIGHTS.exists():
    raise RuntimeError(f"Weights not found at {WEIGHTS}. Run scripts/train_model.py (or scripts/export_weights.py).")

model = NumpyLSTM.from_npz(WEIGHTS)

# LLM commentary is optional: enabled only when an Anthropic API key is set.
COMMENTARY_ENABLED = bool(os.environ.get("ANTHROPIC_API_KEY"))
commentary_client = None
if COMMENTARY_ENABLED:
    import anthropic
    commentary_client = anthropic.Anthropic()
commentary_limiter = RateLimiter(
    per_minute=int(os.environ.get("COMMENTARY_PER_MINUTE", "10")),
    daily_cap=int(os.environ.get("COMMENTARY_DAILY_CAP", "500")),
)

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

app = FastAPI(title="Valorant Win Probability API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Snapshot(BaseModel):
    attackers_alive: int = Field(..., ge=0, le=5)
    defenders_alive: int = Field(..., ge=0, le=5)
    spike_planted: int = Field(..., ge=0, le=1)
    time_elapsed_ms: int = Field(..., ge=0)
    time_remaining_ms: int = Field(..., ge=0)
    plant_time_ms: int = Field(0, ge=0)
    attacker_economy: int = Field(..., ge=0)
    defender_economy: int = Field(..., ge=0)
    round_num: int = Field(..., ge=0)
    attacker_score: int = Field(..., ge=0)
    defender_score: int = Field(..., ge=0)
    plant_site: str = Field("", pattern=r"^[ABC]?$")


class PredictRequest(BaseModel):
    snapshots: list[Snapshot] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    win_probability: float
    per_timestep: list[float]


class CommentaryResponse(BaseModel):
    commentary: str
    source: str          # "llm" or "template"
    rejected: list[str]  # why an LLM line was replaced by the template, if it was


@app.get("/health")
def health():
    return {"status": "ok", "commentary": COMMENTARY_ENABLED}


def _probabilities(request: PredictRequest) -> list[float]:
    vectors = [_snapshot_to_vector(s.model_dump()) for s in request.snapshots]
    return [float(p) for p in model.predict_all(vectors)]


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if len(request.snapshots) == 0:
        raise HTTPException(status_code=400, detail="snapshots list must not be empty")

    per_timestep = [round(p, 4) for p in _probabilities(request)]

    return PredictResponse(
        win_probability=per_timestep[-1],
        per_timestep=per_timestep,
    )


@app.post("/commentary", response_model=CommentaryResponse)
def commentary(request: PredictRequest, http_request: Request):
    """One line of caster commentary on the latest event, grounded in the model's output."""
    if not COMMENTARY_ENABLED:
        raise HTTPException(status_code=503, detail="commentary is not enabled on this server")
    # Behind Railway's proxy the client IP is the first X-Forwarded-For entry
    forwarded = http_request.headers.get("x-forwarded-for", "")
    client_ip = forwarded.split(",")[0].strip() or (http_request.client.host if http_request.client else "unknown")
    if not commentary_limiter.allow(client_ip):
        raise HTTPException(status_code=429, detail="commentary rate limit reached; try again shortly")

    from src.commentary.generate import grounded_commentary  # imports anthropic

    facts = build_facts([s.model_dump() for s in request.snapshots], _probabilities(request))
    return CommentaryResponse(**grounded_commentary(facts, commentary_client))


# Serve frontend
from fastapi.responses import FileResponse

_frontend = Path(__file__).parent.parent / "frontend" / "index.html"

@app.get("/")
def serve_frontend():
    if _frontend.exists():
        return FileResponse(_frontend)
    return {"message": "Frontend not found. See /predict for API."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
