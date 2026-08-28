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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.training.train import WinProbLSTM
from src.features.sequences import _snapshot_to_vector

# ---------------------------------------------------------------------------
# Load model at startup
# ---------------------------------------------------------------------------

CHECKPOINT = Path("artifacts/models/best.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not CHECKPOINT.exists():
    raise RuntimeError(f"Checkpoint not found at {CHECKPOINT}. Run scripts/train_model.py first.")

ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model = WinProbLSTM(ckpt["config"]).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if len(request.snapshots) == 0:
        raise HTTPException(status_code=400, detail="snapshots list must not be empty")

    vectors = [_snapshot_to_vector(s.model_dump()) for s in request.snapshots]
    seq = torch.tensor([vectors], dtype=torch.float32).to(device)  # (1, T, 15)
    length = torch.tensor([len(vectors)], dtype=torch.long).to(device)

    per_timestep = []
    with torch.no_grad():
        for t in range(1, len(vectors) + 1):
            sub = seq[:, :t, :]
            l = torch.tensor([t], dtype=torch.long).to(device)
            p = model(sub, l).item()
            per_timestep.append(round(p, 4))

    return PredictResponse(
        win_probability=per_timestep[-1],
        per_timestep=per_timestep,
    )


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
