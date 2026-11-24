"""
Latency benchmarks for win-probability inference.

1. In-process: the numpy serving model vs. the PyTorch model it was exported
   from, for several sequence lengths (one request = one round so far).
2. HTTP: end-to-end POST /predict against a local uvicorn server.
3. Optional: the same HTTP benchmark against a deployed URL (--url).

Run:
    python3 scripts/benchmark_latency.py
    python3 scripts/benchmark_latency.py --url https://valorant-predictor-production.up.railway.app

Writes artifacts/latency.md.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
import numpy as np

from src.serving.numpy_lstm import NumpyLSTM

ROOT = Path(__file__).parent.parent
SEQ_LENGTHS = [1, 5, 10, 14]


def _pcts(samples_s: list[float]) -> tuple[float, float, float]:
    ms = np.array(samples_s) * 1000
    return tuple(float(x) for x in np.percentile(ms, [50, 95, 99]))


def _time(fn, runs: int, warmup: int = 50) -> list[float]:
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def bench_in_process(runs: int) -> list[tuple]:
    import torch
    from src.training.train import WinProbLSTM

    torch.set_num_threads(1)  # a serving worker handles one request at a time
    ckpt = torch.load(ROOT / "artifacts/models/best.pt", map_location="cpu", weights_only=False)
    torch_model = WinProbLSTM(ckpt["config"]).eval()
    torch_model.load_state_dict(ckpt["model_state"])
    np_model = NumpyLSTM.from_npz(ROOT / "artifacts/models/best_weights.npz")

    rows = []
    rng = np.random.default_rng(0)
    for T in SEQ_LENGTHS:
        seq = rng.random((T, 15), dtype=np.float32)
        x, l = torch.tensor(seq[None]), torch.tensor([T])

        def run_torch():
            with torch.no_grad():
                torch_model.forward_all(x, l)

        rows.append((T, _pcts(_time(lambda: np_model.predict_all(seq), runs)), _pcts(_time(run_torch, runs))))
    return rows


def _payload(T: int) -> dict:
    snap = dict(attackers_alive=5, defenders_alive=5, spike_planted=0, time_elapsed_ms=0,
                time_remaining_ms=100000, plant_time_ms=0, attacker_economy=20000,
                defender_economy=20000, round_num=5, attacker_score=3, defender_score=2, plant_site="")
    snaps = []
    for t in range(T):
        snaps.append({**snap, "time_elapsed_ms": t * 6000, "time_remaining_ms": 100000 - t * 6000,
                      "attackers_alive": max(1, 5 - t // 3), "defenders_alive": max(1, 5 - (t + 1) // 3)})
    return {"snapshots": snaps}


def bench_http(base_url: str, runs: int) -> list[tuple]:
    rows = []
    with httpx.Client(base_url=base_url, timeout=30) as client:
        for T in SEQ_LENGTHS:
            body = _payload(T)

            def call():
                r = client.post("/predict", json=body)
                r.raise_for_status()

            rows.append((T, _pcts(_time(call, runs, warmup=5))))
    return rows


def start_local_server(port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "scripts.serve:app", "--port", str(port), "--log-level", "warning"],
        cwd=ROOT,
    )
    for _ in range(100):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                return proc
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    proc.kill()
    raise RuntimeError("local server did not start")


def _fmt(p: tuple[float, float, float]) -> str:
    return " | ".join(f"{v:.2f}" for v in p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--http-runs", type=int, default=300)
    ap.add_argument("--url", help="also benchmark a deployed server (fewer runs: 50)")
    args = ap.parse_args()

    lines = [
        "# Inference latency",
        "",
        f"Machine: {platform.machine()} / {platform.system()} / Python {platform.python_version()}. "
        "Times in milliseconds. One request carries the round so far (T snapshots).",
        "",
        f"## In-process model call ({args.runs} runs each, single thread)",
        "",
        "| T | numpy p50 | numpy p95 | numpy p99 | PyTorch p50 | PyTorch p95 | PyTorch p99 |",
        "|---|---|---|---|---|---|---|",
    ]
    for T, np_p, torch_p in bench_in_process(args.runs):
        lines.append(f"| {T} | {_fmt(np_p)} | {_fmt(torch_p)} |")

    proc = start_local_server(port=8765)
    try:
        local = bench_http("http://127.0.0.1:8765", args.http_runs)
    finally:
        proc.terminate()
    lines += ["", f"## HTTP `POST /predict`, local uvicorn ({args.http_runs} runs each)", "",
              "| T | p50 | p95 | p99 |", "|---|---|---|---|"]
    lines += [f"| {T} | {_fmt(p)} |" for T, p in local]

    if args.url:
        remote = bench_http(args.url, 50)
        lines += ["", f"## HTTP `POST /predict`, deployed ({args.url}, 50 runs each)", "",
                  "Includes network round trip from the benchmark machine.", "",
                  "| T | p50 | p95 | p99 |", "|---|---|---|---|"]
        lines += [f"| {T} | {_fmt(p)} |" for T, p in remote]

    out = ROOT / "artifacts/latency.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
