# Inference latency

Machine: arm64 / Darwin / Python 3.11.6. Times in milliseconds. One request carries the round so far (T snapshots).

## In-process model call (1000 runs each, single thread)

| T | numpy p50 | numpy p95 | numpy p99 | PyTorch p50 | PyTorch p95 | PyTorch p99 |
|---|---|---|---|---|---|---|
| 1 | 0.03 | 0.04 | 0.04 | 0.09 | 0.09 | 0.10 |
| 5 | 0.14 | 0.14 | 0.15 | 0.18 | 0.19 | 0.20 |
| 10 | 0.25 | 0.28 | 0.29 | 0.29 | 0.30 | 0.33 |
| 14 | 0.35 | 0.36 | 0.37 | 0.37 | 0.39 | 0.41 |

## HTTP `POST /predict`, local uvicorn (300 runs each)

| T | p50 | p95 | p99 |
|---|---|---|---|
| 1 | 0.78 | 1.04 | 1.09 |
| 5 | 0.87 | 1.01 | 1.08 |
| 10 | 1.09 | 1.40 | 1.79 |
| 14 | 1.44 | 1.94 | 2.61 |
