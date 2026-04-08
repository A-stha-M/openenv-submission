---
title: Cold-Chain Dispatch Environment
emoji: 🚚
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
base_path: /web
---

# Cold-Chain Dispatch Environment

This OpenEnv server simulates real-world refrigerated logistics dispatch with deterministic dynamics and programmatic grading.

## Server Setup
### Docker (Recommended)
```bash
cd my_env
docker build -t cold-chain-env:latest .
docker run --rm -p 8000:8000 cold-chain-env:latest
curl http://localhost:8000/health
```

### Without Docker
```bash
cd my_env
python -m venv venv
source venv/bin/activate
pip install -r server/requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Endpoints
- POST /reset
- POST /step
- GET /state
- GET /schema
- GET /health
- GET /docs
- WebSocket /ws

## Tasks
- cold_chain_easy
- cold_chain_medium
- cold_chain_hard
- cold_chain_vaccine_urgent
- cold_chain_grid_outage

## Notes
- Deterministic scenarios for reproducible benchmarking.
- Typed action and observation models in models.py.
- openenv.yaml defines metadata and task descriptions.
