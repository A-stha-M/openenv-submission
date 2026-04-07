---
title: Cold-Chain Dispatch Environment
emoji: "🚚"
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# Cold-Chain Dispatch Environment

This package contains a deterministic OpenEnv environment for refrigerated logistics dispatch.

## Core Idea
The agent controls route target, cooling power, and speed for a truck carrying temperature-sensitive goods. The policy must deliver on time while preventing spoilage and avoiding fuel exhaustion.

## Endpoints
- POST /reset
- POST /step
- GET /state
- GET /schema
- GET /health
- GET /docs
- WebSocket /ws

## Run Locally

```bash
docker build -t cold-chain-env -f server/Dockerfile .
docker run -p 8000:8000 cold-chain-env
```

## OpenEnv Metadata
Environment manifest: openenv.yaml

Tasks:
- cold_chain_easy
- cold_chain_medium
- cold_chain_hard
- cold_chain_vaccine_urgent
- cold_chain_grid_outage

## Notes
The environment is deterministic for reproducible grading and benchmark comparisons.
