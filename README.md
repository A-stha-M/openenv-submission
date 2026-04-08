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

## Environment Overview and Motivation
This benchmark simulates refrigerated logistics dispatch, a real operational task where humans make high-stakes decisions every hour. The agent controls routing, cooling, and speed while balancing:
- perishable cargo safety,
- fuel constraints,
- traffic delays,
- refrigeration degradation,
- compliance risk from thermal excursion windows,
- deadline pressure.

The environment is designed for evaluating long-horizon planning, policy discipline, and safe recovery behavior under coupled constraints.

## Functional Requirements Checklist
- Real-world task simulation: Yes (cold-chain dispatch operations).
- OpenEnv compliance: Yes (typed models, reset/step/state implementation, openenv.yaml, openenv validate passing).
- 3+ tasks with deterministic graders: Yes (5 tasks, deterministic scoring in [0.0, 1.0]).
- Meaningful dense reward: Yes (progress shaping, stability bonuses, behavior penalties, terminal reward).
- Baseline inference script: Yes (root [inference.py](inference.py), OpenAI client, required env vars).

## Action Space
Action fields (typed in [my_env/models.py](my_env/models.py)):
- target_hub: Destination or Repair_Hub
- cooling_power: float in [0.0, 1.0]
- speed_kmh: float in [40.0, 120.0]

## Observation Space
Observation fields (typed in [my_env/models.py](my_env/models.py)):
- task_name
- current_location
- cargo_temp_celsius
- fuel_level_percent
- ambient_temp_celsius
- distance_to_destination_km
- distance_to_emergency_hub_km
- cooling_unit_health
- hours_elapsed
- delivery_deadline_hours
- urgency_index
- cargo_quality_index
- compliance_index
- excursion_hours
- excursion_budget_hours
- route_switch_count
- task_score
- reward, done

## Task Suite and Expected Difficulty
Task metadata lives in [my_env/openenv.yaml](my_env/openenv.yaml).

1. cold_chain_easy (Easy)
- Mild weather, short route.
- Objective: learn stable operation and efficient progress.

2. cold_chain_medium (Medium)
- Hotter corridor and partial cooling degradation.
- Objective: preserve cargo while controlling fuel burn.

3. cold_chain_hard (Hard)
- Long route with severe heat/traffic pulses.
- Objective: long-horizon planning under tight resource coupling.

4. cold_chain_vaccine_urgent (Hard)
- Medical-priority route with tighter deadline and excursion tolerance.
- Objective: schedule-critical delivery with compliance awareness.

5. cold_chain_grid_outage (Very Hard)
- Outage-like heat spikes and weak cooling health.
- Objective: robust recovery policy under compliance pressure.

## Grader Design (Deterministic, Reproducible)
The programmatic grader computes task_score in [0.0, 1.0], combining:
- completion progress,
- cargo quality,
- fuel efficiency,
- policy discipline,
- schedule adherence,
- compliance index.

All scenarios use fixed step-wise heat and traffic profiles (no randomness), making results reproducible.

## Reward Design
Reward is dense and trajectory-aware:
- positive signal for incremental progress,
- bonus for stable safe operating bands,
- penalties for risky behavior (overheating, low fuel, indecisive rerouting, compliance overrun),
- terminal success bonus linked to final quality.

## Server Setup
### Docker (Recommended)
```bash
cd openenv-submission
docker build -t cold-chain-env:latest .
docker run --rm -p 8000:8000 cold-chain-env:latest
curl http://localhost:8000/health
```
On server health success response:
`{"status":"healthy"}`

### Without Docker
```bash
cd openenv-submission
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cd my_env
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## OpenEnv Validation
```bash
cd my_env
openenv validate
```

## Usage Instructions
### API Smoke Test
After starting the server, verify reset works:
```bash
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{}'
```

### Run a Single Task
Use the reset payload to select a task:
```bash
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{"task_name":"cold_chain_hard"}'
```

### Run Baseline Across All Tasks
```bash
cd openenv-submission
set HF_TOKEN=<your_token>
python inference.py
```

## Baseline Inference
Baseline script: [inference.py](inference.py)

Required environment variables:
- HF_TOKEN (required)
- API_BASE_URL (default provided)
- MODEL_NAME (default provided)

Run:
```bash
cd openenv-submission
set HF_TOKEN=<your_token>
set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
python inference.py
```

Output format follows strict [START]/[STEP]/[END] lines required by the hackathon validator.

## Baseline Performance Scores
| Task | Score | Note |
|------|-------|------|
| cold_chain_easy | 0.908 | Baseline succeeds quickly in stable conditions. |
| cold_chain_medium | 0.574 | Baseline reaches destination but loses efficiency under heat stress. |
| cold_chain_hard | 0.289 | Baseline usually fails early, showing long-horizon challenge value. |
| cold_chain_vaccine_urgent | 0.891 | Deadline-focused behavior can perform well when compliance budget is protected. |
| cold_chain_grid_outage | 0.228 | Outage scenario exposes weak recovery policies and raises benchmark difficulty. |

## Hugging Face Space
- Space URL: https://huggingface.co/spaces/Astha28/openenv-cold-chain

## Prevalidation Script
Run organizer script with environment directory as repo_dir:
```bash
bash ./validate-submission.sh https://astha28-openenv-cold-chain.hf.space ./my_env
```
