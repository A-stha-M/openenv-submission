# OpenEnv Submission: Cold-Chain Dispatch Under Thermal Risk

## Why This Environment Matters
Real cold-chain dispatch is a high-stakes operational task, not a toy path-finding problem.

A dispatcher must continuously trade off:
- on-time delivery vs fuel burn,
- cargo safety vs compressor wear,
- route commitment vs emergency repair detours.

This environment models those real dispatch tensions with deterministic, reproducible dynamics that are suitable for agent benchmarking.

## OpenEnv Compliance
- Typed Pydantic models for action/observation/reward in [my_env/models.py](my_env/models.py).
- Full environment interface with reset/step/state in [my_env/server/my_env_environment.py](my_env/server/my_env_environment.py).
- OpenEnv manifest with task metadata in [my_env/openenv.yaml](my_env/openenv.yaml).
- Baseline script named [inference.py](inference.py) at repository root.

## Action Space
Agent outputs one JSON action per step:
- target_hub: Destination or Repair_Hub
- cooling_power: float in [0.0, 1.0]
- speed_kmh: float in [40.0, 120.0]

## Observation Space
Environment returns typed state each step:
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
- route_switch_count
- task_score
- done, reward

## Creativity and Novel Mechanics
This version adds deterministic mechanics designed to improve novelty and reduce exploitability:
- Heatwave pulses: each task has a fixed ambient heat profile, creating timed windows for risk.
- Traffic waves: deterministic congestion multipliers create non-stationary route efficiency.
- Compressor wear: sustained aggressive cooling degrades cooling_unit_health over time.
- Repair hub trade-off: visiting Repair_Hub restores cooling health and fuel, but repeated visits and route-switching are penalized.
- Dispatch discipline scoring: indecisive target switching and low-progress stalling reduce outcome quality.

These mechanics force multi-step strategy rather than one-shot max-speed/max-cooling heuristics.

## Reward Design (Dense + Terminal)
Reward is meaningful throughout the trajectory:
- positive progress reward, urgency-aware for destination progress,
- thermal stability and efficient operation-band bonuses,
- penalties for unsafe temperature/fuel states,
- penalties for route switching and loop-like behavior,
- terminal success bonus tied to deterministic task_score.

Undesirable behavior is explicitly discouraged while still giving partial-progress signal.

## Task Suite and Deterministic Graders
Three tasks are included (easy -> medium -> hard), each with deterministic weather/traffic profiles and grader weights:

1. cold_chain_easy
- Shorter route, milder thermal pressure.
- Teaches stable operation bands and basic energy management.

2. cold_chain_medium
- Longer route, hotter profile, partial cooling wear.
- Strategic one-time repair-hub use becomes useful.

3. cold_chain_hard
- Long route, severe heat/traffic pulses, tight schedule pressure.
- Requires long-horizon policy discipline and risk-aware adaptation.

Task score is deterministic in [0.0, 1.0] and blends:
- completion,
- cargo quality,
- fuel efficiency,
- policy discipline,
- schedule adherence.

## Baseline Inference
Baseline runner is [inference.py](inference.py) and uses the OpenAI client API path as required.

Required environment variables:
- API_BASE_URL
- MODEL_NAME
- HF_TOKEN

Also supported:
- OPENAI_API_KEY (fallback if HF_TOKEN is not set)

The script emits strict structured logs using:
- [START]
- [STEP]
- [END]

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run baseline:

```bash
set HF_TOKEN=your_token
set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
python inference.py
```

## Docker
Build and run locally:

```bash
docker build -t cold-chain-env -f my_env/server/Dockerfile my_env
docker run -p 8000:8000 cold-chain-env
```

## Hugging Face Space Deployment
This environment is designed to run as a Docker HF Space and tagged with openenv in the Space README metadata.

## Baseline Score Table
Fill the table below from a real run of [inference.py](inference.py) before final submission:

| Task | Score (0.0-1.0) | Notes |
|------|------------------|-------|
| cold_chain_easy | 0.892 | Reached Destination in 5 steps. |
| cold_chain_medium | 0.611 | Reached Destination in 9 steps under high heat. |
| cold_chain_hard | 0.255 | Failed at step 2 (spoilage), leaving headroom for stronger policies. |

## Pre-Submission Checklist
- HF Space deploys and /reset responds.
- openenv validate passes for [my_env/openenv.yaml](my_env/openenv.yaml).
- docker build + docker run works with [my_env/server/Dockerfile](my_env/server/Dockerfile).
- [inference.py](inference.py) runs end-to-end and emits strict log format.
- Three tasks are listed and graded in [0.0, 1.0].

## Running The Organizer Prevalidation Script
Because this repository stores the OpenEnv package under [my_env](my_env), run the script with repo_dir set to ./my_env.

```bash
./validate-submission.sh https://<your-space>.hf.space ./my_env
```

Optional local dry-run before deploying (uses local container instead of HF URL):

```bash
docker build -t cold-chain-local-check my_env
docker run -d --name cold-chain-local-check -p 8010:8000 cold-chain-local-check
./validate-submission.sh http://localhost:8010 ./my_env
docker rm -f cold-chain-local-check
```