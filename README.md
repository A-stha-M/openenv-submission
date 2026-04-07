# Cold-Chain Dispatch Benchmark (OpenEnv)

## Environment Description and Motivation
This project implements a real-world cold-chain logistics environment for evaluating agentic decision-making under operational constraints.

The setting models a refrigerated truck dispatch workflow where an agent must balance:
- delivery timeliness,
- cargo thermal safety,
- fuel economy,
- compressor wear,
- compliance risk (thermal excursion budget),
- route discipline (avoid indecisive switching).

This domain is practical for RL and LLM-agent evaluation because real operators routinely make these trade-offs and mistakes in production logistics systems.

## Why This Is Useful for Agent Evaluation
The benchmark is designed to evaluate capabilities that matter in operational autonomy:
- Long-horizon planning under coupled constraints (temperature, traffic, fuel, deadlines).
- Recovery behavior after degradation (repair-hub detour vs direct route commitment).
- Compliance-aware control (excursion budgets rather than single-threshold binary checks).
- Policy discipline (looping and unstable rerouting are penalized).

## OpenEnv Spec Compliance
- Typed models: [my_env/models.py](my_env/models.py)
- Environment API (`reset`, `step`, `state`): [my_env/server/my_env_environment.py](my_env/server/my_env_environment.py)
- Manifest and task metadata: [my_env/openenv.yaml](my_env/openenv.yaml)
- Baseline script at project root: [inference.py](inference.py)

## Action Space
Per step action (JSON):
- `target_hub`: `Destination` or `Repair_Hub`
- `cooling_power`: float in [0.0, 1.0]
- `speed_kmh`: float in [40.0, 120.0]

## Observation Space
Per step observation fields:
- `task_name`
- `current_location`
- `cargo_temp_celsius`
- `fuel_level_percent`
- `ambient_temp_celsius`
- `distance_to_destination_km`
- `distance_to_emergency_hub_km`
- `cooling_unit_health`
- `hours_elapsed`
- `delivery_deadline_hours`
- `urgency_index`
- `cargo_quality_index`
- `compliance_index`
- `excursion_hours`
- `excursion_budget_hours`
- `route_switch_count`
- `task_score`
- `reward`, `done`

## Task Suite (5 Tasks, Deterministic)
This submission includes 5 deterministic tasks (3+ requirement exceeded):

1. `cold_chain_easy` (Easy)
- Mild weather and short route.
- Focus: stable operation and basic efficiency.

2. `cold_chain_medium` (Medium)
- Hotter corridor and partial cooling degradation.
- Focus: balancing thermal control and fuel.

3. `cold_chain_hard` (Hard)
- Long route with severe heat and traffic waves.
- Focus: long-horizon strategic control.

4. `cold_chain_vaccine_urgent` (Hard)
- Tight deadline and low excursion tolerance.
- Focus: schedule-critical delivery under strict compliance.

5. `cold_chain_grid_outage` (Very Hard)
- Outage-like heat spikes and weak initial cooling health.
- Focus: recovery strategy and compliance-preserving control.

## Grader Design and Quality
### Score Range
`task_score` is deterministic and clamped to [0.0, 1.0].

### Determinism and Reproducibility
Each task uses fixed per-step heat and traffic profiles. No stochastic sampling is used in environment dynamics or grading.

### Multi-Objective Scoring
Final task score combines weighted components:
- completion progress,
- cargo quality,
- fuel efficiency,
- policy discipline,
- schedule adherence,
- compliance index (excursion budget usage).

### Hardness Signal
The hard-tier tasks (`cold_chain_hard`, `cold_chain_vaccine_urgent`, `cold_chain_grid_outage`) are intentionally challenging for frontier models because single-rule policies (for example, always max cooling/speed) usually violate fuel or compliance constraints before reaching optimal outcomes.

## Reward Function
Reward is dense and trajectory-aware:
- positive shaping for useful progress,
- stability bonus for safe/efficient operating bands,
- penalties for unsafe behavior, indecisive rerouting, and overusing excursion budget,
- success bonus linked to final grader score.

Undesirable behavior (loops, unsafe thermal behavior, severe compliance overrun) is explicitly penalized.

## Setup and Usage
### 1) Install Dependencies
```bash
pip install -r requirements.txt
```

### 2) Local Server Run
```bash
cd my_env
uv run server
```

### 3) Baseline Inference
Required variables:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Run:
```bash
set HF_TOKEN=your_token
set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
python inference.py
```

### 4) Docker Build and Run
```bash
docker build -t cold-chain-env -f my_env/server/Dockerfile my_env
docker run -p 8000:8000 cold-chain-env
```

### 5) OpenEnv Validation
```bash
cd my_env
openenv validate
```

## Baseline Scores
Latest measured run from [inference.py](inference.py) with `MODEL_NAME=Qwen/Qwen2.5-7B-Instruct`:

| Task | Baseline Score |
|------|----------------|
| cold_chain_easy | 0.908 |
| cold_chain_medium | 0.574 |
| cold_chain_hard | 0.289 |
| cold_chain_vaccine_urgent | 0.891 |
| cold_chain_grid_outage | 0.228 |

## Hugging Face Space
Deployed Space:
- https://huggingface.co/spaces/Astha28/openenv-cold-chain

## Prevalidation Script Usage
For the organizer script layout, use `./my_env` as `repo_dir`:

```bash
bash ./validate-submission.sh https://astha28-openenv-cold-chain.hf.space ./my_env
```

Local dry run:
```bash
docker build -t cold-chain-local-check my_env
docker run -d --name cold-chain-local-check -p 8010:8000 cold-chain-local-check
bash ./validate-submission.sh http://localhost:8010 ./my_env
docker rm -f cold-chain-local-check
```
