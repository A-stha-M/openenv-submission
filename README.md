# OpenEnv: Cold Chain Logistics Environment 🚚❄️

## Motivation & Problem Statement
In the real world, supply chain logistics is not just about finding the shortest path; it is about resource management under harsh, dynamic conditions. Human dispatchers managing refrigerated trucks (Cold Chains) face a constant, stressful optimization problem:

- Drive faster to arrive sooner? *You burn fuel exponentially.*
- Crank up the AC to fight a 40°C heatwave? *You drain the fuel tank faster.*
- Slow down to save gas? *The cargo might spoil in the heat.*

This OpenEnv benchmark simulates this exact physical dilemma. The agent acts as an AI Dispatcher that must output JSON commands (`target_hub`, `cooling_power`, `speed_kmh`) at every time step to safely route a truck to its destination without the cargo temperature exceeding **5.0°C** and without the fuel hitting **0%**.

---

## State & Action Space

### Observation Space (State)
- `cargo_temp_celsius`: Current temp (Critical limit: 5.0°C)
- `fuel_level_percent`: Remaining fuel (Critical limit: 0.0%)
- `distance_to_destination_km`: Target to reach 0
- `ambient_temp_celsius`: External weather
- `cooling_unit_health`: Simulator modifier (1.0 is healthy, lower means AC is failing)
- `current_location`: Textual status (e.g., `"En_Route"`, `"Spoiled"`, `"Destination"`)

### Action Space (JSON)
- `target_hub` *(str)*: e.g., `"Destination"` or `"Repair_Hub"`
- `cooling_power` *(float, 0.0 - 1.0)*: Intensity of refrigeration
- `speed_kmh` *(float, 0.0 - 120.0)*: Velocity

---

## Task Difficulties

We implemented **3 deterministic grading tasks**:

### 1. `cold_chain_easy`
- Short distance (300 km)
- Mild weather (25°C)
- Easily solvable by maxing out speed and cooling

### 2. `cold_chain_medium`
- Medium distance (400 km)
- Hot weather (35°C)
- Partial cooling unit failure

### 3. `cold_chain_hard`
- Extreme distance (600 km)  
- Severe weather (40°C)  
- Heavy traffic  

**Note:** This task is designed to be highly resistant to zero-shot LLM prompting. Baseline agents tend to maximize cooling to fight the heat, which inevitably leads to running out of fuel (**Stranded**) before reaching the destination.

---

## Setup & Validation

This project is built to the exact Meta OpenEnv specifications.

### 1. Local Docker Build

```bash
docker build -t my_env -f server/Dockerfile .
docker run -p 8000:8000 my_env
```

### 2. Automated Validation

Run Meta's validation script against the Hugging Face Space:

```bash
./validate-submission.sh <YOUR_HF_SPACE_URL>
```

---

## Running the Baseline Inference

The baseline uses **Qwen/Qwen2.5-7B-Instruct** via the Hugging Face Inference API.

### Note on Model Selection
We deliberately selected the **7B model** instead of the **72B model**. Logistics loops require up to **15 continuous steps**. Using 72B rapidly depletes Hugging Face free-tier credits, resulting in **HTTP 402: Payment Required** errors mid-episode.

The **7B model** ensures the baseline is highly reproducible for evaluators without rate-limiting.

### Run Commands

```bash
# 1. Export required credentials
export HF_TOKEN="your_hf_read_token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"

# 2. Run the evaluator
python inference.py
```

---

## Baseline Results

| Task | Result | Score |
|------|--------|-------|
| Easy | Success | ~0.88 |
| Medium | Success | ~0.78 |
| Hard | Failed | 0.00 |

**Hard Task Failure Reason:**  
The agent anchors to high cooling/speed and runs out of fuel on **Step 8**. Left as a challenge for advanced RL agents!