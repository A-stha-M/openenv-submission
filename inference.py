# Copyright (c) Meta Platforms, Inc. and affiliates.
import asyncio
import os
import sys
import textwrap
import json

import requests
from openai import OpenAI

# ── Environment Variables ──────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

# Your running HF Space — the environment server
ENV_URL = os.getenv("ENV_URL", "https://astha28-openenv-cold-chain.hf.space")

print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", flush=True)
print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)
print(f"[DEBUG] API_KEY_set={bool(API_KEY)}", flush=True)
print(f"[DEBUG] ENV_URL={ENV_URL}", flush=True)

TASKS = [
    "cold_chain_easy",
    "cold_chain_medium",
    "cold_chain_hard",
    "cold_chain_vaccine_urgent",
    "cold_chain_grid_outage",
]

SYSTEM_PROMPT = textwrap.dedent("""
You are an AI cold-chain dispatch planner.
Objective: maximize final task_score while safely reaching Destination.

Environment mechanics:
- Ambient heat and traffic change each step.
- Excessive cooling can damage compressor health and increase fuel burn.
- Thermal excursion hours are tracked against a scenario-specific compliance budget.
- Repair_Hub can restore cooling health/fuel, but route switching is penalized.
- Task score blends completion, cargo quality, fuel, schedule, discipline, and compliance.

Strategy hints:
- Keep cargo in ~1.4C to 4.2C zone and protect compliance budget.
- Prefer smooth policy (avoid frequent target switching).
- Use Repair_Hub only when cooling_health drops below 0.4 or fuel is critically low.

Output ONLY valid JSON with no markdown fences:
{"target_hub": "Destination", "cooling_power": <float 0.0-1.0>, "speed_kmh": <float 40.0-120.0>}
""").strip()


# ── Logging ────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={'true' if done else 'false'} error={str(error) if error else 'null'}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    print(
        f"[END] success={'true' if success else 'false'} steps={steps} "
        f"score={score:.2f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )


# ── HTTP Environment Client ────────────────────────────────────────────

def env_reset(task_name: str) -> dict:
    """Call /reset on the HF Space environment server."""
    resp = requests.post(
        f"{ENV_URL}/reset",
        json={"task_name": task_name},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()

def env_step(action_dict: dict) -> dict:
    """Call /step on the HF Space environment server."""
    resp = requests.post(
        f"{ENV_URL}/step",
        json={"action": action_dict},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ── Helpers ────────────────────────────────────────────────────────────

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def normalize_action(payload: dict) -> dict:
    target = payload.get("target_hub", "Destination")
    if target not in ["Destination", "Repair_Hub"]:
        target = "Destination"
    cooling = clamp(float(payload.get("cooling_power", 0.7)), 0.0, 1.0)
    speed   = clamp(float(payload.get("speed_kmh", 75.0)), 40.0, 120.0)
    return {
        "target_hub":    target,
        "cooling_power": round(cooling, 3),
        "speed_kmh":     round(speed, 2),
    }

def call_model(client: OpenAI, prompt: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.1,
        max_tokens=200,
    )
    text = response.choices[0].message.content.strip()
    if "```" in text:
        parts = text.split("```")
        text  = parts[1] if len(parts) > 1 else parts[0]
        if "```" in text:
            text = text.split("```")[0]
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON found in response: {text!r}")
    return json.loads(text[start : end + 1])

def extract_obs(response: dict) -> dict:
    """Pull observation fields from /reset or /step response."""
    obs = response.get("observation", response)
    return obs


# ── Task Runner ────────────────────────────────────────────────────────

async def run_task(task_name: str) -> float:
    # ✅ OpenAI client uses evaluator-injected API_KEY and API_BASE_URL
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    rewards     = []
    steps_taken = 0
    score       = 0.0
    success     = False

    log_start(task_name, "cold_chain_logistics", MODEL_NAME)

    # Reset via HTTP — no local environment import
    reset_resp = env_reset(task_name)
    obs = extract_obs(reset_resp)

    for step in range(1, 21):
        user_prompt = (
            f"Task={task_name}; "
            f"location={obs.get('current_location', 'En_Route')}; "
            f"temp={obs.get('cargo_temp_celsius', 2.0):.2f}; "
            f"fuel={obs.get('fuel_level_percent', 100.0):.2f}; "
            f"ambient={obs.get('ambient_temp_celsius', 25.0):.2f}; "
            f"dist_dest={obs.get('distance_to_destination_km', 0.0):.2f}; "
            f"dist_hub={obs.get('distance_to_emergency_hub_km', 0.0):.2f}; "
            f"cooling_health={obs.get('cooling_unit_health', 1.0):.3f}; "
            f"deadline={obs.get('delivery_deadline_hours', 8)}; "
            f"elapsed={obs.get('hours_elapsed', 0)}; "
            f"urgency={obs.get('urgency_index', 0.0):.3f}; "
            f"quality={obs.get('cargo_quality_index', 1.0):.3f}; "
            f"compliance={obs.get('compliance_index', 1.0):.3f}; "
            f"excursions={obs.get('excursion_hours', 0)}/{obs.get('excursion_budget_hours', 2)}; "
            f"switches={obs.get('route_switch_count', 0)}; "
            f"score={obs.get('task_score', 0.0):.3f}"
        )

        try:
            parsed      = call_model(client, user_prompt)
            action_dict = normalize_action(parsed)
            error       = None
        except Exception as e:
            print(f"[DEBUG] LLM call failed step={step} type={type(e).__name__} msg={e}", flush=True)
            action_dict = {"target_hub": "Destination", "cooling_power": 0.75, "speed_kmh": 75.0}
            error       = str(e)

        # Step via HTTP
        try:
            step_resp = env_step(action_dict)
        except Exception as e:
            print(f"[DEBUG] env_step failed step={step}: {e}", flush=True)
            break

        obs    = extract_obs(step_resp)
        reward = float(step_resp.get("reward", obs.get("reward", 0.0)))
        done   = bool(step_resp.get("done", obs.get("done", False)))
        score  = clamp(float(obs.get("task_score", 0.0)), 0.0, 1.0)

        rewards.append(reward)
        steps_taken = step

        log_step(step, json.dumps(action_dict), reward, done, error)

        if done:
            success = obs.get("current_location") == "Destination"
            break

    log_end(success, steps_taken, score, rewards)
    return score


# ── Main ───────────────────────────────────────────────────────────────

async def main() -> None:
    scores = {}
    for task in TASKS:
        scores[task] = await run_task(task)
    print("\nFINAL SCORES", flush=True)
    for t, s in scores.items():
        print(f"{t}: {s:.2f}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())