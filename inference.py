# Copyright (c) Meta Platforms, Inc. and affiliates.
import asyncio
import os
import textwrap
import json
from typing import Dict, Optional
from openai import OpenAI

# Direct import guarantees state preservation
from my_env.server.my_env_environment import MyEnvironment
from my_env.models import MyEnvAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # Optional when using from_docker_image()

TASKS = [
    "cold_chain_easy",
    "cold_chain_medium",
    "cold_chain_hard",
    "cold_chain_vaccine_urgent",
    "cold_chain_grid_outage",
]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI cold-chain dispatch planner.
    Objective: maximize final task_score while safely reaching Destination.

    Environment mechanics:
    - Ambient heat and traffic change each step.
    - Excessive cooling can damage compressor health and increase fuel burn.
    - Thermal excursion hours are tracked against a scenario-specific compliance budget.
    - Repair_Hub can restore cooling health/fuel, but route switching and repeated hub visits are penalized.
    - Task score blends completion, cargo quality, fuel, schedule, discipline, and compliance.

    Strategy hints:
    - Keep cargo in ~1.4C to 4.2C zone and protect compliance budget.
    - Prefer smooth policy (avoid frequent target switching).
    - Use Repair_Hub only when risk is rising (hot cargo + low cooling health + long distance left).

    Output ONLY valid JSON. Do not include markdown blocks like ```json.
    Format exactly like this but change the values based on the status:
    {"target_hub": "Destination", "cooling_power": <float>, "speed_kmh": <float>}
    """
).strip()


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def heuristic_action(obs) -> Dict[str, float | str]:
    """Fallback policy to keep baseline reproducible even when model JSON fails."""
    danger_temp = obs.cargo_temp_celsius >= 4.4
    low_fuel = obs.fuel_level_percent <= 16.0
    weak_cooling = obs.cooling_unit_health <= 0.58
    far_remaining = obs.distance_to_destination_km >= 220.0
    compliance_tight = obs.compliance_index <= 0.5 or (obs.excursion_budget_hours - obs.excursion_hours) <= 1

    should_repair = (
        far_remaining
        and obs.distance_to_emergency_hub_km <= 160.0
        and (weak_cooling or (danger_temp and low_fuel))
        and obs.route_switch_count < 2
    )

    if danger_temp or compliance_tight:
        cooling = 0.94
        speed = 84.0 if not low_fuel else 72.0
    elif obs.urgency_index >= 0.72:
        cooling = 0.84
        speed = 90.0 if obs.fuel_level_percent > 24.0 else 78.0
    else:
        cooling = 0.66 if obs.cargo_temp_celsius < 3.5 else 0.76
        speed = 76.0 if obs.fuel_level_percent > 18.0 else 68.0

    target = "Repair_Hub" if should_repair else "Destination"
    return {
        "target_hub": target,
        "cooling_power": round(_clamp(cooling, 0.0, 1.0), 3),
        "speed_kmh": round(_clamp(speed, 40.0, 120.0), 2),
    }


def normalize_action(payload: Dict[str, object], obs) -> Dict[str, float | str]:
    """Constrain model output to strict schema and safe bounds."""
    fallback = heuristic_action(obs)

    target = payload.get("target_hub", fallback["target_hub"])
    if target not in {"Destination", "Repair_Hub"}:
        target = fallback["target_hub"]

    try:
        cooling = float(payload.get("cooling_power", fallback["cooling_power"]))
    except (TypeError, ValueError):
        cooling = float(fallback["cooling_power"])

    try:
        speed = float(payload.get("speed_kmh", fallback["speed_kmh"]))
    except (TypeError, ValueError):
        speed = float(fallback["speed_kmh"])

    return {
        "target_hub": target,
        "cooling_power": round(_clamp(cooling, 0.0, 1.0), 3),
        "speed_kmh": round(_clamp(speed, 40.0, 120.0), 2),
    }


def request_model_action_json(client: OpenAI, user_prompt: str) -> Optional[Dict[str, object]]:
    """Attempt strict JSON response first, then fallback to a minimal chat call."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    raw_content: Optional[str] = None
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        raw_content = completion.choices[0].message.content or "{}"
    except Exception:
        # Fallback call avoids optional params that may be rejected by some providers.
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
            )
            raw_content = completion.choices[0].message.content or "{}"
        except Exception:
            return None

    try:
        return json.loads(raw_content)
    except Exception:
        return None

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

async def run_task(client, env, task_name):
    rewards = []
    steps_taken = 0
    success = False
    score = 0.0
    
    log_start(task_name, "cold_chain_logistics", MODEL_NAME)

    try:
        # Reset environment with the specific task
        obs = env.reset(task_name=task_name)
        score = _clamp(float(getattr(obs, "task_score", 0.0)), 0.0, 1.0)
        
        for step in range(1, 21):
            user_prompt = (
                f"Task={task_name}; location={obs.current_location}; temp={obs.cargo_temp_celsius:.2f}; "
                f"fuel={obs.fuel_level_percent:.2f}; ambient={obs.ambient_temp_celsius:.2f}; "
                f"dist_dest={obs.distance_to_destination_km:.2f}; dist_hub={obs.distance_to_emergency_hub_km:.2f}; "
                f"cooling_health={obs.cooling_unit_health:.3f}; deadline={obs.delivery_deadline_hours}; "
                f"elapsed={obs.hours_elapsed}; urgency={obs.urgency_index:.3f}; quality={obs.cargo_quality_index:.3f}; "
                f"compliance={obs.compliance_index:.3f}; excursions={obs.excursion_hours}/{obs.excursion_budget_hours}; "
                f"switches={obs.route_switch_count}; score={obs.task_score:.3f}"
            )
            
            action_dict = heuristic_action(obs)

            parsed = request_model_action_json(client, user_prompt)
            if parsed is not None:
                action_dict = normalize_action(parsed, obs)

            action_raw = json.dumps(action_dict, separators=(",", ":"))
            
            # Convert raw JSON to Pydantic Action and step the environment
            action = MyEnvAction(**action_dict)
            obs = env.step(action)
            
            # The observation object directly holds the reward and done state
            reward = obs.reward
            done = obs.done
            score = _clamp(float(getattr(obs, "task_score", 0.0)), 0.0, 1.0)
            
            rewards.append(reward)
            steps_taken = step
            
            log_step(step, action_raw, reward, done, None)
            
            if done:
                success = obs.current_location == "Destination"
                break

    except Exception as e:
        log_step(steps_taken+1, "error", 0.0, True, str(e))
    finally:
        log_end(success, steps_taken, score, rewards)

async def main():
    try:
        client = OpenAI(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["API_KEY"],
        )
    except KeyError as exc:
        raise RuntimeError(f"Missing required environment variable: {exc.args[0]}") from exc

    # Mandatory proxy warm-up: fail early if the injected API proxy is unreachable.
    try:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "Reply with exactly: ok"},
            ],
        )
    except Exception as exc:
        raise RuntimeError(
            "LLM proxy warm-up failed. Ensure evaluator-injected API_BASE_URL and API_KEY are used."
        ) from exc

    env = MyEnvironment() # Create exactly ONE truck
    
    for task in TASKS:
        await run_task(client, env, task)

if __name__ == "__main__":
    asyncio.run(main())