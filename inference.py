# Copyright (c) Meta Platforms, Inc. and affiliates.
import asyncio
import os
import sys
import textwrap
import json
from typing import Dict, List, Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from my_env.server.my_env_environment import MyEnvironment
from my_env.models import MyEnvAction

# Default is safe for model selection only; API endpoint/key must come from evaluator env.
MODEL_NAME_DEFAULT = "Qwen/Qwen2.5-7B-Instruct"

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


# -- Logging helpers ---------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_str = str(error) if error is not None else "null"
    done_str = "true" if done else "false"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# -- Core helpers ------------------------------------------------------

def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def heuristic_action(obs) -> Dict[str, float | str]:
    """Fallback policy when model JSON fails."""
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


def get_model_action(client: OpenAI, user_prompt: str, model_name: str) -> Optional[Dict[str, object]]:
    """Call the LLM and return parsed JSON action dict, or None on failure."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.1,
        )
        raw = (completion.choices[0].message.content or "{}").strip()
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return None

    # Strip markdown fences if present
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:]) if len(lines) >= 2 else raw
        if raw.endswith("```"):
            raw = raw[:-3].strip()
    if raw.lower().startswith("json"):
        raw = raw[4:].strip()

    try:
        return json.loads(raw)
    except Exception as exc:
        print(f"[DEBUG] JSON parse failed: {exc} | raw: {raw[:200]}", flush=True)
        return None


# -- Task runner -------------------------------------------------------

async def run_task(client: OpenAI, env: MyEnvironment, task_name: str, model_name: str) -> float:
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=task_name, env="cold_chain_logistics", model=model_name)

    try:
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

            # Always call the model first - proxy must receive calls
            parsed = get_model_action(client, user_prompt, model_name)
            if parsed is not None:
                action_dict = normalize_action(parsed, obs)
            else:
                action_dict = heuristic_action(obs)

            action_raw = json.dumps(action_dict, separators=(",", ":"))

            action = MyEnvAction(**action_dict)
            obs = env.step(action)

            reward = obs.reward
            done = obs.done
            score = _clamp(float(getattr(obs, "task_score", 0.0)), 0.0, 1.0)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_raw, reward=reward, done=done, error=None)

            if done:
                success = obs.current_location == "Destination"
                break

    except Exception as e:
        log_step(step=steps_taken + 1, action="error", reward=0.0, done=True, error=str(e))
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# -- Entry point -------------------------------------------------------

async def main() -> None:
    try:
        client = OpenAI(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["API_KEY"],
        )
    except KeyError as exc:
        raise RuntimeError(f"Missing required environment variable: {exc.args[0]}") from exc

    model_name = os.getenv("MODEL_NAME", MODEL_NAME_DEFAULT)

    # Ensure at least one authenticated request goes through the injected proxy credentials.
    _ = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Reply with exactly: ok"},
        ],
        temperature=0.0,
    )

    env = MyEnvironment()

    for task in TASKS:
        await run_task(client, env, task, model_name)


if __name__ == "__main__":
    asyncio.run(main())
